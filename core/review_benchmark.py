"""Create a blind, offline review page or summarize its exported CSV.

These are project-specific human criteria, not the official VIBE judge scores.
"""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
import random
from urllib.parse import quote

CRITERIA = ("removal_success", "marker_clear", "background_natural", "outside_preserved")
RUBRIC = {
    "removal_success": "All requested objects removed; no recognizable fragments remain.",
    "marker_clear": "No annotation strokes remain anywhere in the output.",
    "background_natural": "Restored background is plausible, without obvious seams or distortion.",
    "outside_preserved": "Unrequested scene content is preserved; no obvious geometry or color changes.",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def create_review(results: Path) -> None:
    rows = [row for row in read_csv(results) if row["status"] == "ok"]
    random.Random(42).shuffle(rows)
    output = results.parent
    review_csv = output / "review.csv"
    if not review_csv.exists():
        with review_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["review_id", *CRITERIA, "notes"])
            writer.writeheader()
            writer.writerows({"review_id": row["review_id"]} for row in rows)
    cards = []
    for row in rows:
        identity = html.escape(row["review_id"], quote=True)
        image_uri = quote(Path(row["sheet"]).resolve().relative_to(output.resolve()).as_posix())
        full_output = ""
        if row.get("image"):
            uri = quote(Path(row["image"]).resolve().relative_to(output.resolve()).as_posix())
            full_output = f'<p><a href="{uri}" target="_blank">Open full output</a></p>'
        selects = "".join(
            f'<label>{html.escape(key)} <select data-key="{key}">'
            '<option value="">Pending</option><option value="1">Pass</option>'
            '<option value="0">Fail</option></select></label>' for key in CRITERIA)
        cards.append(f'<article data-id="{identity}"><h2>{identity}</h2>'
                     f'<p>{html.escape(row["prompt"])}</p><a href="{image_uri}" target="_blank"><img src="{image_uri}"></a>{full_output}'
                     f'<div>{selects}</div><textarea data-key="notes" placeholder="Notes"></textarea></article>')
    rubric = "".join(f"<li><b>{key}</b>: {html.escape(value)}</li>" for key, value in RUBRIC.items())
    page = r'''<!doctype html><meta charset="utf-8"><title>Blind removal review</title>
<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;background:#f2f4f6;color:#202530}
article{background:white;margin:24px 0;padding:20px;border-radius:10px}img{width:100%;height:auto}
label{display:inline-block;margin:12px}textarea{display:block;width:95%;margin:12px}
button{padding:12px;cursor:pointer}header{position:sticky;top:0;background:#f2f4f6;padding:12px}</style>
<h1>Blind removal review</h1><p>Columns: original | reference | output | target (when available).
Judge the output; target similarity is not required for a plausible restoration.</p>
<ul>RUBRIC</ul><header><button id="export">Download review.csv</button>
<label>Import previous review.csv <input type="file" id="import" accept=".csv"></label>
<span id="progress"></span></header>CARDS
<script>
const keys = ['review_id', 'removal_success', 'marker_clear', 'background_natural', 'outside_preserved', 'notes'];
const cards = [...document.querySelectorAll('article')];
const stateKey = 'removal-review-' + cards.map(c => c.dataset.id).join(',');
function data() { return cards.map(c => Object.fromEntries(keys.map(k => [k, k === 'review_id' ? c.dataset.id : c.querySelector('[data-key="'+k+'"]').value]))); }
function update() {
  const rows = data(); localStorage.setItem(stateKey, JSON.stringify(rows));
  document.querySelector('#progress').textContent = rows.filter(r => keys.slice(1,5).every(k=>r[k] !== '')).length + '/' + rows.length + ' reviewed';
}
function restore(rows) {
  const byId = new Map(rows.map(r=>[r.review_id,r]));
  cards.forEach(c => { const r=byId.get(c.dataset.id); if(r) keys.slice(1).forEach(k=>{c.querySelector('[data-key="'+k+'"]').value=r[k]||'';}); });
  update();
}
function parseCSV(text) {
  const rows=[]; let row=[], value='', quoted=false;
  for(let i=0;i<text.length;i++) {
    const c=text[i];
    if(c==='"') {if(quoted && text[i+1]==='"'){value+='"';i++;}else quoted=!quoted;}
    else if(c===',' && !quoted){row.push(value);value='';}
    else if(c==='\n' && !quoted){row.push(value.replace(/\r$/, ''));rows.push(row);row=[];value='';}
    else value+=c;
  }
  if(value || row.length){row.push(value);rows.push(row);}
  const header=rows.shift().map(s=>s.replace(/^\uFEFF/,''));
  return rows.filter(r=>r.length===header.length).map(r=>Object.fromEntries(header.map((k,i)=>[k,r[i]])));
}
document.querySelector('#import').onchange=async e=>restore(parseCSV(await e.target.files[0].text()));
document.querySelector('#export').onclick=()=>{
  const quote=v=>'"'+String(v).replaceAll('"','""')+'"';
  const csv=[keys,...data().map(r=>keys.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='review.csv';a.click();URL.revokeObjectURL(url);
};
cards.forEach(c=>c.onchange=update);
try { restore(JSON.parse(localStorage.getItem(stateKey)||'[]')); } catch { update(); }
</script>'''
    (output / "review.html").write_text(page.replace("RUBRIC", rubric).replace("CARDS", "\n".join(cards)), encoding="utf-8")


def summarize(results: Path, review: Path) -> Path:
    rows = read_csv(results)
    ratings = read_csv(review)
    by_id = {}
    known = {row["review_id"] for row in rows if row["status"] == "ok"}
    for rating in ratings:
        identity = rating["review_id"]
        if identity not in known or identity in by_id:
            raise ValueError(f"unknown or duplicate review ID: {identity}")
        if any(rating.get(key, "") not in {"", "0", "1"} for key in CRITERIA):
            raise ValueError(f"{identity}: ratings must be blank, 0, or 1")
        by_id[identity] = rating
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in ("dataset", "split", "checkpoint", "resolution"))
        grouped.setdefault(key, []).append(row)
    lines = ["# Human removal evaluation", "",
             "Project-specific rubric; not official VIBE scores. Overall pass requires all four criteria.",
             "Rates below use fully reviewed successful generations. Pending reviews are not passes or failures.", "",
             "| Dataset | Split | Checkpoint | Resolution | Generated/attempted | Reviewed | Removal | Marker clear | Background | Preservation | Overall |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, items in sorted(grouped.items()):
        generated = [row for row in items if row["status"] == "ok"]
        reviewed = [by_id[row["review_id"]] for row in generated if row["review_id"] in by_id
                    and all(by_id[row["review_id"]].get(k) in {"0", "1"} for k in CRITERIA)]
        n = len(reviewed)
        counts = [sum(r[k] == "1" for r in reviewed) for k in CRITERIA]
        counts.append(sum(all(r[k] == "1" for k in CRITERIA) for r in reviewed))
        rates = [f"{count}/{n} ({100*count/n:.1f}%)" if n else "pending" for count in counts]
        lines.append("| " + " | ".join(map(str, key)) + f" | {len(generated)}/{len(items)} | {n} | " + " | ".join(rates) + " |")
    lines += ["", "Compare baseline and LoRA on the same images/seeds with complete reviews and matching coverage."]
    path = results.parent / "human_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--review", type=Path, help="summarize completed review.csv")
    args = parser.parse_args()
    if args.review:
        print(summarize(args.results, args.review))
    else:
        create_review(args.results)
        print(args.results.parent / "review.html")
