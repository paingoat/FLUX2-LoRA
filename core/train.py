"""Run the ai-toolkit EditLoRA job and keep a second progress bar in front of it.

The trainer already prints its own tqdm line. This wrapper records that output,
parses the step and loss, and draws one bar for the whole 1500-step run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    AITOOLKIT_DIR,
    BASE_REPO,
    OUTPUT_DIR,
    PREPARED_DIR,
    TEXT_ENCODER_REPO,
    TRAIN_CONFIG,
    TRANSFORMER_FILE,
    VAE_FILE,
    VAE_REPO,
    assert_not_old_removal,
    hub_snapshot_has,
    load_runtime_env,
)

STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
LOSS_RE = re.compile(r"\bloss:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
SAVE_RE = re.compile(r"Saving at step\s+(\d+)")
PATH_RE = re.compile(r"(?:folder_path|control_path):\s*\"([^\"]+)\"")
CTRL_RE = re.compile(r"--ctrl_img\s+([^\s\"]+)")
FIELD_RE = re.compile(r"(?m)^\s*(steps|save_every):\s*(\d+)\s*$")


def config_numbers(text: str) -> dict[str, int]:
    found = {key: int(value) for key, value in FIELD_RE.findall(text)}
    if "steps" not in found or "save_every" not in found:
        raise SystemExit(f"{TRAIN_CONFIG} must set train.steps and save.save_every")
    return found


def config_paths(text: str) -> list[Path]:
    paths = [Path(item) for item in PATH_RE.findall(text)]
    paths.extend(Path(item) for item in CTRL_RE.findall(text))
    if len(paths) < 2:
        raise SystemExit(f"{TRAIN_CONFIG} is missing folder_path or control_path")
    return paths


def require_ready(config_text: str) -> None:
    card = PREPARED_DIR / "dataset_card.json"
    if not card.is_file():
        raise SystemExit(f"missing {card}. Run core/prepare_dataset.py on the pod first.")
    source = assert_not_old_removal(Path(json.loads(card.read_text(encoding="utf-8"))["source"]))
    print(f"dataset source: {source}")

    for path in config_paths(config_text):
        checked = assert_not_old_removal(path)
        if not checked.exists():
            raise SystemExit(f"training path does not exist: {checked}")
        print(f"using: {checked}")

    missing = []
    if not hub_snapshot_has(BASE_REPO, TRANSFORMER_FILE):
        missing.append(f"{BASE_REPO}/{TRANSFORMER_FILE}")
    if not hub_snapshot_has(VAE_REPO, VAE_FILE):
        missing.append(f"{VAE_REPO}/{VAE_FILE}")
    if not hub_snapshot_has(TEXT_ENCODER_REPO, "config.json"):
        missing.append(TEXT_ENCODER_REPO)
    if missing:
        raise SystemExit("weights are not in HF_HOME. Run core/setup.sh first.\n  " + "\n  ".join(missing))
    run_py = AITOOLKIT_DIR / "run.py"
    if not run_py.is_file():
        raise SystemExit(f"missing {run_py}. Run core/setup.sh first.")


def existing_checkpoints() -> list[Path]:
    if not OUTPUT_DIR.is_dir():
        return []
    return sorted(OUTPUT_DIR.glob("*.safetensors"))


def parse_trainer_line(line: str) -> dict[str, int | float] | None:
    match = STEP_RE.search(line)
    if match is None or "loss:" not in line and "|" not in line:
        return None
    parsed: dict[str, int | float] = {"step": int(match.group(1)), "total": int(match.group(2))}
    loss_match = LOSS_RE.search(line)
    if loss_match is not None:
        parsed["loss"] = float(loss_match.group(1))
    return parsed


def stream_training(process: subprocess.Popen[bytes], log_path: Path, total: int, save_every: int) -> int:
    from tqdm import tqdm

    losses: deque[float] = deque(maxlen=20)
    bar = tqdm(total=total, unit="step", dynamic_ncols=True, file=sys.stdout)
    pending = b""
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            pending += chunk
            pieces = re.split(rb"[\r\n]", pending)
            pending = pieces.pop()
            for piece in pieces:
                line = piece.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                log.write(line + "\n")
                log.flush()
                parsed = parse_trainer_line(line)
                if parsed is None:
                    bar.write(line)
                    saved = SAVE_RE.search(line)
                    if saved:
                        bar.set_postfix_str(f"saved step {saved.group(1)}")
                    continue
                step = min(int(parsed["step"]), total)
                if "loss" in parsed:
                    losses.append(float(parsed["loss"]))
                delta = step - bar.n
                if delta > 0:
                    bar.update(delta)
                average = sum(losses) / len(losses) if losses else float("nan")
                next_save = ((max(step, 1) - 1) // save_every + 1) * save_every
                bar.set_postfix(loss=f"{average:.4f}", next_ckpt=min(next_save, total))
        if pending.strip():
            log.write(pending.decode("utf-8", errors="replace") + "\n")
    bar.close()
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TRAIN_CONFIG)
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in the output folder")
    args = parser.parse_args()

    load_runtime_env()
    config_text = args.config.read_text(encoding="utf-8")
    numbers = config_numbers(config_text)
    require_ready(config_text)

    checkpoints = existing_checkpoints()
    if checkpoints and not args.resume:
        print("checkpoints already exist:", file=sys.stderr)
        for path in checkpoints:
            print(f"  {path}", file=sys.stderr)
        print("re-run with --resume to continue. ai-toolkit loads the latest file automatically.", file=sys.stderr)
        return 1
    if args.resume and not checkpoints:
        print("warning: --resume was set, but no checkpoint exists. Starting from step 0.")

    python = Path(os.environ.get("KLEIN_PYTHON", "/workspace/miniconda3/envs/klein/bin/python"))
    if not python.is_file():
        raise SystemExit(f"missing trainer python: {python}. Run core/setup.sh and use the klein env.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "train.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = [str(python), "run.py", str(args.config.resolve())]
    print(" ".join(command))
    process = subprocess.Popen(
        command,
        cwd=AITOOLKIT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def stop(_signum: int, _frame: object) -> None:
        if process.poll() is None:
            print("\nstopping the trainer")
            os.killpg(process.pid, signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    code = stream_training(process, log_path, numbers["steps"], numbers["save_every"])
    print(f"trainer exit code: {code}")
    print(f"log: {log_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
