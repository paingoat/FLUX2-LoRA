---
name: EditLoRA scribble removal
overview: "Thiết kế và code toàn bộ thí nghiệm huấn luyện EditLoRA cho FLUX.2 [klein] base 4B (xóa vật thể theo scribble đỏ) trong core/, chạy trên RunPod: setup script, dataset prep có held-out split, 1 run train với progress bar, benchmark chọn checkpoint, và Gradio demo."
todos:
  - id: env-setup
    content: Viết core/env.sh, .env.example và core/setup.sh (miniconda, 2 conda env, clone ai-toolkit, prefetch weight vào /workspace/data)
    status: completed
  - id: dataset
    content: "Viết core/prepare_dataset.py: chỉ copy reference/target/txt từ data/removal_dataset, giữ 1 prompt chung, split từ meta.jsonl, từ chối path data/removal"
    status: completed
  - id: config
    content: Viết core/configs/removal_scribble_4b.yaml cho arch flux2_klein_4b + control_path
    status: completed
  - id: train
    content: "Viết core/train.py: kiểm tra tiền điều kiện, chạy ai-toolkit, progress bar tổng + log"
    status: completed
  - id: benchmark
    content: "Viết core/benchmark.py: held-out chỉ lấy ảnh trong removal_dataset, vùng đo là vòng tròn đỏ trên reference/, tính scribble_residue / preserve PSNR-SSIM / target LPIPS, xuất report + contact sheet"
    status: completed
  - id: gradio
    content: "Viết core/app.py: Gradio với brush đỏ, chọn checkpoint, before/after slider, gallery held-out"
    status: completed
  - id: guide
    content: Viết core/guide.md hướng dẫn setup -> train -> benchmark -> test Gradio + troubleshooting
    status: completed
isProject: false
---

# EditLoRA xóa vật thể theo scribble cho FLUX.2 [klein] base 4B

## Cơ sở từ blog (phần "Train an edit LoRA")

Blog nói rõ: edit LoRA = style LoRA + đúng 3 thay đổi, phần còn lại của config giữ nguyên.

- Dataset dạng cặp: `target/` (ảnh đích + caption `.txt`) và `reference/` (ảnh đầu vào), ghép theo tên file. 50-200 cặp là đủ.
- Caption là **câu lệnh biến đổi**, không phải mô tả ảnh. Không cần trigger word.
- Một dòng config: `control_path:` trỏ vào thư mục reference. Đây chính là thứ biến style LoRA thành edit LoRA.
- Mọi thứ khác giữ nguyên: `arch: "flux2_klein_4b"`, `lr: 1e-4`, `flowmatch`, train trên base-4B.
- Edit LoRA đạt đỉnh khoảng step 1000-1750; checkpoint mỗi 250 step rồi **chọn bằng mắt**, đừng tin loss.
- Train trên `FLUX.2-klein-base-4B`, nhưng inference trên bản distilled `FLUX.2-klein-4B` (4 step) cho kết quả tốt hơn và nhanh hơn.
- Cảnh báo của tác giả: LoRA của anh ấy chỉ có 3 cách diễn đạt caption, nên kém với prompt diễn đạt khác. Run này không theo khuyến nghị đó. Mọi `reference/` đều là vòng tròn đỏ và inference cũng dùng một câu, nên một prompt chung là đúng dữ liệu.

## Ràng buộc dữ liệu

Nguồn huấn luyện duy nhất là [data/removal_dataset/](data/removal_dataset/). [data/removal/](data/removal/) là tập cũ, không được dùng để train, sample, benchmark hay Gradio.

- `reference/` là ảnh đầu vào. Toàn bộ là vòng tròn đỏ. Đã xem `0001`, `0002`, `0005`, `0007`: `meta.jsonl` ghi các mẫu này là `cross` / `hatch` / `box`, nhưng ảnh trong `reference/` đều là ellipse đỏ. Nét hatch xanh nằm ở `data/removal/.../scribble.png`, không nằm trong dataset train.
- `target/` là ảnh đích: vật thể và vòng tròn đã mất.
- `target/*.txt` dùng chung một câu, và câu đó đúng với ảnh train: `Remove the object inside the red circle and naturally restore the background.`
- Giữ nguyên câu này cho cả 100 cặp và cho inference. Không sinh bank phrasing, không đổi "red circle" theo `scribble_style`.

`mask/` và trường `scribble_style` trong `meta.jsonl` được copy từ tập cũ. Chúng mô tả nét vẽ cũ, không phải vòng tròn đỏ trên `reference/`. Train không đọc hai thứ này. Benchmark khoanh vùng đỏ trên chính `reference/` của `removal_dataset`. `meta.jsonl` chỉ được dùng cho trường `split` (85 train / 7 validation / 8 test), vì đó là nhãn mẫu, không phải ảnh.

`prepare_dataset.py` và `train.py` resolve path rồi từ chối nếu path nằm trong `data/removal` mà không nằm trong `data/removal_dataset`. `folder_path`, `control_path` và mọi `--ctrl_img` chỉ trỏ vào bản copy của `removal_dataset`.

## Thiết kế thí nghiệm

Một run duy nhất trên 4B base, chọn checkpoint tốt nhất bằng benchmark trên 15 mẫu held-out.

- **Split**: lấy `split` trong `meta.jsonl` (85/7/8). Ảnh vẫn là file `reference/` và `target/` của `removal_dataset`. Val dùng cho sample lúc train; test chỉ xuất hiện ở benchmark.
- **Caption**: một câu, copy nguyên `target/*.txt`. Không có `--caption-mode`.
- **Cái model phải học**: xóa vật thể trong vòng tròn đỏ và xóa luôn vòng tròn, rồi vá nền. Residue của nét đỏ được đo trên vòng tròn trong `reference/`, không đo trên `mask/` của tập cũ.
- **Hyperparams (chốt theo bảng của bạn)**:
  - Backbone `black-forest-labs/FLUX.2-klein-base-4B`, `arch: flux2_klein_4b`
  - LoRA rank 32: `linear: 32`, `linear_alpha: 32`
  - `lr: 1e-4`, `adamw8bit`, `flowmatch`, `timestep_type: weighted`
  - `steps: 1500` (pilot run)
  - `resolution: [512]`
  - `save_every: 250` -> 6 checkpoint (250...1500)
  - batch 1 + grad accum 4, `gradient_checkpointing: true`, `quantize_te: true`

Rank 32 ở 512 là cấu hình nhẹ: VRAM thấp hơn nhiều so với mức 24 GB blog nêu, và 1500 step nằm ngay trong vùng đỉnh 1000-1750 mà blog quan sát được cho edit LoRA. Vì checkpoint cuối cùng (1500) đã nằm trong vùng đỉnh chứ không vượt qua nó, benchmark cần chấm đủ cả 6 mốc chứ không chỉ các mốc giữa; nếu điểm vẫn còn đang lên ở step 1500 thì kết luận của pilot là "cần kéo dài run", và `--resume` trong `train.py` cho phép train tiếp mà không mất gì.

## Cấu trúc code trong `core/`

```
core/
  env.sh                 # HF_HOME=/workspace/data, HF_HUB_ENABLE_HF_TRANSFER=1, nạp .env
  setup.sh               # miniconda -> 2 conda env -> clone ai-toolkit -> tải weight
  prepare_dataset.py     # chi copy reference/target/txt tu removal_dataset
  configs/
    removal_scribble_4b.yaml
  train.py               # wrapper chạy ai-toolkit, progress bar
  benchmark.py           # sinh ảnh + chấm điểm mọi checkpoint
  app.py                 # Gradio demo
  guide.md
```

### `setup.sh`

Chạy một lần trên pod, idempotent:

1. `mkdir -p /workspace/data` rồi export `HF_HOME=/workspace/data`, `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_HUB_DISABLE_XET=1`; đọc `HF_TOKEN` từ `.env` ở repo root (có `.env.example`).
2. Tải Miniconda vào `/workspace/miniconda3` nếu chưa có.
3. Env `klein` (py 3.12) cho train: `torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130` rồi `pip install -r ai-toolkit/requirements.txt` (đúng theo README của ai-toolkit).
4. Env `klein-infer` (py 3.12) cho benchmark + Gradio: torch tương tự + `diffusers` bản mới nhất từ git, `peft`, `gradio`, `transformers`, `accelerate`, `lpips`, `scikit-image`, `opencv-python-headless`. Tách env vì ai-toolkit pin diffusers ở một commit cũ và dùng `Flux2Pipeline` nội bộ của nó, còn blog/Gradio cần `diffusers.Flux2KleinPipeline`. Weight dùng chung qua HF cache nên không tốn dung lượng gấp đôi.
5. `git clone https://github.com/ostris/ai-toolkit` vào `/workspace/ai-toolkit` (repo local hiện không phải git repo nên clone sạch trên pod).
6. Prefetch weight bằng `hf download`: `black-forest-labs/FLUX.2-klein-base-4B` (file `flux-2-klein-base-4b.safetensors`), `ai-toolkit/flux2_vae` (file `ae.safetensors`), `Qwen/Qwen3-4B` (text encoder), và `black-forest-labs/FLUX.2-klein-4B` (distilled, cho inference). Ba repo đầu là đúng những gì `Flux2Klein4BModel` đi tìm:

```44:44:ai-toolkit/toolkit/models/registry.py
    "flux2_klein_4b": {"modality": "image", "model": {"name_or_path": "black-forest-labs/FLUX.2-klein-base-4B", "quantize_te": True}, "sample": dict(IMG)},
```

### `prepare_dataset.py`

Chỉ đọc `data/removal_dataset/` (`reference/`, `target/`, `mapping.csv`, `meta.jsonl` cho cột `split`). Không mở `data/removal/`, không copy `mask/`. Xuất ra `/workspace/datasets/removal_scribble/`:

- `target/<index>.png` + `<index>.txt`, `reference/<index>.png` cho **85 mẫu `split=train`**. File `.txt` copy nguyên, không viết lại.
- `eval/manifest.json` cho 15 mẫu val+test. Mỗi dòng trỏ file trong `removal_dataset`: `reference`, `target`, `prompt` (câu chung), `split`, `index`.
- `dataset_card.json`: số mẫu mỗi split, danh sách index held-out, câu prompt duy nhất, và đường dẫn nguồn đã resolve.
- Dừng nếu thiếu cặp reference/target/txt, nếu kích thước reference và target lệch nhau, hoặc nếu path nguồn resolve vào `data/removal`.

### `configs/removal_scribble_4b.yaml`

Dựa trên [ai-toolkit/config/examples/train_lora_flux_kontext_24gb.yaml](ai-toolkit/config/examples/train_lora_flux_kontext_24gb.yaml), đổi sang klein:

```yaml
datasets:
  - folder_path:  "/workspace/datasets/removal_scribble/target"
    control_path: "/workspace/datasets/removal_scribble/reference"
    caption_ext: "txt"
    caption_dropout_rate: 0.05
    cache_latents_to_disk: true
    resolution: [512]
network:
  type: "lora"
  linear: 32
  linear_alpha: 32
train:
  steps: 1500
  lr: 1e-4
  optimizer: "adamw8bit"
  noise_scheduler: "flowmatch"
  timestep_type: "weighted"
  batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
save:
  save_every: 250
  max_step_saves_to_keep: 6
model:
  name_or_path: "black-forest-labs/FLUX.2-klein-base-4B"
  arch: "flux2_klein_4b"
  quantize_te: true
```

`sample.prompts` dùng đúng câu caption chung, với `--ctrl_img` trỏ vào 4 `reference/` của val split trong bản copy của `removal_dataset`.

### `train.py`

- Nạp `env.sh`, kiểm tra dataset + weight đã sẵn sàng trước khi tốn GPU.
- Chạy `python run.py <config>` trong env `klein` với `PYTHONUNBUFFERED=1`, stream nguyên output (ai-toolkit đã có tqdm riêng).
- Parse dòng step để vẽ thêm một progress bar tổng (`tqdm`) hiển thị step/total, loss trung bình trượt, ETA, và mốc checkpoint kế tiếp; đồng thời ghi `output/<name>/train.log`.
- Hỗ trợ `--resume` và bắt Ctrl-C để tắt tiến trình con cho gọn.

### `benchmark.py`

Chạy trong env `klein-infer`, lặp qua mọi `.safetensors` trong output, sinh ảnh cho 15 mẫu held-out ở seed cố định (4 step, `guidance_scale=4.0`, distilled pipeline). Ảnh và prompt lấy từ `removal_dataset`. Vùng đo là pixel đỏ trên `reference/` (vòng tròn), không dùng `mask/` copy từ tập cũ.

- `scribble_residue` — tỉ lệ pixel đỏ bão hòa còn sót trong output, trong vùng vòng tròn đã dilate nhẹ. Thấp = đã xóa sạch nét vẽ.
- `preserve_psnr` / `preserve_ssim` — so output với `reference` **bên ngoài** vòng tròn. Phần đó là ảnh gốc, nên metric này đo "đừng đụng phần còn lại".
- `target_lpips` — LPIPS giữa output và `target/*.png`. Cùng kích thước canvas nhưng nội dung target do model khác sinh, nên đây là tín hiệu mềm.

Xuất `results.csv`, `report.md` xếp hạng checkpoint, và contact sheet `reference | output | target` mỗi checkpoint để bạn chốt bằng mắt đúng như blog khuyên. Thêm `--baseline` để chạy pipeline không LoRA làm mốc so sánh.

### `app.py`

Gradio app dùng `Flux2KleinPipeline` theo đúng snippet blog:

- Tab 1 "Scribble & Remove": `gr.ImageEditor` với brush đỏ để vẽ vòng tròn lên ảnh upload (hoặc upload sẵn ảnh đã có vòng tròn), ô prompt điền sẵn đúng câu `Remove the object inside the red circle and naturally restore the background.`, và `gr.ImageSlider` hiện before/after.
- Sidebar: dropdown chọn checkpoint (quét thư mục output), `lora_scale`, `num_inference_steps`, `guidance_scale`, seed, và switch distilled (`FLUX.2-klein-4B`, 4 step) hay base (`FLUX.2-klein-base-4B`, 50 step).
- Tab 2 "Held-out gallery": bấm một phát chạy 8 mẫu test trong `removal_dataset` để xem nhanh.
- Tự resize về bội số 16 và giữ `(W*H)/256 <= 4096` theo ràng buộc của klein. `peft` nằm trong requirements vì không có nó `load_lora_weights()` sẽ chết lúc runtime.

### `guide.md`

Hướng dẫn tiếng Việt theo đúng thứ tự bạn yêu cầu: chuẩn bị pod và `.env` -> `setup.sh` -> upload `data/removal_dataset` (không cần `data/removal`) -> `prepare_dataset.py` -> `train.py` -> `benchmark.py` và cách đọc bảng kết quả -> `app.py` (kèm lệnh SSH port-forward để mở Gradio từ máy bạn) -> mục troubleshooting (OOM thì hạ resolution, cache latents, `HF_TOKEN` sai, thiếu `peft`, path trỏ nhầm vào `data/removal`).

## Rủi ro đã lường trước

- `mask/` trong `removal_dataset` là nét của tập cũ, lệch với vòng tròn đỏ trên `reference/`. Dùng nó để chấm điểm sẽ đo sai vùng. Benchmark không đọc thư mục này.
- Canvas `target` cùng kích thước `reference`, nhưng nội dung target do model khác sinh nên không pixel-aligned. Metric cứng (`preserve_psnr`) so output với `reference` ở ngoài vòng tròn đỏ. `target_lpips` chỉ là tín hiệu mềm.
- Một prompt chung là cố ý: mọi `reference/` đều là vòng tròn đỏ, và inference cũng dùng câu đó. Blog cảnh báo phrasing chỉ áp dụng khi muốn model nghe nhiều cách nói. Run này không đặt mục tiêu đó.
- 85 cặp train là ít so với khoảng 50-200 blog đề xuất, nhưng nằm trong vùng chấp nhận được. Nếu kết quả yếu, thêm cặp vào `removal_dataset` theo cùng format `reference/` + `target/` + câu prompt này. Không trỏ train sang `data/removal`.
- Train ở 512 nhưng inference ở 1024 (cỡ an toàn của klein) là lệch domain. `benchmark.py` sẽ chạy held-out ở cả 512 và 1024 để đo mức độ ảnh hưởng; nếu 1024 tệ rõ rệt thì run kế tiếp nâng `resolution` lên `[512, 768]`.
- Rank 32 có thể thiếu dung lượng để học đồng thời hai việc (vá nền + xóa nét đỏ). Nếu benchmark cho thấy `scribble_residue` cao mà background tốt (hoặc ngược lại), hướng xử lý là tăng rank chứ không phải kéo dài step.