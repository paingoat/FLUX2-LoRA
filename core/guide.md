# Huấn luyện EditLoRA xóa vật thể trong vòng tròn đỏ

Mục tiêu: một LoRA cho `FLUX.2-klein-base-4B`, học xóa vật thể nằm trong vòng tròn đỏ và xóa luôn vòng tròn. Inference dùng bản distilled `FLUX.2-klein-4B`.

Nguồn ảnh duy nhất là `data/removal_dataset`:

- `reference/` là ảnh có vòng tròn đỏ.
- `target/` là ảnh đã xóa vật thể và vòng tròn.
- Mọi caption là một câu: `Remove the object inside the red circle and naturally restore the background.`

`data/removal` là tập cũ (nét hatch, box, cross). Không upload tập đó lên pod và không trỏ train vào đó. `prepare_dataset.py` và `train.py` sẽ dừng nếu một path resolve vào thư mục tên `removal`.

`data/removal_dataset/mask/` và `scribble_style` trong `meta.jsonl` mô tả nét của tập cũ, không trùng vòng tròn đỏ. Train và benchmark không đọc chúng. `meta.jsonl` chỉ được dùng để lấy cột `split` (85 train / 7 validation / 8 test).

Làm lần lượt trên pod. Không cần GPU để chạy `prepare_dataset.py`.

## 1. Pod và token

Tạo file `.env` ở gốc repo, cùng cấp với `core/`:

```bash
cp .env.example .env
```

Điền `HF_TOKEN`. Weight trong thí nghiệm này là public; token dùng để tránh bị Hub giới hạn tốc độ tải.

Repo trên pod có thể nằm ở bất kỳ đâu. Các đường dẫn cache, conda, ai-toolkit, dataset đã chuẩn bị và checkpoint đều nằm dưới `/workspace`.

## 2. Setup

```bash
bash core/setup.sh
```

Script này:

1. Tạo `/workspace/data` và trỏ `HF_HOME` vào đó. Bật `HF_HUB_ENABLE_HF_TRANSFER=1` và tắt Xet.
2. Cài Miniconda vào `/workspace/miniconda3` nếu chưa có.
3. Clone `ostris/ai-toolkit` vào `/workspace/ai-toolkit` trước khi cài dependency của nó.
4. Tạo env `klein` (Python 3.12): PyTorch 2.13.0 cu130, rồi `ai-toolkit/requirements.txt`.
5. Tạo env `klein-infer`: cùng bản PyTorch, cộng diffusers mới từ git, `peft==0.18.1`, Gradio, LPIPS, scikit-image, OpenCV.
6. Tải `FLUX.2-klein-base-4B` (file transformer), `ai-toolkit/flux2_vae`, `Qwen/Qwen3-4B`, và `FLUX.2-klein-4B`.

Chạy lại `setup.sh` sẽ bỏ qua bước đã xong. Xóa `/workspace/data/.klein-requirements-ok` hoặc `/workspace/data/.klein-infer-requirements-ok` nếu muốn cài lại package.

## 3. Dataset

Đưa cả repo, gồm `data/removal_dataset`, lên pod. Không cần `data/removal`.

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
source core/env.sh
python core/prepare_dataset.py
```

Script chỉ đọc `reference/`, `target/`, `mapping.csv` và cột `split` của `meta.jsonl`. Nó ghi ra `/workspace/datasets/removal_scribble/`:

- `reference/` và `target/`: 85 cặp train. File `.txt` được copy nguyên.
- `sample_controls/`: 4 ảnh validation (`0015`, `0022`, `0040`, `0057`) để ai-toolkit vẽ preview. Chúng không nằm trong folder train.
- `eval/manifest.json`: 15 mẫu validation + test, path trỏ về `removal_dataset` trên pod.
- `dataset_card.json`: số mẫu, câu prompt, path nguồn.

Chạy `prepare_dataset.py` trên pod. Manifest ghi path tuyệt đối của máy đang chạy.

Kỳ vọng in ra: `counts` gồm train 85, validation 7, test 8.

## 4. Train

```bash
source core/env.sh
conda activate klein
python core/train.py
```

Wrapper kiểm tra dataset đã chuẩn bị, từ chối path `data/removal`, kiểm tra weight trong `/workspace/data`, rồi chạy:

```bash
python run.py /path/to/core/configs/removal_scribble_4b.yaml
```

trong `/workspace/ai-toolkit`.

Config đã chốt:

| Tham số | Giá trị |
| --- | --- |
| Backbone | `black-forest-labs/FLUX.2-klein-base-4B` |
| LoRA | rank 32, alpha 32 |
| Learning rate | 1e-4, AdamW 8-bit |
| Steps | 1500 |
| Resolution | 512 |
| Save | mỗi 250 steps, giữ 6 checkpoint |
| Batch | 1, gradient accumulation 4 |
| Scheduler | flowmatch |

Log đầy đủ nằm ở `/workspace/output/removal_scribble_4b/train.log`. Thanh tqdm phía trên là của wrapper: step/1500, loss trung bình 20 step gần nhất, và mốc checkpoint kế tiếp. ai-toolkit vẫn in dòng của nó vào log.

Checkpoint nằm trong `/workspace/output/removal_scribble_4b/`. Các mốc 250, 500, 750, 1000 và 1250 có dạng `removal_scribble_4b_000000250.safetensors`. File cuối `removal_scribble_4b.safetensors` không có số ở đuôi; đó là weight sau đúng 1500 step. Vòng lặp của ai-toolkit không ghi thêm một file `_000001500`.

Dừng bằng Ctrl-C. Chạy tiếp:

```bash
python core/train.py --resume
```

Nếu không truyền `--resume` mà folder output đã có `.safetensors`, script dừng để tránh nối run nhầm. ai-toolkit tự đọc checkpoint mới nhất khi được phép resume.

Preview trong lúc train dùng đúng câu prompt chung và 4 ảnh validation. Ảnh sample nằm cạnh checkpoint, trong `samples/`.

## 5. Benchmark

```bash
source core/env.sh
conda activate klein-infer
python core/benchmark.py --baseline
```

Bỏ `--baseline` nếu không cần mốc pipeline không LoRA.

Script load bản distilled, 4 step, `guidance_scale=4`, seed 42. Với mỗi checkpoint nó chạy 15 mẫu held-out ở cạnh dài 512 và 1024. Ảnh được làm tròn về bội số của 16 và giữ `(width * height) / 256 <= 4096`.

Vùng đo là nét đỏ thuần trên `reference/` (R cao, G và B thấp), không phải mọi vùng ấm của ảnh. Vòng tròn được lấp kín rồi dilate nhẹ. `mask/` không được đọc.

| Metric | Ý nghĩa |
| --- | --- |
| `scribble_residue` | Tỉ lệ pixel của nét đỏ còn đỏ trong output. Thấp hơn là tốt. |
| `preserve_psnr`, `preserve_ssim` | So output với `reference` bên ngoài vòng tròn. Cao hơn là tốt. |
| `target_lpips` | Khoảng cách tới `target/`. Thấp hơn là gần ảnh đích hơn, nhưng target không pixel-aligned nên chỉ là tín hiệu mềm. |

Kết quả:

- `/workspace/output/removal_scribble_4b/benchmark/results.csv` — từng ảnh.
- `benchmark/report.md` — trung bình theo checkpoint và resolution, kèm checkpoint nên xem trước.
- `benchmark/sheets/<checkpoint>_<resolution>.jpg` — mỗi hàng là `reference | output | target`.
- `benchmark/images/` — output đầy đủ.

Đọc `report.md` trước, rồi mở contact sheet của checkpoint được đề xuất. Nếu step 1500 vẫn tốt hơn step 1250, report sẽ nói pilot chưa chạm đỉnh. Copy checkpoint muốn giữ ra ngoài folder output (vì config chỉ giữ 6 file), tăng `steps` trong yaml cho lớn hơn 1500, rồi `python core/train.py --resume`.

## 6. Gradio

```bash
source core/env.sh
conda activate klein-infer
python core/app.py
```

App nghe `0.0.0.0:7860` và bật Gradio `share`, nên log in một URL public dạng `https://....gradio.live`. Mở URL đó từ máy bất kỳ; không cần SSH tunnel. Link share sống khoảng 72 giờ và hết hạn khi process dừng.

Trên pod, `http://<pod-ip>:7860` vẫn dùng được nếu cổng 7860 được mở.

Tab **Scribble & Remove**: upload ảnh, vẽ vòng tròn đỏ, bấm Remove. Slider hiện ảnh trước và sau. Prompt bị khóa đúng một câu đã train. Sidebar chọn checkpoint, LoRA scale, số step, guidance và seed. Pipeline mặc định là distilled 4 step; chuyển sang base sẽ đẩy số step lên 50.

Tab **Held-out gallery**: chạy 8 mẫu test trong `removal_dataset`.

Bấm **Refresh checkpoints** nếu train vừa xong và dropdown chưa thấy file mới.

## Xử lý lỗi

**OOM lúc train.** Config đã là 512, gradient checkpointing bật, text encoder quantize. Đóng process khác trên GPU. Nếu OOM ngay lúc sample, thêm `disable_sampling: true` dưới `train:` rồi chấm bằng `benchmark.py`.

**OOM lúc benchmark ở 1024.** Dòng đó bị bỏ qua và ghi `OOM` ra stderr. Kết quả 512 vẫn được ghi. Chạy lại không cần `--resolutions 1024` nếu chỉ muốn 512:

```bash
python core/benchmark.py --resolutions 512
```

**`HF_TOKEN` sai hoặc thiếu.** Hub trả 401 hoặc chậm. Sửa `.env`, `source core/env.sh`, chạy lại `setup.sh`.

**`PEFT backend is required`.** Đang ở nhầm env. Benchmark và Gradio phải dùng `klein-infer`, nơi đã cài `peft==0.18.1`.

**Path bị từ chối vì `data/removal`.** Một `folder_path`, `control_path` hoặc `--ctrl_img` đang trỏ vào tập cũ. Train chỉ được đọc `/workspace/datasets/removal_scribble`, là bản copy của `removal_dataset`.

**Thiếu `dataset_card.json`.** Chưa chạy `prepare_dataset.py` trên pod.

**Thiếu weight.** `train.py` dừng và liệt kê repo chưa có trong `/workspace/data`. Chạy lại `setup.sh`.

**Cache latents.** `cache_latents_to_disk: true` đã bật. Lần train đầu sẽ encode trước khi thấy step tăng. Đó không phải là treo.
