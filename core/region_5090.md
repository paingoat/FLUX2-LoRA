# Fine-tune removal trên RTX 5090 với region-weighted MSE

Branch: `experiment/region-weighted-loss-5090`.
Giữ hai input `[original, reference]`, target đã xóa vật thể, Klein **base 4B**,
LoRA rank/alpha 32, 512, batch 1, accumulation 4, LR 1e-4, 1500 steps.
Hai config A/B chỉ khác tên run và strength: A=0, B=1.5.

`loss = mean((1 + strength * latent_mask) * (velocity_pred - velocity_target)^2)`.
Nền có trọng số 1; vùng ưu tiên tối đa 2.5. Mask được broadcast qua channels.
Không chia trọng số cho trung bình mask. Target vẫn là `noise - target_latent`.
Hai run bắt đầu độc lập từ base, không resume checkpoint của nhau.
Loss mới có thể lớn hơn vì đổi trọng số; không chọn checkpoint bằng loss train A/B.

Môi trường PSNR/SAM3 cũ được giữ riêng. Code chỉ được kiểm tra CPU ở local;
setup Linux, dependencies, optimizer CUDA và VRAM full model cần bạn xác nhận
trên RTX 5090. Không có smoke test hoặc train GPU nào đã được chạy thay bạn.

## 1. Đưa code và dữ liệu lên máy

Clone riêng nhánh thử nghiệm từ GitHub:

```bash
cd /workspace
git clone --depth 1 --single-branch --branch experiment/region-weighted-loss-5090 \
  https://github.com/paingoat/FLUX2-LoRA.git FLUX2-LoRA-region
cd /workspace/FLUX2-LoRA-region
```

Trong các lệnh bên dưới, nếu dùng thư mục clone này thì thay
`/workspace/FLUX2-LoRA` bằng `/workspace/FLUX2-LoRA-region`.
Nếu dùng checkout đã có, lưu thay đổi riêng của bạn trước khi chuyển nhánh.

Gói `.local/region_5090_code.zip` chứa code hiện tại (không có `.env`, weights,
ảnh hoặc `.git`). Có thể upload gói này và `.sha256` lên `/workspace` rồi chạy:

```bash
cd /workspace
sha256sum -c region_5090_code.zip.sha256
mkdir -p /workspace/FLUX2-LoRA
unzip -o region_5090_code.zip -d /workspace/FLUX2-LoRA
```

Giải nén vào checkout dành riêng cho thử nghiệm nếu trên máy đang có code khác
chưa lưu. Gói zip không tạo Git branch trên máy đích.

Dataset archive cũ **không chứa mask**. Tại repo local, đóng gói lại:

```powershell
python core/package_dataset.py --include-masks --output .local/removal_region_dataset.tar.gz
```

Upload `.tar.gz` và `.sha256` lên `/workspace`, rồi trên máy GPU:

```bash
cd /workspace
sha256sum -c removal_region_dataset.tar.gz.sha256
tar -xzf removal_region_dataset.tar.gz -C /workspace/FLUX2-LoRA
cd /workspace/FLUX2-LoRA
```

Archive chứa bộ original/reference/target/caption và `mask/<index>.png` sẵn có.
Không cần upload toàn bộ Pico hoặc chạy SAM3 lại. Nếu đã có dữ liệu đúng trên
máy, chỉ cần bảo đảm thư mục `data/removal_dataset/mask/` có đủ mask.

### Tải archive từ Hugging Face

Repo dữ liệu private: `Thoandanh/flux2-klein-removal-dual`.
Sau khi setup env ở mục 2 (để có CLI `hf`), tải đúng hai file, tránh tải lại
toàn bộ repo dữ liệu:

```bash
hf download Thoandanh/flux2-klein-removal-dual \
  removal_region_dataset.tar.gz removal_region_dataset.tar.gz.sha256 \
  --repo-type dataset --local-dir /workspace
cd /workspace
sha256sum -c removal_region_dataset.tar.gz.sha256
tar -xzf removal_region_dataset.tar.gz -C /workspace/FLUX2-LoRA-region
cd /workspace/FLUX2-LoRA-region
```

Dùng HF_TOKEN trong `.env` của tài khoản có quyền đọc repo, hoặc `hf auth login`.
Nếu cache/dataset đã có đủ thì bỏ bước download tương ứng.

## 2. Setup riêng cho RTX 5090

Cần Linux, Conda trên PATH, Git, driver hỗ trợ CUDA 12.8, dung lượng cho base
model/text encoder và env. Dùng `screen` để giữ tiến trình khi SSH ngắt:

```bash
screen -S region-train
cd /workspace/FLUX2-LoRA
cp -n .env.example .env
# Sửa HF_TOKEN và các đường dẫn cache trong .env.
# Dùng /backup/data/art-gen nếu đã mount cache của pilot PSNR ở đây.
# Nếu Pod chỉ có /workspace persistent: HF_HOME=/workspace/art-gen,
# và đổi cả HF_HUB_CACHE, HF_XET_CACHE, HF_ASSETS_CACHE tương ứng.
set -o pipefail
bash core/setup_region_5090.sh 2>&1 | tee /workspace/setup-region.log
```

Setup tạo `klein-region-5090`, Python 3.12, FFmpeg 7, torch/torchaudio 2.8.0,
torchvision 0.23.0 cu128. AI Toolkit pin commit
`beb94824c51480a04d3c86c718603676fdef6938`, clone riêng trong cache.
Giữ requirements image trainer của commit đó, đổi optional audio dependency
torchcodec 0.15 thành 0.6 để khớp torch 2.8; constraints ngăn pip đổi bộ torch.
Chạy `pip check`, import trainer/pipeline, lưu package freeze trong env.
Nếu resolver/import thất bại, dừng và giữ log; không dùng env chưa setup xong.

Patch trainer kiểm tra SHA256 mã nguồn, áp dụng lại an toàn và không làm thay
đổi masked loss cũ khi không bật biến môi trường của thí nghiệm.
Lưu `region-packages.txt`, `region-runtime.json` trong `$CONDA_PREFIX` cùng kết quả.

Mỗi lần mở terminal mới:

```bash
cd /workspace/FLUX2-LoRA
source "$(conda info --base)/etc/profile.d/conda.sh"
source core/env.sh
conda activate klein-region-5090
export AITOOLKIT="$HF_HOME/source/ai-toolkit-region"
export KLEIN_PYTHON="$CONDA_PREFIX/bin/python"
export HF_HUB_DISABLE_XET=0
export HF_HUB_ENABLE_HF_TRANSFER=0
```

Nếu bạn đổi `REGION_ENV` hoặc `AITOOLKIT` lúc setup, dùng đúng giá trị đó ở đây.
Bạn có thể tự chạy preflight GPU nhẹ, không tải model hoặc bắt đầu fine-tune:

```bash
nvidia-smi
python core/check_region_5090.py
```

Preflight kiểm tra sm_120, BF16 backward và AdamW8bit; không chứng minh đủ VRAM
cho full model. Hai config bật checkpointing, low_vram và cache/unload text
encoder. Chưa xác nhận full train/preview hai reference vừa 32 GB trên máy thật.
Nếu OOM, giữ cùng điều chỉnh bộ nhớ cho A và B rồi chạy lại bằng tên run mới.

## 3. Chuẩn bị dữ liệu và mask từ dữ liệu sẵn có

```bash
python core/prepare_dataset.py
python core/prepare_region_masks.py
```

Thực hiện bước prepare_dataset trước khi tạo region masks. Nếu chạy lại prepare
hoặc thay ảnh, tạo lại mask bằng `--output` mới và cập nhật `mask_path` trong cả
hai config. Không xóa thư mục kết quả cũ; script từ chối ghi đè output đã tồn tại.

Mask nguồn: `data/removal_dataset/mask/<index>.png`, nhị phân 0/255,
trắng là vùng edit. Đây là mask sẵn có của dataset; không mặc định coi là
segmentation ground truth chính xác. Có thể dùng `--mask-root /path/to/masks`
nếu muốn thay bằng mask đã duyệt, cùng tên index và kích thước ảnh gốc.

Script lấy hợp mask nguồn với pixel annotation (`abs(reference-original)>8`),
giãn **bán kính 25 px tại cạnh dài 512**, scale theo cạnh dài ảnh nguồn;
kernel vuông cạnh `2*r+1`. Đây là quy ước thử nghiệm, không khẳng định tái hiện
chính xác cách dilation 50 px của paper. Mask đi qua cùng scale/crop/flip với
target trong loader AI Toolkit rồi được nội suy về latent; loader còn có blur
mask nhẹ mặc định nên biên có trọng số mềm trong [1, 2.5].

Kết quả tại `/workspace/datasets/removal_dual/region_loss/`:

- `train/`: mask loss của **train split**.
- `preview/`: vùng loss tô tím trên ảnh gốc; xem trước để kiểm tra mask đúng.
- `region/`, `stroke/`: mask đánh giá cố định, **chưa giãn cho loss**.
- `eval_manifest.json`: validation/test cùng đường dẫn mask.
- `mask_provenance.json`: nguồn, split, radius và checksum ảnh/mask.

Không dùng mask loss đã giãn làm vùng loại trừ cho PSNR; không tạo vùng đánh giá
từ output của model. Train kiểm tra checksum để phát hiện ảnh/mask bị thay đổi.

## 4. Chạy A/B

Kiểm tra CPU về đường dẫn dữ liệu, mask, toolkit và model cache:

```bash
python core/train.py --config core/configs/removal_uniform_5090.yaml --check-only
python core/train.py --config core/configs/removal_region_5090.yaml --check-only
```

Chạy tuần tự, một process train trên GPU:

```bash
python core/train.py --config core/configs/removal_uniform_5090.yaml
python core/train.py --config core/configs/removal_region_5090.yaml
```

Muốn thử loss mới trước, chạy riêng lệnh thứ hai. Khi cần kết luận A/B vẫn phải
chạy đối chứng A. Các run dùng seed 42 cho Python/NumPy/Torch; cùng seed không
bảo đảm bitwise deterministic giữa các máy hoặc khi resume.

Outputs: `/workspace/output/removal_uniform_5090/` và
`/workspace/output/removal_region_5090/`, checkpoint mỗi 250 steps, log,
resolved config và `experiment.json`. Resume đúng config bằng `--resume`;
wrapper từ chối nối checkpoint khi config/loss, mask provenance hoặc trainer đổi.
AI Toolkit resume không đảm bảo khôi phục chính xác trạng thái RNG/dataloader.

Rời screen: Ctrl+A, D. Quay lại: `screen -d -r region-train`.
screen không giữ process khi Pod reboot/stop; dữ liệu cần nằm trên storage bền.

## 5. Đánh giá nhiều tiêu chí

Dùng cùng env `klein-region-5090`. Lần đầu benchmark sẽ tải thêm pipeline diffusers
base và LPIPS weights vào cache; cần mạng. Đánh giá chính trên **base + LoRA**,
50 steps, guidance 4, cùng resolution/seeds, có CPU offload:

```bash
python core/benchmark.py --model base --steps 50 --guidance 4 --offload \
  --manifest /workspace/datasets/removal_dual/region_loss/eval_manifest.json \
  --split validation --baseline \
  --checkpoints /workspace/output/removal_uniform_5090 \
  --resolutions 512 --seeds 42 123 2026 \
  --output /workspace/output/region_eval_A_validation

python core/benchmark.py --model base --steps 50 --guidance 4 --offload \
  --manifest /workspace/datasets/removal_dual/region_loss/eval_manifest.json \
  --split validation --baseline \
  --checkpoints /workspace/output/removal_region_5090 \
  --resolutions 512 --seeds 42 123 2026 \
  --output /workspace/output/region_eval_B_validation
```

`results.csv`, `report.md`, `review.html` gồm:

- PSNR ngoài vùng cố định, PSNR ngoài vùng +8 px, SSIM ngoài vùng.
- Background composite LPIPS: thay vùng edit trong output bằng source rồi tính
  LPIPS toàn ảnh. Đây là **chỉ số thay thế** cho chất lượng nền, không phải
  LPIPS được mask chính xác theo từng pixel.
- Target LPIPS toàn ảnh và LPIPS crop bounding box vùng edit. Target có thể là
  một trong nhiều cách tái tạo nền hợp lệ; không dùng LPIPS riêng để kết luận.
- Tỷ lệ nét annotation còn giống reference tại vị trí ban đầu; không phát hiện
  mọi nét bị dịch chuyển.
- Thời gian inference và peak allocated CUDA VRAM từng mẫu, không gồm metric.
- Blind human review: xóa thành công, sạch nét vẽ, nền tự nhiên, bảo toàn ngoài vùng.

Chọn checkpoint từ validation bằng chất lượng xóa và bảo toàn, không chỉ PSNR.
Sau đó đánh giá hai checkpoint đã chọn trên test, dùng tên file thật thay A/B:

```bash
python core/benchmark.py --model base --steps 50 --guidance 4 --offload \
  --manifest /workspace/datasets/removal_dual/region_loss/eval_manifest.json \
  --split test --baseline \
  --checkpoint /workspace/output/removal_uniform_5090/CHOSEN_A.safetensors \
               /workspace/output/removal_region_5090/CHOSEN_B.safetensors \
  --resolutions 512 1024 --seeds 42 123 2026 \
  --output /workspace/output/region_eval_test
```

Mỗi lần chạy dùng output mới. Không so trung bình nếu coverage thành công khác
nhau. PSNR cao với ảnh không xóa gì không phải kết quả tốt.
Chấm và export `review.csv` từ review.html, rồi tổng hợp theo
[guide.md](guide.md). VIBE Removal dùng [quy trình chuẩn bị hiện có](guide.md#5-test-nội-bộ-và-vibe-removal),
chạy benchmark với `--model base --steps 50 --guidance 4 --offload`,
`--split external --manifest /workspace/datasets/vibe_removal/manifest.json`
và hai `--checkpoint` đã chọn, output riêng. VIBE chưa có mask vật thể chuẩn
trong pipeline này: metric vùng ngoài dựa trên hình học annotation, báo riêng
với test nội bộ. Human review là rubric dự án, không phải VIBE leaderboard judge.

Phép thử distilled bổ sung dùng `--model distilled --steps 4 --guidance 1`;
báo riêng khỏi kết quả base. Không dùng 20 ảnh PSNR pilot để chọn checkpoint
nếu chúng trùng train. Không tự gọi VLM judge trả phí trong các lệnh trên.
