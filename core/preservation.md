# Pilot bảo toàn ngoài mask: một ảnh đầu vào

20 mẫu **remove** lấy từ Pico đã có tại `samples/remove/` và `metadata.jsonl`.
Ba nhánh: ảnh sạch + text; scribble đỏ overlay; SAM3 mask phủ đỏ alpha 0.4.
Mỗi lượt chỉ gửi một PIL image. Model mặc định là Klein 4B distilled, không LoRA,
BF16, 4 steps, guidance 1.0, seed 42. Không ghép nền sau inference.

## Chạy trước bằng mask có sẵn (không cần SAM3)

Gói private `Thoandanh/flux2-preservation-pico20` có bản bổ sung
`pico_removal_20_existing_masks.zip`: đúng 20 ảnh gốc như bản đầu, thêm mask.png và
metadata chứa đường dẫn mask. Không dùng change_mask.png thay cho mask.png.

```bash
conda activate preservation-flux
source core/env.sh
hf download Thoandanh/flux2-preservation-pico20 --repo-type dataset \
  --include 'pico_removal_20_existing_masks.zip' 'pico_removal_20_existing_masks.zip.sha256' \
  --local-dir output/pico_existing_download
(cd output/pico_existing_download && sha256sum -c pico_removal_20_existing_masks.zip.sha256)
python -m zipfile -e output/pico_existing_download/pico_removal_20_existing_masks.zip .

python core/prepare_preservation.py --mask-source existing --output output/preservation_existing_data
python core/benchmark_preservation.py --manifest output/preservation_existing_data/manifest.json \
  --allow-unreviewed --validate-only
python core/benchmark_preservation.py --manifest output/preservation_existing_data/manifest.json \
  --allow-unreviewed --limit 2 --output output/preservation_existing_smoke
python core/benchmark_preservation.py --manifest output/preservation_existing_data/manifest.json \
  --allow-unreviewed --output output/preservation_existing_20
```

Mask được kiểm tra binary/diện tích/kích thước trước khi resize nearest-neighbor.
Chế độ existing chạy CPU, không import SAM3 hoặc tải SAM3 weights. Chọn output riêng
để giữ nguyên lần chạy SAM3 cũ. Selection cố định mask_source, không chuyển nguồn
mask trong cùng selection. Cả ba nhánh dùng cùng mask để chấm.

`--allow-unreviewed` chỉ cho phép pilot với existing mask chưa được duyệt; report/CSV
ghi rõ `unreviewed-existing-pilot`. Không cho phép mask đã đánh rejected hoặc SAM3
mask chưa được duyệt. Có thể mở mask_review.html rồi đặt approved=1 ở mask_review.csv,
sau đó bỏ flag này. Mask có sẵn là cue/change mask từ pipeline SIGMA, không phải
ground truth vật thể chính xác: PSNR pilot chỉ để kiểm tra pipeline và xem xu hướng.

## Khôi phục khi SAM3 không tìm được mask

Phiên bản mới xử lý tiếp các mẫu còn lại, lưu từng mask thành công và liệt kê lỗi
ở `prepare_errors.csv`. Không tạo manifest hoàn chỉnh nếu vẫn thiếu mẫu; không
âm thầm lấy mask cũ thay cho SAM3. Sửa overrides rồi chạy lại cùng lệnh/output.
Những mask thành công có cùng nguồn ảnh, revision và overrides được tái sử dụng;
đổi override của một mẫu chỉ tính lại mẫu đó. Mask từ script cũ chưa có thông tin
resume sẽ tính lại một lần. `pkg_resources` và `timm` deprecation warnings không
phải lỗi forward; giữ setuptools pin, không cần sửa để chạy ảnh.

## 1. Chuẩn bị máy Linux RTX 5090

Chạy từ thư mục repo chứa `core/` và `samples/`. Cần Conda sẵn trên PATH, Git,
driver NVIDIA tương thích CUDA 12.8, dung lượng trống cho SAM3 và FLUX.
Các ảnh samples và metadata bị gitignore: **copy chúng lên máy GPU** nếu clone mới.
Script không cài trainer/ai-toolkit và không tải lại Pico.

```bash
cp .env.example .env
# Điền HF_TOKEN của tài khoản đã có quyền facebook/sam3.
# Giữ các biến cache sau trong .env:
# HF_HOME=/backup/data/art-gen
# HF_HUB_CACHE=/backup/data/art-gen/hub
# HF_XET_CACHE=/backup/data/art-gen/xet
# HF_ASSETS_CACHE=/backup/data/art-gen/assets
bash core/setup_preservation.sh
```

Setup tạo `preservation-sam3` và `preservation-flux`, Python 3.12,
torch 2.8.0 / torchvision 0.23.0 cu128. Kiểm tra GPU capability `(12,0)`, wheel
có `sm_120`, BF16 matmul thực tế và `pip check`. Không cần FlashAttention hoặc
extension CUDA tự biên dịch. `TORCH_CUDA_ARCH_LIST=12.0` không sửa được wheel sai.

SAM3 code pin tại commit `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`.
Có thể đặt `SAM3_REVISION` và `SAM3_SOURCE` trước setup để dùng commit/checkout khác.
Đã đối chiếu với fork `KhoaLeDang2375/sam3`, nhánh `dev1` tại commit
`b9da68d3c6f2b4b86a1a2255d6848ddf1cda076e`: image processor giống upstream;
app bọc cả set_image và prompt trong BF16 autocast để tránh lỗi Float32/BFloat16.
Script chuẩn bị dữ liệu dùng cùng phạm vi autocast. Chỉ dùng text/box grounding,
không cần bật interactive predictor dành cho point click.

Muốn dùng đúng fork đã triển khai thay cho upstream mặc định:

```bash
SAM3_REPOSITORY=https://github.com/KhoaLeDang2375/sam3.git \
SAM3_REVISION=b9da68d3c6f2b4b86a1a2255d6848ddf1cda076e \
SAM3_SOURCE=/backup/data/art-gen/source/sam3-dev1 \
bash core/setup_preservation.sh
```

Setup từ chối checkout bẩn hoặc không đúng revision. Model revision được resolve
một lần khi setup và lưu cùng package freeze tại `$CONDA_PREFIX/preservation-*-lock.json`.
Rerun setup giữ model revision; khi cần đổi revision hãy tạo môi trường mới rõ ràng.
Mask preparation và benchmark dùng đúng revision trong lock.
SAM3 requirements giữ SciPy/scikit-image giống runtime trong fork; SciPy dùng ở
nhánh training/video (lazy import), không phải bắt buộc cho text grounding đơn lẻ.
Các import runtime trực tiếp gồm einops, pycocotools, psutil và Hydra qua module
mà model_builder nạp. Pin NumPy <2 và OpenCV 4.11 để tránh xung đột resolver.
Setuptools 80.9 giữ pkg_resources mà upstream vẫn dùng để tìm BPE vocabulary.
Torch cu128 trên Linux cung cấp Triton; preflight kiểm tra import Triton và
cả model_builder/image processor. Không cài Gradio vì benchmark không dùng UI server.
Không cài decord: đây là dependency của video loader được import khi gọi đọc video,
không dùng trong thí nghiệm ảnh. Setup tự gỡ decord khỏi môi trường preservation-sam3
nếu đã được cài bởi phiên bản script cũ, để tránh lỗi platform tag từ pip check.

HF_HOME mới có thể chưa chứa token đã login ở cache cũ. Đặt HF_TOKEN trong .env
hoặc `source core/env.sh` rồi `hf auth login` dưới đúng cache này; không phải xin lại quyền.
Setup kiểm tra môi trường/import; forward SAM3/FLUX được kiểm tra ở các bước sau.

## 2. Chọn 20 ảnh và chạy SAM3

```bash
conda activate preservation-sam3
python core/prepare_preservation.py --select-only
python core/prepare_preservation.py
```

Selection seed 42, sort ID trước khi random, không trùng source_key. Danh sách
`output/preservation_data/selection.json` được cố định ngay lần đầu; chạy lại dùng
cấu hình trong file này. Muốn đổi seed/count/resolution/alpha hãy dùng `--output` mới.
Mặc định cạnh dài 1024, giữ tỷ lệ rồi làm tròn theo Klein; SAM3 chạy trên đúng ảnh
đã resize, nên mask và ảnh đều ở cùng hệ tọa độ. Không dùng mask SIGMA cũ.

Instruction được tạo từ mô tả vật thể `prompt_object`, tránh prompt_short có
"inside the oval" ở nhánh text. SAM3 chọn mask confidence cao nhất làm **ứng viên**;
người duyệt phải xác nhận đúng vật thể, đủ phần và không ăn sang vật thể bên cạnh.
Nếu mô tả vật thể mơ hồ, sửa instruction/object_prompt trong selection trước khi
tạo mask; chọn thư mục output mới nếu dữ liệu đã được duyệt.

Nếu SAM3 không tìm được hoặc chọn sai instance, tạo overrides JSON rồi chạy lại
trước khi manifest đã hoàn tất:

```json
{
  "remove_pico_EXAMPLE": {
    "object_prompt": "the cup on the left",
    "candidate_index": 0,
    "box": [0.3, 0.5, 0.2, 0.4]
  }
}
```

Box là `[center_x, center_y, width, height]` chuẩn hóa 0..1, tùy chọn.
`candidate_index` là index các file `candidate_*.png`, tùy chọn; bỏ thì chọn score cao nhất.

```bash
python core/prepare_preservation.py --overrides overrides.json
```

Khi manifest đã hoàn tất, có thể copy một `candidate_N.png` phù hợp thành
`object_mask.png`, hoặc sửa mask nhị phân bằng công cụ ảnh. Không đổi kích thước.
Sau đó dựng lại overlay và duyệt lại tất cả mask:

```bash
python core/prepare_preservation.py --rebuild-overlays
```

Mở `output/preservation_data/mask_review.html` xem original | scribble | mask overlay.
Trong `mask_review.csv`, đặt `approved=1` cho từng mask đạt yêu cầu; ghi sửa mask
vào notes. Giữ nguyên cột hash. Benchmark từ chối mask pending/rejected, asset bị
đổi hash và sai kích thước. `--rebuild-overlays` reset approvals.

Scribble là một vòng nét đỏ quanh bounding box mask, cùng thuật toán/độ dày cho
các mẫu. Đây là so sánh chiến lược chỉ dẫn, có khác phần prompt giải thích overlay;
mask overlay không phải hard inpainting và không khóa pixel ngoài mask.

## 3. Smoke test rồi chạy đủ 60 output

```bash
conda activate preservation-flux
python core/benchmark_preservation.py --validate-only
python core/benchmark_preservation.py --limit 2 --output output/preservation_smoke
python core/benchmark_preservation.py --output output/preservation_20
```

Kiểm tra 6 output smoke trước khi chạy đủ. SAM3 và FLUX chạy ở process riêng, không
cùng chiếm VRAM. Nếu cần, thêm `--offload` cho cả smoke và pilot; không chỉ thay
offload riêng một nhánh. Có thể dùng `--seeds 42 123 2026` cho 180 output.
Thư mục output phải mới/rỗng để không ghi đè kết quả; run lỗi trả exit code 1,
ghi CSV sau từng lượt rồi tiếp tục các lượt còn lại. Run config chứa GPU, cache,
versions, model revision, manifest hash, số reference=1, seeds và thông số inference.

Kết quả:

- `results.csv`: PSNR từng ảnh/seed/nhánh, thời gian, peak allocated VRAM và lỗi.
- `report.md`: mean/median và paired delta so với text; chỉ dùng bộ ba đủ thành công.
- `gallery.html`: original | mask | text output | scribble output | mask output.
- `review.html`: chấm removal_success, marker_clear, background_natural, outside_preserved;
  xuất review.csv bằng nút trên trang, lưu cùng kết quả để đối chiếu PSNR.

PSNR so output thô với original sạch, chỉ lấy pixel **ngoài cùng mask SAM3 đã duyệt**.
Điểm phụ dùng mask giãn vuông bán kính 8 px tại độ phân giải inference. MSE=0 ghi
`inf`, không chặn thành 99. Khi có inf, báo riêng số lượng và median; paired delta
chỉ tính cặp hữu hạn. Nếu có nhiều seed, n là số cặp ảnh/seed, không phải số ảnh độc lập.
Nét scribble nằm ngoài object mask vẫn thuộc vùng chấm primary: model cần xóa nét
và khôi phục những pixel đó. PSNR cao không chứng minh xóa thành công.

20 ảnh có thể gồm split train gốc nhưng model không dùng LoRA. Không gọi đây là
held-out LoRA evaluation: repo chỉ có 15 mẫu removal validation/test trong metadata.
Muốn thử LoRA cần bộ held-out riêng và giữ protocol/model thành bảng khác.

## Kiểm tra CPU

```bash
python -m unittest discover -s tests -p test_preservation.py -v
```

Không cần tải weights hoặc CUDA. GPU forward chưa được chứng minh bởi CPU mocks;
chỉ nghiệm thu GPU sau khi smoke SAM3/FLUX chạy trên RTX 5090 thực tế.
