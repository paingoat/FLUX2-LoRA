# EditLoRA removal với hai input

Lần đầu dùng RunPod: đọc [chọn template, screen, upload và reconnect](runpod.md) trước. Repo Git chỉ chứa code/metadata; ảnh dataset upload riêng bằng archive tạo bởi `core/package_dataset.py`.

Model luôn nhận `[original, reference]`: original là ảnh gốc sạch vẫn còn vật thể; reference là cùng ảnh đó có vòng tròn/nét chỉ dẫn. Target là ảnh sau khi xóa. Thứ tự và phần giải thích vai trò hai ảnh giống nhau ở train, preview, benchmark và Gradio.

Train: `FLUX.2-klein-base-4B`, rank/alpha 32, resolution 512, LR 1e-4, 1500 steps, batch 1 và accumulation 4. Inference mặc định: distilled `FLUX.2-klein-4B`, 4 steps, guidance 4, seed 42. Preview trong lúc train dùng base, 20 steps, không phải cấu hình benchmark distilled.

## 1. Mapping original và chuẩn bị dữ liệu

Chạy từ gốc repo (thư mục chứa `core/`, `data/`, `samples/`):

```bash
python core/prepare_dataset.py --originals-root samples/remove
```

`mapping.csv` nối `index` với `sample_id`. Script lấy `samples/remove/<sample_id>/original.png`, kiểm tra kích thước và tỷ lệ khác biệt với reference, rồi ghi vào `data/removal_dataset/original/<index>.png`. Giữ nguyên reference vòng tròn đỏ, target và caption nguồn; caption trong thư mục train được tạo bằng prompt hai input thống nhất.

Kiểm tra alignment là kiểm tra thô: tỷ lệ pixel khác >8 phải trong (0, 12%]. Không thay thế xem ảnh trực quan. `mask/` và scribble_style cũ không được dùng làm vùng edit.

Sau khi mapping trên máy local, có thể chỉ upload `data/removal_dataset` lên pod. Trên pod chạy lại để tạo manifest với đường dẫn của pod:

```bash
python core/prepare_dataset.py
```

Output mặc định `/workspace/datasets/removal_dual/`:

- `original/`, `reference/`, `target/`: 85 mẫu train, cùng tên stem.
- `sample_controls/`: hai input cho 4 mẫu validation (`0015`, `0022`, `0040`, `0057`).
- `eval/manifest.json`: 7 validation + 8 test. Không có ảnh held-out trong folder train.
- `dataset_card.json`: thứ tự reference, prompt, split và thống kê alignment.

Kiểm tra local không cần GPU:

```bash
python core/prepare_dataset.py --originals-root samples/remove --output .local/removal_dual
```

## 2. Setup và train trên pod

Tạo `.env` theo `.env.example`, điền HF_TOKEN nếu cần, rồi:

```bash
bash core/setup.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
source core/env.sh
conda activate klein
python core/train.py --check-only
python core/train.py --smoke-steps 2
```

Wrapper kiểm tra AI Toolkit có hỗ trợ `control_path_1`, `control_path_2` và structured `sample.samples`. Nếu báo trainer cũ, cập nhật checkout `/workspace/ai-toolkit` và dependencies trước khi chạy tiếp. Config hai input được đối chiếu với mã nguồn AI Toolkit: https://github.com/ostris/ai-toolkit/blob/main/toolkit/config_modules.py

Smoke run dùng output riêng `/workspace/output/removal_dual_4b_smoke`. Xem preview, xác nhận đủ hai input và theo dõi VRAM trước khi chạy đầy đủ:

```bash
python core/train.py
```

Output mới: `/workspace/output/removal_dual_4b/`, tách khỏi run một input cũ. Log `train.log`; checkpoint mỗi 250 steps. File cuối `removal_dual_4b.safetensors` tương ứng step cuối, có thể không có số ở tên. Tiếp tục run hai input bằng:

```bash
python core/train.py --resume
```

Không resume checkpoint cũ một input để so sánh thí nghiệm này. Nếu OOM, giảm tải sampling hoặc cấu hình model sau khi xem log; hai reference có thể dùng nhiều VRAM hơn.

## 3. Chọn checkpoint bằng validation

```bash
conda activate klein-infer
python core/benchmark.py --baseline --split validation \
  --output /workspace/output/removal_dual_4b/benchmark_validation
```

Chạy tất cả checkpoint và baseline không LoRA trên 7 validation. Cùng input, prompt, seeds và cấu hình inference. Mặc định resolution 512 + 1024; có thể dùng `--resolutions 512` hoặc `--seeds 42 123 2026` cho cả hai nhánh.

Kết quả gồm `results.csv`, `run_config.json`, `report.md`, ảnh output, contact sheet từng mẫu và `review.html`. Không ghi đè một folder đã có results.csv; chọn output mới khi chạy lại. Lỗi OOM được giữ trong results.csv và báo coverage, không âm thầm bỏ mẫu.

Metric tự động:

| Metric | Vùng và ý nghĩa |
| --- | --- |
| scribble_residue ↓ | Tỷ lệ vị trí nét rõ còn màu gần màu marker; không phát hiện nét bị dịch vị trí. |
| preserve_psnr ↑ | So output với original sạch ngoài vùng edit ước lượng từ nét. |
| preserve_ssim ↑ | Trung bình bản đồ SSIM trên các cửa sổ 7x7 nằm hoàn toàn ngoài vùng edit và biên ảnh. |
| target_lpips ↓ | LPIPS toàn ảnh so với target, chỉ tham khảo vì target có thể thay đổi ngoài vùng xóa. |
| edit_lpips ↓ | LPIPS crop bounding box vùng edit, có cả context; không phải LPIPS mask theo từng pixel. |

Vùng edit được ước lượng bằng đóng nét, convex hull theo thành phần liên thông và giãn nhẹ. Đây không phải segmentation vật thể; kiểm tra contact sheet và edit_fraction khi giải thích preservation.

## 4. Chấm thành công bằng người

Mở `review.html` (tải cả folder benchmark về máy nếu chạy trên pod). Thứ tự mẫu được xáo trộn, không hiện checkpoint. Cột ảnh: original | reference | output | target nếu có. Chấm 0/1 cho bốn tiêu chí:

- removal_success: xóa hết đúng vật thể được chỉ định, không còn mảnh nhận ra được.
- marker_clear: không còn nét chỉ dẫn ở bất kỳ đâu.
- background_natural: nền phục hồi hợp lý, không có vết vá/méo rõ.
- outside_preserved: không thay đổi rõ ràng nội dung ngoài chỉnh sửa được yêu cầu.

Page lưu tạm trong browser, có import/export CSV. Bấm Download review.csv rồi:

```bash
python core/review_benchmark.py /path/to/results.csv --review /path/to/review.csv
```

`human_report.md` báo từng tỷ lệ, số đếm và Overall Success Rate (đạt cả bốn tiêu chí), riêng theo dataset/split/checkpoint/resolution. Review chưa đủ được để pending. Chọn checkpoint trên validation bằng removal/overall success và preservation; metric tự động không tự chứng minh đã remove thành công. Không so sánh các nhánh có coverage khác nhau.

## 5. Test nội bộ và VIBE Removal

Sau khi chốt checkpoint bằng validation, đánh giá 8 ảnh test:

```bash
python core/benchmark.py --baseline --split test \
  --checkpoint /workspace/output/removal_dual_4b/CHOSEN.safetensors \
  --output /workspace/output/removal_dual_4b/benchmark_test
```

Chuẩn bị toàn bộ VIBE Removal (100 mẫu trước khi kiểm tra overlap):

```bash
python core/prepare_vibe.py --download
```

Script tải riêng Removal.json, ảnh nguồn và lớp chỉ dẫn cần thiết. Revision được cố định bằng commit SHA trong lần tải và ghi vào dataset_card.json. Có thể tái lập bằng `--revision <sha>`. Nếu dataset đã có sẵn, dùng `--root /path/to/VIBE-Benchmark` không có `--download`.

VIBE được ghép từ original + visual instruction gốc, không vẽ lại vòng tròn. Prompt giữ input_prompt của VIBE, thêm cùng phần giải thích hai ảnh cho baseline và LoRA. Loại ảnh trùng pixel chính xác với toàn bộ original nội bộ; ảnh gần trùng cần xem thủ công. VIBE Removal hiện không có target sau chỉnh sửa nên LPIPS để trống.

```bash
python core/benchmark.py --baseline --split external \
  --manifest /workspace/datasets/vibe_removal/manifest.json \
  --checkpoint /workspace/output/removal_dual_4b/CHOSEN.safetensors \
  --output /workspace/output/removal_dual_4b/benchmark_vibe
```

VIBE chỉ dùng đánh giá cuối, không chọn checkpoint. Báo riêng VIBE và test nội bộ, không gộp trung bình. Chấm review.html giống bước 4. Đây là thiết lập VIBE hai input với rubric của dự án; không phải điểm leaderboard chính thức. Pipeline judge chính thức tham khảo https://github.com/hwanyu112/VIBE-Benchmark . Chưa tự động gọi dịch vụ judge bên ngoài.

## 6. Gradio và kiểm tra local

```bash
conda activate klein-infer
python core/app.py
```

Upload original sạch và vẽ trong editor để giữ được cả background và composite. Gallery test cũng dùng hai input. App bật share như trước.

Kiểm tra CPU, không tải weights:

```bash
python -B -m unittest discover -s tests -v
```

Cần Pillow, NumPy, OpenCV, scikit-image và PyYAML. Test kiểm tra mapping/split, thứ tự input, metric vùng edit, VIBE schema/overlap và tổng hợp human review. GPU train/inference vẫn cần smoke run trên pod.
