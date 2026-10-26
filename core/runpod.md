# RunPod: chọn template, giữ session bằng screen và train

## 1. Chọn Pod và template

Khuyến nghị cho pilot hai input:

| Mục | Chọn |
| --- | --- |
| Loại | GPU Pod, On-Demand (non-interruptible) |
| GPU | 1 × L40S 48 GB hoặc RTX A6000 48 GB để có dư VRAM hơn |
| Tiết kiệm | RTX 4090 24 GB, chạy smoke test trước; chưa xác nhận đủ VRAM cho mọi preview/benchmark |
| Template | RunPod PyTorch **Official**, PyTorch 2.13 / CUDA 13.0 nếu có trong dropdown |
| Container disk | khoảng 30 GB |
| Persistent storage | khoảng 150 GB, mount tại `/workspace` |
| Kết nối | SSH terminal access; SSH over exposed TCP để upload SCP/SFTP |
| Jupyter | Tùy chọn, tiện upload file; không cần notebook để train |

Script setup tạo conda Python 3.12 riêng, nên Python cài sẵn của template không quyết định env train. Repo hiện cài torch cu130; chọn máy có driver hỗ trợ CUDA 13.0, kiểm tra `nvidia-smi` trước setup. Không cần template AI Toolkit có UI vì repo dùng trainer qua CLI.

Nếu muốn giữ dữ liệu khi xóa/thay Pod, tạo Network Volume và gắn lúc deploy. Volume disk thông thường giữ `/workspace` qua stop/restart nhưng bị xóa khi terminate Pod. `screen` giữ process khi SSH/mạng của bạn rớt; không giữ process nếu Pod bị stop/reboot/terminate.

Nguồn: [PyTorch templates của RunPod](https://www.runpod.io/blog/rebuilding-pytorch-templates), [storage](https://docs.runpod.io/pods/storage/types), [kết nối](https://docs.runpod.io/pods/connect-to-a-pod).

## 2. Vào screen trước khi setup

Mở Pod → Connect → SSH hoặc Web Terminal. Trong terminal trên Pod:

```bash
apt update
apt install -y screen git curl wget rsync build-essential
screen -S train
```

Từ đây chạy mọi bước setup/train trong screen. Kiểm tra:

```bash
nvidia-smi
df -h /workspace
echo "$STY"
```

`STY` có giá trị nghĩa là đang trong screen. Cài screen cần làm lại nếu container bị tạo lại; model/env/output vẫn ở `/workspace` nếu volume còn.

## 3. Clone code nhẹ

Trong screen:

```bash
cd /workspace
git clone --depth 1 --single-branch --branch multi-ref-editing \
  https://github.com/paingoat/FLUX2-LoRA.git
cd /workspace/FLUX2-LoRA
```

Branch mới chỉ chứa code, mapping, metadata và caption. Ảnh, cache, env, checkpoint và samples không đi theo Git. Dùng shallow clone như trên để không tải lịch sử commit cũ từng chứa ảnh. Nếu repo private, xác thực bằng SSH key hoặc credential manager; không ghi token trực tiếp trong URL.

## 4. Upload dataset riêng

Máy local đã có bộ original/reference/target được mapping. Tạo archive từ gốc repo local:

```powershell
python core/package_dataset.py
```

Archive `.local/removal_dataset.tar.gz` gồm 100 original, 100 reference, 100 target, 100 caption, mapping.csv và meta.jsonl. Không chứa samples toàn bộ, mask cũ hoặc VIBE. File `.sha256` đi kèm để kiểm tra sau upload.

Trong cửa sổ PowerShell **trên máy local**, thay POD_HOST và POD_SSH_PORT bằng thông tin SSH over exposed TCP trong Connect:

```powershell
scp -P POD_SSH_PORT .local/removal_dataset.tar.gz .local/removal_dataset.tar.gz.sha256 root@POD_HOST:/workspace/
```

Nếu mạng upload chập chờn, dùng SFTP client có hỗ trợ resume (ví dụ WinSCP) để tiếp tục file dở. `screen` không bảo vệ kết nối upload từ máy local; nó bảo vệ process đang chạy trên Pod.

Quay lại terminal trong screen trên Pod:

```bash
cd /workspace
sha256sum -c removal_dataset.tar.gz.sha256
tar -xzf removal_dataset.tar.gz -C /workspace/FLUX2-LoRA
cd /workspace/FLUX2-LoRA
```

Chỉ giải nén khi checksum OK. Có thể upload trong lúc setup ở bước 5 đang chạy để tiết kiệm thời gian thuê GPU; phải giải nén xong trước prepare/train.

## 5. Setup và train trong screen

```bash
cd /workspace/FLUX2-LoRA
cp .env.example .env
nano .env
```

Điền HF_TOKEN của bạn. Nếu không có nano, cài `apt install -y nano` hoặc sửa file bằng Jupyter. `.env` được Git ignore.

```bash
set -o pipefail
bash core/setup.sh 2>&1 | tee /workspace/setup.log
```

Chỉ tiếp tục khi setup thành công. Nếu lỗi tải package/weights do mạng, chạy lại setup trong cùng screen; các bước đã hoàn thành có marker để bỏ qua. Log nằm ở `/workspace/setup.log`.

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
source core/env.sh
conda activate klein
python core/prepare_dataset.py
python core/train.py --check-only
python core/train.py --smoke-steps 2
```

Kiểm tra preview và VRAM của smoke run. Nếu ổn, chạy đầy đủ:

```bash
python core/train.py
```

Checkpoint/log ở `/workspace/output/removal_dual_4b/`. Smoke output riêng tại `/workspace/output/removal_dual_4b_smoke/`. Không chạy hai process train cùng lúc trên một GPU. Muốn chạy lại smoke, đổi tên output smoke cũ hoặc dùng config tên run khác để không nối nhầm.

## 6. Rớt mạng và quay lại

Rời screen chủ động mà train tiếp: **Ctrl+A**, thả phím, rồi bấm **D**. Sau đó có thể đóng SSH/laptop.

Khi kết nối SSH trở lại cùng Pod:

```bash
screen -ls
screen -r train
```

Nếu báo session vẫn attached do kết nối cũ chưa thoát:

```bash
screen -d -r train
```

Không chạy thêm `screen -S train` hoặc lệnh train mới nếu session cũ vẫn chạy. `screen -ls` chỉ liệt kê; `screen -r train` mới vào lại.

Nếu **Pod đã reboot/stop** thì session screen/process cũ không còn. Sau khi Pod hoạt động lại:

```bash
apt update
apt install -y screen
screen -S train
cd /workspace/FLUX2-LoRA
source /workspace/miniconda3/etc/profile.d/conda.sh
source core/env.sh
conda activate klein
python core/train.py --resume
```

Resume dùng checkpoint mới nhất còn lưu, không khôi phục các step chưa checkpoint. Nếu không có checkpoint, wrapper báo bắt đầu từ step 0. Nếu Pod đã bị terminate với volume disk thường, cần dữ liệu/env/checkpoint từ bản sao lưu.

## 7. Đánh giá sau train

Tiếp tục trong screen hoặc tạo session riêng `screen -S eval` sau khi train kết thúc:

```bash
conda activate klein-infer
python core/benchmark.py --baseline --split validation \
  --output /workspace/output/removal_dual_4b/benchmark_validation
```

Chọn checkpoint bằng validation rồi chạy test/VIBE theo [guide.md](guide.md). VIBE tải trực tiếp trên Pod bằng `python core/prepare_vibe.py --download`, không cần upload từ local. Tải checkpoint muốn giữ về máy trước khi giải phóng storage.
