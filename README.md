# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường

**Bản của Đinh Quang Lâm · MSSV 202602875:** đã triển khai notebook/code và kiểm tra dữ liệu/CPU. **Chưa có E0–E4 chạy GPU; chưa đủ điều kiện nộp.** Xem [RUNNING.md](RUNNING.md) và [mở notebook Colab T4](https://colab.research.google.com/github/qlam1208/K4-Track-4-Day10-VisionLanguageModel/blob/main/lab_day10.ipynb). `report/diagnostics` và ảnh GT chỉ là QA, không phải kết quả model.

> Track 4 · Ngày 10 · *Nhãn, auto-labeling và Vision–Language Model* · **bài cá nhân** · ≈ 4 giờ trên **Google Colab T4 (miễn phí)**
> Bạn **tự viết** phần huấn luyện và suy luận theo pseudo code trong [`GUIDE.md`](GUIDE.md). Công cụ chia dữ liệu và chấm điểm do giảng viên viết sẵn và **không được sửa**.

| File | Dùng để làm gì |
|---|---|
| `README.md` (file này) | Bức tranh toàn lab, dữ liệu, môi trường, cách nộp |
| [`GUIDE.md`](GUIDE.md) | Từng bước + pseudo code + câu hỏi |
| [`RUBRIC.md`](RUBRIC.md) | Thang điểm 100 (+10) |
| [`REPORT.md`](REPORT.md) | Mẫu báo cáo 10 câu |
| `lab_day10.ipynb` | Khung notebook để bạn điền mã |
| `configs/e0…e5.yaml` | Siêu tham số từng thí nghiệm |
| `scripts/prepare_data.py` | Tải dữ liệu, chia split |
| `scripts/score.py` | **Trọng tài**: chấm `predictions.jsonl` |
| `scripts/check_submission.py` | Kiểm tra bài trước khi nộp |

---

## 1. Mục tiêu học tập

Sau bài lab, bạn có thể:

1. Mô tả được **ViT → projector → LLM** của Qwen3-VL-4B, và nói rõ khối nào bạn đã cập nhật trong từng thí nghiệm.
2. Finetune VLM bằng **QLoRA** và **so sánh bằng số** các cách đóng băng/mở khối (LoRA LLM; thêm projector; thêm ViT).
3. Giải thích vì sao phải chia dữ liệu **theo phiên quay** (không chia ngẫu nhiên) và vì sao **lệch thang toạ độ 0–1000** làm hỏng kết quả.
4. Đọc F1 **kèm khoảng tin cậy** trên một tập test nhỏ và biết kết luận nào còn đứng vững.
5. (Tuỳ chọn) Viết hàm **reward IoU** cho RL với phần thưởng kiểm chứng được.

## 2. Bài toán

Ảnh dashcam 640×360 → model trả **một JSON**:

```json
[{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}]
```

Toạ độ là **số nguyên trên thang 0–1000** (tương đối so với chiều rộng/cao ảnh), đúng quy ước grounding của Qwen3-VL trong slide Day 10. Ba lớp: `pothole` (ổ gà), `crack` (nứt), `manhole` (nắp cống). Mỗi ảnh 640×360 chiếm **220 token** vision.

## 3. Dữ liệu

Kaggle [`lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes), giấy phép **MIT**.

| | |
|---|---|
| Ảnh | 2.009 ảnh JPG 640×360 (dashcam, quay tháng 2–3/2025) |
| Box | 4.737 (crack 2.519 · pothole 1.261 · manhole 957), trung bình 2,4 box/ảnh |
| Độ khó | Box nhỏ: diện tích trung vị chỉ ≈ 0,7% khung hình |
| Nhãn dùng | `labels-YOLO/` (`class x_center y_center w h`, chuẩn hoá [0,1]) |

**Các ảnh là frame cắt từ video**, nên hai frame liền kề gần như trùng nhau. Nếu chia ngẫu nhiên, ảnh gần giống nhau sẽ nằm ở cả train và test (**rò rỉ**). `prepare_data.py` gom ảnh theo **phiên quay** (ngày + giờ trong tên file) và gán **cả phiên** vào một split. Split này **cố định**: ai chạy cũng ra cùng kết quả.

| Split | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|
| train | 1.422 | 1.074 | 1.750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

Lưu ý: split theo phiên nên phân bố lớp **không đều** (val có rất ít pothole). Đó là tình huống thật, và là lý do bạn phải đọc **F1 từng lớp** và **khoảng tin cậy**. Dùng **val để gỡ lỗi**, chỉ chấm **test** khi nộp kết quả.

## 4. Các thí nghiệm

| Mã | Setup | Bắt buộc |
|---|---|---|
| E0 | Zero-shot, không huấn luyện | ✔ |
| E1 | LoRA chỉ LLM; ViT và projector đóng băng | ✔ |
| E2 | E1 + mở đầy đủ projector (`visual.merger`, `visual.deepstack_merger_list`, ≈ 109M tham số) | ✔ |
| E3 | E2 + LoRA trên các khối attention/MLP của ViT | ✔ |
| E4 | Ablation: chọn **một** biến (rank / kích thước dữ liệu / thang toạ độ) | ✔ (chọn 1) |
| E5 | GRPO với reward IoU | tuỳ chọn, +10 điểm |

Mọi setup dùng **QLoRA 4-bit** (NF4). Riêng ViT và `lm_head` **không lượng tử hoá** (xem GUIDE bước 3 để biết vì sao).

## 5. Chuẩn bị (≈ 30 phút)

1. Mở `lab_day10.ipynb` trên Colab: Runtime → *Change runtime type* → **T4 GPU**. Mount Google Drive để lưu adapter (session miễn phí có thể bị ngắt).
2. Tạo **Colab Secrets** tên `KAGGLE_USERNAME` và `KAGGLE_KEY` bằng tài khoản Kaggle của bạn. **Không ghi khoá vào notebook hay commit.**
3. `pip install -r requirements-colab.txt`, rồi `python scripts/prepare_data.py --kaggle`.
   Không có tài khoản Kaggle? Tải zip dataset về tay và dùng `--zip <đường dẫn>` hoặc `--source <thư mục đã giải nén>`.
4. `python -m pytest -q` để chắc công cụ của đề chạy đúng trên máy bạn.

## 6. Yêu cầu và cách nộp

- Colab **T4 16 GB**. Mỗi lần huấn luyện ước tính **20–30 phút** (300 bước; con số ước lượng, hãy ghi thời gian thật của bạn vào báo cáo). Nếu thấy hết VRAM hoặc quá chậm, xem bảng "Khi gặp sự cố" trong GUIDE.
- Nộp **repo GitHub công khai** (≥ 5 commit) gồm: `report/evidence/e0–e3.json`, ít nhất một `e4*.json`, `REPORT.md` đã điền, ảnh minh hoạ, notebook/mã của bạn.
- Trước khi nộp: `python scripts/check_submission.py` không còn FAIL.
- **Không commit**: `data/`, `work/`, `.env`, trọng số (`*.safetensors`, `*.bin`, `*.pt`).
- **Không sửa** `scripts/` và `tests/` của đề.
- Làm cá nhân: bạn có thể thảo luận ý tưởng, nhưng mã, số liệu và báo cáo phải là của bạn. Giảng viên đối chiếu các repo với nhau và có thể gọi **vấn đáp 5 phút**.
