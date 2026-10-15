# Chạy bài Day 10

**Hiện tại: triển khai và CPU/data checks đã có; E0–E4 chưa được chạy trên GPU.**
Không có CUDA trên máy chuẩn bị repo, và chưa chạy được thí nghiệm Colab. Vì vậy
`report/evidence/e0.json`…`e4*.json` chưa tồn tại; `check_submission.py` còn FAIL.
Các file `report/diagnostics/` và ảnh `gt_train_*` là QA dữ liệu/cấu trúc, không phải kết quả model.

Mở [notebook trên Colab](https://colab.research.google.com/github/qlam1208/K4-Track-4-Day10-VisionLanguageModel/blob/main/lab_day10.ipynb),
đăng nhập, chọn **T4 GPU**, rồi chạy các cell lần lượt. Các dependency giữ đúng bản ghim trong đề.
Notebook tải repo và zip dataset công khai, kiểm tra split/mask, chạy E0–E3, chọn base E4 bằng VAL,
chạy hai rank, chấm bằng `scripts/score.py` gốc và tạo ảnh lỗi. Drive mount cần bạn xác thực;
checkpoint mỗi 100 bước lưu `latest` mỗi setup trên Drive để tránh vượt dung lượng vì nhiều bản trùng.

## CLI trên máy có CUDA

```bash
python -m pip uninstall -y torchao  # optional Colab package; NF4 uses bitsandbytes
pip install -r requirements-colab.txt pytest
python scripts/prepare_data.py --zip 'https://www.kaggle.com/api/v1/datasets/download/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes'
python -m student_vlm inspect
python -m student_vlm all --checkpoint-root /path/to/persistent/checkpoints
```

Hoặc chạy từng bước:

```bash
python -m student_vlm evaluate --setup E0
python -m student_vlm train --setup E1 --checkpoint-root /path/to/persistent/checkpoints
python -m student_vlm evaluate --setup E1 --checkpoint work/e1/step-0300 --split val
python -m student_vlm evaluate --setup E1 --checkpoint work/e1/step-0300 --split test
# Tương tự E2/E3; chọn base E4 từ work/validation/*.json.
python -m student_vlm train --setup E4_rank_4 --base E2
python -m student_vlm evaluate --setup E4_rank_4 --base E2 --checkpoint work/e4_rank_4/step-0300
# Chỉ dùng --base E2 nếu E2 thật sự được chọn bằng VAL; chạy cả rank 64.
python -m student_vlm report
python scripts/check_submission.py
```

Resume sau ngắt:

```bash
python -m student_vlm train --setup E2 --resume /path/to/checkpoints/e2/latest
```

Resume đòi cùng config và dataset version; khôi phục optimizer, scaler, scheduler, RNG và vị trí batch.
Không load adapter mà quên `projector.pt`; loader báo lỗi nếu E2/E3 thiếu projector.
Nếu GPU hết bộ nhớ, dùng batch=1 và grad_accum=8 trong một config mới, áp dụng nhất quán cho mọi setup
và ghi rõ thay đổi; không coi partial run là bằng chứng final. Không chọn LR/rank theo TEST.
Debug 40 ảnh VAL có thể dùng `evaluate --split val --limit 40`; không tạo evidence final.

## Kiểm thử CPU

Cài torch/torchvision CPU cùng phiên bản tương thích, transformers/peft/accelerate như requirements của đề,
và numpy/pytest/Pillow/PyYAML. Sau đó:

```bash
python -m pytest -q tests student_tests \
  --deselect=tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions
```

Test được deselect chỉ kiểm tra REPORT còn placeholder; nó mâu thuẫn với check_submission yêu cầu đã xóa
placeholder. **Không sửa tests/scripts để bỏ mâu thuẫn**. Test CPU thực sự kiểm tra prefix/mask/padding,
module matching trên model 4B meta, projector round-trip, backward qua Qwen nhỏ và ablation chọn bằng VAL.
Không kiểm thử được NF4 CUDA, tốc độ hay peak VRAM của model 4B tại máy này.

## Trước khi nộp

Chạy đủ E0/E1/E2/E3, hai rank E4; giữ nguyên raw. Sao chép `work/*/{loss.jsonl,run.json}` sang `report/logs/`
nhưng không copy checkpoint. Sau `student_vlm report`, đọc ảnh lỗi và hoàn tất các kết luận thực nghiệm
trong REPORT.md. `check_submission.py` phải không còn FAIL, repo có ≥5 commit.
Commit `REPORT.md`, notebook/code, `report/evidence/`, `report/figures/` và log nhẹ; không commit data/work,
Kaggle/HF key, optimizer, adapter hay projector tensors. Notebook tạo zip kết quả nhẹ và lưu Drive.

## Lỗi torchao trên Colab

Nếu PEFT báo `Found an incompatible version of torchao ... 0.10.0`, chạy lại cell
môi trường của notebook mới nhất: cell gỡ torchao trước khi import model và chạy test.
Torchao là backend tùy chọn; bài dùng bitsandbytes NF4 nên không cần cài/nâng cấp nó.
Giữ nguyên phiên bản PyTorch/CUDA có sẵn trên Colab. Nếu đã import PEFT/torchao trong
kernel trước đó, khởi động lại phiên Python sau khi gỡ rồi chạy lại từ đầu.
Cell cũng dùng `git pull --ff-only` để cập nhật checkout cũ mà không ghi đè chỉnh sửa local.
