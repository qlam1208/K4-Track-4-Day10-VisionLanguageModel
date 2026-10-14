# Ảnh kiểm tra và phân tích lỗi

`gt_train_*.png`: sáu ảnh TRAIN với nhãn gốc chuyển YOLO → norm1000 → pixel,
**GT only**, chưa có dự đoán model. Nguồn ảnh/nhãn:
[Road damage dataset của Lorenzo Arcioni](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes).

Sau khi có raw E1 thật, `student_vlm.reporting.prediction_figures` tạo
`e1_error_*.png` với GT xanh/pred đỏ và `captions.json` chứa image_id, F1 ảnh và raw.
Ảnh GT không thay thế ảnh dự đoán đúng/sai trong RUBRIC.
