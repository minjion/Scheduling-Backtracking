# Scheduling – Backtracking vs PSO

Ứng dụng minh họa giải bài toán xếp lịch với hai thuật toán:
- **Backtracking** trên miền thời gian rời rạc (tìm kiếm toàn bộ, có cắt nhánh).
- **Particle Swarm Optimization (PSO)** tối ưu tổng độ trễ, phạt khi trùng lịch.

## Yêu cầu
- Python 3.8+ có Tkinter (cài từ https://www.python.org/downloads/).
- Không cần thư viện ngoài.

## Cài đặt (tùy chọn môi trường ảo)
```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# hoặc: source .venv/bin/activate  # Linux/macOS
```

## Chạy ứng dụng
```bash
cd "(ổ đĩa cài đặt):\\(tên thư mục cài đặt)" ví dụ cd d:\\Scheduling-Backtracking
python app.py
```
- Nếu lệnh `python` mở Microsoft Store, tắt App Execution Aliases trong Settings > Apps > Advanced app settings.
- Kiểm tra Tkinter: `python - <<\"PY\"\nimport tkinter; print('OK')\nPY`. Nếu lỗi, cài lại Python và bật tùy chọn “tcl/tk and IDLE”.

## Cách dùng
- Nhập công việc (Tên, Thời lượng, Deadline, Release), bấm **Thêm**; **Xóa** để bỏ mục chọn; **Demo** để nạp dữ liệu mẫu.
- Thiết lập **Horizon**, **Bước thời gian** (backtracking), **Swarm size**, **Số vòng lặp** (PSO).
- Bấm **Chạy Backtracking** hoặc **Chạy PSO** để xem lịch; **So sánh** chạy cả hai, hiển thị thời gian chạy (ms) và tổng độ trễ, chỉ ra lời giải tốt hơn.

## Ghi chú thuật toán
- **Backtracking**: sắp xếp công việc theo deadline, thử các mốc bắt đầu với bước `step`, cắt nhánh khi tổng độ trễ hiện tại đã tệ hơn lời giải tốt nhất.
- **PSO**: mỗi hạt là vector thời điểm bắt đầu; hàm mục tiêu = tổng độ trễ + phạt trùng lịch; sau khi hội tụ, thời điểm được sắp xếp và đẩy để không trùng nhau.
