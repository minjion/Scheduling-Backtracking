# Lập lịch đơn máy — Backtracking vs GWO

Ứng dụng GUI nhỏ minh họa bài toán sắp xếp lịch đơn máy với hai thuật toán:
- **Backtracking** trên lưới thời gian rời rạc (DFS + cắt nhánh).
- **Grey Wolf Optimizer (GWO)** trên thời điểm liên tục, sau đó chiếu về lịch khả thi.

## Yêu cầu
- Python 3.8+ có Tkinter (đi kèm trình cài đặt Python chính thức).
- Không cần thư viện ngoài.

## Cài đặt (tùy chọn dùng môi trường ảo)
```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# hoặc: source .venv/bin/activate # Linux/macOS
```

## Chạy giao diện
```bash
python app.py
```
- Nếu lệnh `python` mở Microsoft Store, tắt App Execution Aliases (Settings > Apps > Advanced app settings).
- Kiểm tra Tkinter: `python - <<\"PY\"\nimport tkinter; print('OK')\nPY`.

## Cách dùng nhanh (GUI)
- Nhập công việc (Tên, Thời lượng, Deadline, Release), bấm **Thêm**; **Xóa** để bỏ mục chọn; **Demo** để nạp dữ liệu mẫu.
- Thiết lập **Horizon**, **Bước thời gian** (backtracking), **Pack size**, **Số vòng lặp** (GWO).
- Bấm **Chạy Backtracking** hoặc **Chạy GWO** để xem lịch; **So sánh** chạy cả hai, hiển thị thời gian chạy (ms), tổng độ trễ và thuật toán tốt hơn.

## So sánh hiệu năng nhiều kịch bản (batch)
Chạy nhanh các kịch bản mẫu, xem bảng tổng hợp độ trễ và thời gian chạy:
```bash
python benchmark.py
```
Kịch bản gồm: deadline gắt, release lệch pha, quá tải gần deadline và bộ ngẫu nhiên có seed cố định. Bảng kết quả cho biết tổng độ trễ và thời gian chạy của mỗi thuật toán trong từng kịch bản để so sánh.

## Thuật toán
- **Backtracking**: sắp xếp công việc theo deadline, thử các mốc bắt đầu với bước `step`, cắt nhánh khi chi phí hiện tại đã tệ hơn nghiệm tốt nhất.
- **GWO**: mỗi “sói” là vector thời điểm bắt đầu; hàm mục tiêu = tổng độ trễ + phạt vi phạm cửa sổ + phạt chồng lắp; cập nhật theo alpha/beta/delta, sau hội tụ sắp xếp và đẩy mốc để không chồng lắp rồi cắt về miền khả thi.
