# Scheduling Backtracking vs PSO

Chuong trinh minh hoa giai bai toan xep lich bang hai phuong phap:

- Backtracking tren mien thoi gian roi rac (tim kiem toan bo, co cat nhanh).
- Particle Swarm Optimization (PSO) de toi uu tong do tre, co xu ly phat khi trung lich.

## Cach chay

Yeu cau Python 3.8+. Chay giao dien Tkinter:

```bash
python app.py
```

## Su dung giao dien

- Nhap cong viec (ten, thoi luong, deadline, release) va bam **Them**.
- Co the nhan **Demo** de nap san du lieu mau.
- Chinh **Horizon** (tong thoi gian), **Buoc thoi gian**, **Swarm size**, **So vong lap**.
- Bam **Chay Backtracking** hoac **Chay PSO** de xem lich. Nut **So sanh** chay ca hai va hien tong do tre, thoi gian chay.

## Ghi chu thuat toan

- **Backtracking**: sap xep cong viec theo deadline, thu cac moc bat dau voi buoc `step`, cat nhanh khi tong do tre hien tai vuot muc tot nhat.
- **PSO**: moi hat bieu dien vector thoi diem bat dau, ham muc tieu = tong do tre + phat trung lich; sau khi hoi tu, thoi gian bat dau duoc sap xep va day de khong trung nhau.

