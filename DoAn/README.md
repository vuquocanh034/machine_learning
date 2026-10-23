# Phân khúc khách hàng bán lẻ trực tuyến bằng RFM và học không giám sát

Đồ án cuối kỳ môn Máy học – Trường Đại học Sài Gòn. GV hướng dẫn: Đỗ Như Tài.
Nhóm: Vũ Quốc Anh, Nguyễn Minh Huy, Trần Huỳnh Phát, Đinh Công Thành.

## 1. Giới thiệu
Phân khúc khách hàng trên dữ liệu giao dịch Online Retail bằng RFM và gom cụm (K-Means, Hierarchical, DBSCAN), so sánh với baseline phân khúc theo luật RFM. Chia dữ liệu theo thời gian (mốc `CUTOFF = 2011-09-09`) để tránh data leakage và kiểm chứng cụm bằng hành vi mua ở giai đoạn sau.

## 2. Dữ liệu
- Nguồn: UCI Machine Learning Repository – *Online Retail* (Chen, 2015): https://archive.ics.uci.edu/dataset/352/online+retail
- Đặt file `Online_Retail.xlsx` vào `data/raw/` (thư mục này không được commit).
- _(Mô tả chi tiết các cột và đặc điểm dữ liệu sẽ bổ sung sau khi hoàn thành EDA.)_

## 3. Cài đặt
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Kiểm tra nhanh:
```bash
python -c "from src.data_loader import load_raw; print(load_raw().shape)"
```
Lần đầu đọc file Excel mất khoảng vài chục giây; kết quả được cache ở `data/interim/` (không commit).

## 4. Cách chạy
_(Bổ sung thứ tự chạy các notebook và demo Streamlit ở tuần 5.)_

## 5. Cấu trúc repo
```
├── data/ (raw, interim, processed)
├── notebooks/        # 01_..., 02_..., ...
├── src/              # config, data_loader, cleaning, features, baseline, clustering, evaluation
├── models/           # scaler, mô hình cuối
├── app/              # streamlit_app.py
├── reports/          # figures, tables
└── docs/             # report, slides, notes, ai_usage
```

## 6. Tái lập
`SEED = 42`, hằng số dùng chung trong `src/config.py`. Mọi notebook chạy được bằng Restart & Run All.

## 7. Công cụ AI hỗ trợ
Mỗi thành viên ghi nhật ký tại `docs/ai_usage/<tên>.md`. Tổng hợp sẽ được cập nhật ở phần này khi hoàn thành.
