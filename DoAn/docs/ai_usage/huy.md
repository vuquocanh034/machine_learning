# Khai báo sử dụng AI — Nguyễn Minh Huy, tuần 3A (K-Means và tuning)

**Công cụ:** Claude (Anthropic), chạy trong môi trường Claude Code.

## AI đã làm gì

| Sản phẩm | Vai trò của AI |
|---|---|
| `src/clustering.py` (K-Means, lưới tuning, chọn top 3, độ nhạy ngoại lệ/đặc trưng, kiểm tra `n_init`) | viết mã theo yêu cầu và quy ước repo (`src/config.py`, `TUNING_SCHEMA`, `baseline.internal_metrics`) |
| `src/viz_kmeans.py`, hằng số K-Means trong `src/config.py` | viết mã |
| `notebooks/03_kmeans.ipynb` | viết mã và phần diễn giải markdown tiếng Việt; nhận xét dùng số liệu tính trong notebook |
| `docs/notes/notes_kmeans.md` | soạn nội dung từ kiến thức K-Means và kết quả notebook |

## Phần người phải kiểm tra

- Đọc lại các nhận xét ở mục 2, 4, 5, 6, 8 của notebook và đối chiếu với bảng/hình. Trong lúc làm, nhiều câu nhận xét bản nháp đã bị phát hiện **không khớp số liệu** và được sửa (ví dụ: nhận định "n_init = 50 là đủ" sai ở k = 10 nên nâng lên 200; docstring cũ nói k = 2 thắng cả ba metric ở mọi bộ đặc trưng, nhưng Davies–Bouldin của `RFM_PLUS` thấp nhất ở k = 9).
- Các quyết định mang tính phán đoán, cần nhóm đồng ý: chỉ xét k ≥ 3 khi chọn top 3; chọn k = 4 cho `RFM`; ngưỡng "cụm nhỏ" (2% hoặc 30 khách); khuỷu inertia bằng heuristic khoảng cách tới đường nối hai đầu mút.
- Thành viên viết phần kết luận RQ1/RQ2 cần tự đánh giá lại, không chép nguyên văn.

## Lưu ý về cách chạy notebook

Môi trường của AI không có Jupyter/`ipykernel` và không cài được, nên notebook được thực thi bằng một bộ chạy tối giản (chạy tuần tự các ô code trong một tiến trình mới, bắt output và hình matplotlib rồi ghi vào `.ipynb`). Output trong file là kết quả thật của mã, nhưng **chưa chạy bằng Jupyter**. Trước khi merge, người nhận cần mở notebook trong Jupyter và chạy Restart & Run All để xác nhận.

## Kiểm soát chất lượng đã làm

- `assert` trong notebook: đầu vào chỉ gồm `features_obs.csv` và scaler tuần 2; mã K-Means không đọc `data/processed/`; đúng schema, đủ 36 cấu hình, đủ file đầu ra; chạy lại một cấu hình cho đúng số đã ghi.
- Không dùng `future_outcome.csv`; không sửa file của thành viên khác.
