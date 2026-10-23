# Phân khúc khách hàng bán lẻ trực tuyến bằng RFM và học không giám sát

Đồ án cuối kỳ môn Máy học – Trường Đại học Sài Gòn. GV hướng dẫn: Đỗ Như Tài.
Nhóm: Vũ Quốc Anh, Nguyễn Minh Huy, Trần Huỳnh Phát, Đinh Công Thành.

## 1. Giới thiệu
Phân khúc khách hàng trên dữ liệu giao dịch Online Retail bằng RFM và gom cụm (K-Means, Hierarchical, DBSCAN), so sánh với baseline phân khúc theo luật RFM. Chia dữ liệu theo thời gian (mốc `CUTOFF = 2011-09-09`) để tránh data leakage và kiểm chứng cụm bằng hành vi mua ở giai đoạn sau.

## 2. Dữ liệu

Nguồn: Chen, D. (2015), *Online Retail*, [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail), [DOI: 10.24432/C5BW33](https://doi.org/10.24432/C5BW33), giấy phép CC BY 4.0. Dữ liệu giao dịch của doanh nghiệp bán lẻ trực tuyến tại Anh; mỗi dòng là một mặt hàng trong hóa đơn, không phải một khách hàng hay một đơn hàng độc lập. Đơn giá tính bằng bảng Anh (GBP).

Tải file từ UCI, đặt tại `data/raw/Online_Retail.xlsx`. File tải về có thể tên `Online Retail.xlsx`; đổi tên hoặc sao chép thành tên có dấu gạch dưới để tương thích `load_raw()`. Không commit dữ liệu gốc hoặc cache.

| Cột | Ý nghĩa và lưu ý |
| --- | --- |
| `InvoiceNo` | Mã hóa đơn, giữ dạng chuỗi; tiền tố `C` biểu thị hủy. Một hóa đơn có nhiều dòng hàng. |
| `StockCode` | Mã mặt hàng, giữ dạng chuỗi vì có hậu tố chữ và các mã phí/điều chỉnh. |
| `Description` | Mô tả mặt hàng; có giá trị thiếu, có thể thay đổi theo cùng mã. |
| `Quantity` | Số lượng trong dòng giao dịch; giá trị âm có thể là hoàn/hủy hoặc điều chỉnh kho. |
| `InvoiceDate` | Ngày giờ giao dịch, đọc bằng `datetime`; dùng chia theo thời gian. |
| `UnitPrice` | Đơn giá GBP; cần kiểm tra giá 0, âm và các bút toán có giá rất lớn. |
| `CustomerID` | Mã khách hàng, đọc thành `Int64` nullable; là định danh, không phải biến liên tục; không suy đoán ID bị thiếu. |
| `Country` | Quốc gia khách hàng; dùng mô tả địa lý và phân tích hồ sơ cụm. |

**Kết quả kiểm kê từ file thực:** [notebook EDA sản phẩm/quốc gia](notebooks/01_eda_product_thanh.ipynb) ghi nhận 541.909 dòng, 8 cột, từ `2010-12-01 08:26:00` đến `2011-12-09 12:50:00`, 38 quốc gia. Toàn file có 1.454 `Description` thiếu, 135.080 `CustomerID` thiếu và 5.268 dòng trùng hoàn toàn tính sau lần xuất hiện đầu. Đây là kiểm kê nguồn; không dùng thông tin giai đoạn kiểm chứng để thiết kế bộ lọc hoặc chọn mô hình.

**EDA trước cutoff:** 332.447 dòng có `InvoiceDate < 2011-09-09`; trong đó có 1.181 mô tả thiếu/rỗng, 6.928 dòng số lượng âm, 2 dòng giá âm và 1.830 dòng giá 0. `Quantity` nằm trong [−74.215; 74.215], `UnitPrice` trong [−11.062,06; 38.970,00] GBP. Cặp bán–hủy mã `23166`, khách `12346`, số lượng ±74.215 cùng ngày cho thấy không thể đồng nhất tổng bán dương với doanh thu ròng.

Các mã `POST`, `DOT`, `C2`, `D`, `M`/`m`, `BANK CHARGES`, `AMAZONFEE`, `B`, `CRUK`, `S` và `gift_*` xuất hiện trong bảng kiểm tra kèm mô tả thật. Chúng được tách khỏi xếp hạng hàng hóa theo lý do phí, vận chuyển, điều chỉnh, mẫu hoặc phiếu quà tặng; không áp dụng quy tắc xóa mọi mã có chữ, vì `DCGSSGIRL`, `DCGSSBOY`, `PADS` và một số `DCGS...` là hàng hóa. Quy tắc này chỉ phục vụ EDA, không thay pipeline làm sạch hay làm mất dữ liệu để tính `CancelRate` sau này.

Bộ lọc nhẹ EDA giữ dòng không thuộc hóa đơn `C`, `Quantity > 0`, `UnitPrice > 0` và không thuộc các mã phi hàng hóa đã rà soát: còn 323.032 dòng quan sát. Giữ thiếu ID/mô tả, dòng trùng và ngoại lệ lớn để không xóa dữ liệu máy móc. Top số lượng là mã `23166`, top doanh thu bán dương là `22423`; doanh thu này chưa trừ hoàn/hủy. Notebook có kiểm tra độ nhạy khi tạm bỏ dòng trùng và chỉ rõ ảnh hưởng của lượt bán 74.215 đơn vị.

**Quốc gia và mẫu số:** UK chiếm 91,43% số dòng toàn file chưa lọc; 91,20% số dòng quan sát chưa lọc. Trên 323.032 dòng bán dương hàng hóa quan sát, UK chiếm 91,44% số dòng và 83,92% doanh thu. Đề xuất giữ mọi quốc gia, gắn cờ ngoài UK để profiling thay vì tách mô hình ngay: tách làm giảm mẫu ngoài UK và đổi quần thể so sánh. Cờ này không tự ý thêm vào RFM/RFM_PLUS; nếu tổng hợp theo khách, chỉ dùng giao dịch trước cutoff và kiểm tra ID xuất hiện ở nhiều nước.

**Tái lập EDA:** chạy `notebooks/01_eda_product_thanh.ipynb` bằng Kernel → Restart & Run All từ gốc repo hoặc thư mục `notebooks/`, sau khi cài `requirements.txt`. Notebook gọi `load_raw(use_cache=False)` để tránh cache cũ, sử dụng `SEED`/`CUTOFF` và đường dẫn từ `src/config.py`; logic riêng tại `src/eda_product.py`. Mỗi biểu đồ có nhận xét tiếng Việt; 5 hình ở `reports/figures/eda_prod_*.png`, bảng đối soát và số liệu nguồn ở `reports/tables/eda_prod_*`, đặc biệt [tóm tắt số liệu](reports/tables/eda_prod_summary.json). Phiên bản thư viện của lần chạy được lưu trong notebook và JSON. Các giai đoạn tạo đặc trưng, scaler và tuning sau này chỉ fit trên quan sát; dữ liệu từ cutoff chỉ phục vụ RQ3.

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
