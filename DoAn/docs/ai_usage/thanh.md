# Nhật ký AI hỗ trợ — Đinh Công Thành

- Ngày: 22/10/2026 (Asia/Saigon).
- Công cụ: OpenAI Codex trong ứng dụng desktop.
- Phạm vi: `notebooks/01_eda_product_thanh.ipynb`, module riêng `src/eda_product.py`, mục 2 “Dữ liệu” trong README, kết quả `reports/figures/eda_prod_*` và `reports/tables/eda_prod_*`.

## Nội dung yêu cầu và AI hỗ trợ

Yêu cầu AI đọc repo và dùng `load_raw()` để EDA sản phẩm/quốc gia: xếp hạng số lượng/doanh thu, phân phối Quantity/UnitPrice, kiểm tra phí và mã đặc biệt dựa trên mô tả thật, Description thiếu, tỷ trọng UK, giải thích bộ lọc và kiểm soát leakage. AI hỗ trợ tổ chức module/notebook, soạn nhận xét tiếng Việt, tổng hợp số liệu README và kiểm tra chạy lại.

AI đã đọc README, `src/config.py`, `src/data_loader.py`, requirements và cấu trúc hiện có; đối chiếu nguồn UCI tại https://archive.ics.uci.edu/dataset/352/online+retail và DOI https://doi.org/10.24432/C5BW33. Số liệu được tính từ Excel trong notebook, không chép từ câu trả lời hoặc trí nhớ AI. Không thay đổi các hàm đọc/cấu hình và không viết phần clustering của thành viên khác.

## Quyết định cần người thực hiện hiểu và chịu trách nhiệm

- Chỉ kiểm kê kích thước/ngày/thiếu dữ liệu và tỷ trọng dòng UK toàn nguồn; phân tích định hướng tiền xử lý dùng giai đoạn trước cutoff 2011-09-09. Không dùng hành vi tương lai để chọn bộ lọc hay mô hình.
- Chỉ gắn cờ các mã khác mẫu phổ biến để rà soát; đối chiếu mô tả thật, giữ các mã chữ là hàng hóa. Danh sách loại có lý do và số dòng thực tế.
- Tập bán dương EDA tách hóa đơn C, lượng/giá không dương và mã phi hàng hóa; không xóa chúng khỏi dữ liệu nguồn hoặc áp đặt cho pipeline RFM. Giữ thiếu ID, trùng và ngoại lệ để phân tích, không gán khách giả.
- Biểu đồ signed-log và giới hạn p99 chỉ phục vụ hiển thị. Không cắt đuôi, không fit scaler và không dùng ngưỡng tương lai.
- Khuyến nghị giữ mọi quốc gia, dùng cờ ngoài UK cho profiling; không tự thêm đặc trưng ngoài đề cương.
- Cặp bán–hủy ±74.215 cho thấy doanh thu bán dương không phải Monetary ròng. Người làm phần cleaning/RFM cần quyết định chính sách hoàn/hủy nhất quán cho mọi mô hình.

## Kiểm chứng đã thực hiện

- Dùng môi trường Python riêng `.venv` (bỏ qua bởi Git); file nguồn thực tế tên `Online Retail.xlsx` được sao chép thành `Online_Retail.xlsx` trong thư mục raw đã bỏ qua bởi Git.
- Khởi động kernel mới qua nbclient, xóa trạng thái đầu ra cũ và thực thi tuần tự tất cả ô từ thư mục `notebooks/`, đọc lại Excel bằng `load_raw(use_cache=False)`: **10 ô code thành công, execution_count 1–10, không có error output**, khoảng 43,4 giây. Đây là kiểm tra tương đương Restart & Run All; không tuyên bố đã bấm giao diện Jupyter.
- Kiểm tra schema 8 cột, cutoff nghiêm ngặt, đối soát số dòng trước/sau lọc, miền giá/lượng dương và mã bị loại. Xác thực cấu trúc notebook bằng nbformat và biên dịch module Python.
- Mở và xem trực tiếp cả 5 hình PNG: chữ tiếng Việt đọc được, nhãn không bị cắt, đủ đơn vị và mẫu số. README dùng số liệu xuất từ chính notebook tại `reports/tables/eda_prod_summary.json` và các bảng audit.
- Kết quả chính: 541.909 dòng toàn file; 332.447 dòng quan sát; 323.032 dòng bán dương hàng hóa. UK chiếm 91,44% dòng và 83,92% doanh thu trên tập bán dương quan sát.
- Môi trường đã chạy: Python 3.13.15, pandas 3.0.6, numpy 2.5.4, matplotlib 3.11.2, openpyxl 3.1.5, nbclient 0.11.0; notebook và JSON cũng lưu phiên bản. Không sửa requirements chung vì chưa được giao phần phụ thuộc toàn nhóm.

## Giới hạn và minh bạch

Đây là nội dung và mã do AI hỗ trợ tạo; người nộp cần đọc, hiểu và tự chịu trách nhiệm. Chưa chạy mô hình, chưa tuning, chưa đánh giá RQ3; notebook chỉ nêu liên hệ phương pháp. Các mã chưa rõ và bản chất dòng trùng vẫn cần chuyên môn nghiệp vụ. Không suy ra nhân quả hoặc chất lượng phân cụm từ EDA.
