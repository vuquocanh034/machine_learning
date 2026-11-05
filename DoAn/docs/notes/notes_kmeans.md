# Ghi chú K-Means (nguyên liệu Chương 2) — Huy, tuần 3A

## 1. Nguyên lý

K-Means chia n điểm thành k cụm bằng cách cực tiểu hóa tổng bình phương khoảng cách Euclid từ mỗi điểm đến **tâm (trung bình)** cụm của nó:

J = Σⱼ Σ_{x∈Cⱼ} ‖x − μⱼ‖²  (đây chính là *inertia*).

Thuật toán Lloyd lặp hai bước đến khi nhãn không đổi:

1. **Gán:** mỗi điểm vào cụm có tâm gần nhất.
2. **Cập nhật:** mỗi tâm = trung bình các điểm của cụm.

Mỗi bước không làm J tăng nên thuật toán hội tụ, nhưng **chỉ về cực tiểu cục bộ**. Vì vậy kết quả phụ thuộc khởi tạo; cách xử lý là chạy `n_init` lần khởi tạo độc lập và giữ lần có J nhỏ nhất. Khởi tạo `k-means++` chọn các tâm đầu xa nhau (xác suất tỷ lệ với bình phương khoảng cách tới tâm đã chọn); `random` chọn k điểm ngẫu nhiên. Độ phức tạp mỗi vòng lặp O(n·k·d).

## 2. Giả định, ưu và nhược điểm

| | Nội dung |
|---|---|
| Giả định | Cụm gần dạng **cầu**, kích thước và mật độ gần nhau; khoảng cách Euclid có nghĩa nên các biến phải cùng thang đo; k biết trước. |
| Ưu | Đơn giản, nhanh, dễ diễn giải (tâm cụm = "khách trung bình" của nhóm); mọi điểm đều được gán cụm nên phù hợp làm phân khúc hành động. |
| Nhược | Phải chọn k; nhạy khởi tạo (cực tiểu cục bộ); **nhạy ngoại lệ** vì tâm là trung bình và mục tiêu là bình phương; không tìm được cụm hình dạng bất kỳ; luôn trả về k cụm kể cả khi dữ liệu là một dải liên tục, nên không tự nói được dữ liệu "có cụm hay không". |

## 3. Lý do chọn cho bài toán này

- Đề bài phân khúc khách hàng cần mọi khách có một nhãn và mỗi nhãn có hồ sơ diễn giải được: K-Means đáp ứng trực tiếp.
- Dữ liệu RFM sau `log1p` + `StandardScaler` có thang đo đồng nhất, đúng điều kiện K-Means cần. Bước `log1p` có lý do cụ thể: Monetary có skew gốc 16,3 và Frequency 9,2, vài khách rất lớn sẽ chi phối khoảng cách Euclid.
- Là phương pháp chuẩn của các nghiên cứu RFM nên làm mốc so sánh với Hierarchical và DBSCAN.
- Hạn chế đã thấy trên dữ liệu thật (xem notebook 03): dữ liệu là dải liên tục (thành phần chính thứ nhất giải thích 73% phương sai của RFM), nên K-Means cắt dải thành các khúc; Silhouette tốt nhất chỉ 0,35–0,41.

## 4. Tham số và quy trình đã dùng

| Tham số | Giá trị | Lý do |
|---|---|---|
| Bộ đặc trưng | `RFM` (3 cột), `RFM_PLUS` (7 cột) | theo proposal; đã `log1p` (cột lệch phải) + `StandardScaler` fit trên giai đoạn quan sát |
| k | 2–10 | theo proposal; k = 1 không phải phân cụm |
| init | `k-means++`, `random` | so sánh ảnh hưởng của khởi tạo |
| n_init | 200 | k = 4 ổn từ 50 (ARI thấp nhất qua 20 seed ≥ 0,99) nhưng k = 10 cần nhiều hơn (ARI thấp nhất của `RFM`: 0,73 → 0,96 khi 50 → 200); chi phí chỉ vài chục giây |
| random_state | `SEED = 42` | tái lập |
| Metric | Silhouette, Davies–Bouldin, Calinski–Harabasz, Inertia | tính trên toàn bộ điểm bằng cùng hàm `baseline.internal_metrics` |

**Chọn cấu hình:** với mỗi (bộ đặc trưng, k) giữ `init` có inertia thấp nhất; top 3 chọn bằng hạng trung bình của ba metric **trong từng bộ đặc trưng** (metric không so được giữa 3 và 7 chiều), chỉ xét k ≥ 3. Ràng buộc k ≥ 3 là phán đoán theo ý nghĩa kinh doanh (k = 2 chỉ tách "còn hoạt động" khỏi "đã im lặng"), không phải theo metric. Giai đoạn kiểm chứng không tham gia bước nào.

## 5. Kết quả chính cần đưa vào báo cáo

- Silhouette và Calinski–Harabasz giảm theo k, cao nhất ở k = 2; Davies–Bouldin của `RFM` gần phẳng ở k = 2–7; khuỷu inertia ở k = 4 (`RFM`) và k = 5 (`RFM_PLUS`). Các chỉ số xung đột, nên k = 4 cho `RFM` là lựa chọn có lập luận (tách khách mua một lần còn gần và đã rời đi), không phải tối ưu theo metric.
- `init` không ảnh hưởng ở k ≤ 7 (ARI giữa hai init ≥ 0,98); ở k lớn có nhiều nghiệm gần tương đương nhưng khác nhau (ARI tới 0,86 ở `RFM`, k = 8).
- `RFM_PLUS` không cải thiện độ tách nội tại (Silhouette 0,27 so với 0,35 ở k = 4) nhưng cho phân hoạch khác (ARI 0,32 với `RFM`). Cụm lớn nhất của nó gồm 96% khách mua một lần (giá trị điền của `AvgInterPurchaseDays`), và cụm hay hủy đơn dựa trên mẫu nhỏ.
- Không có cụm nhỏ (cụm nhỏ nhất 134 khách, 4%) trên toàn lưới. Bỏ 1% khách xa tâm nhất làm tâm dịch tối đa 0,28 sd, ARI còn lại trên 0,9, nhưng 1% điểm này chiếm 8–12% inertia: K-Means bị đuôi chi phối dù đã `log1p`.

## 6. Hạn chế cần nói trong báo cáo

Chưa đo ổn định theo bootstrap hay mốc `CUTOFF` khác; chưa có đánh giá ngoài (RQ3) và sẽ chỉ làm sau khi nhóm chốt mô hình; so sánh Silhouette/DB giữa `RFM` và `RFM_PLUS` chỉ tương đối.
