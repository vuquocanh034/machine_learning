# Lý Thuyết và Cơ Sở Thực Nghiệm: Hierarchical Clustering & DBSCAN

Tài liệu này tổng hợp cơ sở lý thuyết, phân tích ưu/nhược điểm, độ phức tạp tính toán, các tham số chính và lý do áp dụng hai thuật toán **Hierarchical Clustering (Phân cụm phân cấp)** và **DBSCAN (Phân cụm dựa trên mật độ)** cho bài toán phân khúc khách hàng bán lẻ trực tuyến (Online Retail Customer Segmentation). Nội dung phục vụ trực tiếp làm nguyên liệu cho **Chương 2 (Phương pháp nghiên cứu & Mô hình hóa)** của báo cáo đồ án.

---

## 1. Hierarchical Clustering (Agglomerative Hierarchical Clustering)

### 1.1. Nguyên lý hoạt động
Hierarchical Clustering (phân cụm phân cấp) xây dựng một cấu trúc phân cấp dạng cây (gọi là **Dendrogram**) biểu diễn mức độ tương đồng giữa các đối tượng hoặc nhóm đối tượng. Có hai hướng tiếp cận chính:
1. **Agglomerative (Bottom-up - Tích lũy)**: Bắt đầu bằng việc coi mỗi điểm dữ liệu là một cụm riêng lẻ ($N$ cụm ban đầu). Ở mỗi bước, thuật toán tìm cặp cụm có khoảng cách nhỏ nhất theo một tiêu chuẩn liên kết (linkage criterion) xác định và gộp chúng lại với nhau. Quá trình lặp lại $N-1$ lần cho đến khi toàn bộ dữ liệu thuộc về một cụm duy nhất.
2. **Divisive (Top-down - Phân tách)**: Bắt đầu từ 1 cụm chứa toàn bộ điểm dữ liệu, sau đó đệ quy chia tách thành các cụm nhỏ hơn. Cách tiếp cận này ít phổ biến hơn do chi phí tính toán phân tách tổ hợp rất lớn.

Trong đồ án này, chúng ta tập trung vào **Agglomerative Clustering**.

### 1.2. Các tiêu chuẩn liên kết (Linkage Criteria)
Khoảng cách giữa hai cụm $A$ và $B$ được xác định thông qua hàm liên kết:

1. **Ward's Linkage (`ward`)**:
   - Mục tiêu: Giảm thiểu sự gia tăng tổng bình phương sai số trong cụm (within-cluster variance) sau khi gộp:
     $$\Delta \text{ESS}_{AB} = \text{ESS}_{AB} - (\text{ESS}_A + \text{ESS}_B) = \frac{|A| \cdot |B|}{|A| + |B|} \|\mathbf{m}_A - \mathbf{m}_B\|^2$$
     với $\mathbf{m}_A, \mathbf{m}_B$ lần lượt là trọng tâm (mean vector) của cụm $A$ và cụm $B$.
   - Tính chất: Ward tạo ra các cụm có kích thước đồng đều, mật độ tương đối đồng nhất và dạng hình cầu (tương tự như K-Means nhưng theo cấu trúc phân cấp). Ward chỉ áp dụng được với khoảng cách Euclid (`metric='euclidean'`).
2. **Complete Linkage (`complete` - Maximum linkage)**:
   - Khoảng cách giữa 2 cụm là khoảng cách lớn nhất giữa một cặp điểm bất kỳ thuộc về 2 cụm:
     $$D(A, B) = \max_{x \in A, y \in B} d(x, y)$$
   - Tính chất: Nhạy cảm với đường kính tối đa của cụm, xu hướng tạo ra các cụm nhỏ gọn (compact), tránh hiện tượng kéo chuỗi dài nhưng dễ bị ảnh hưởng bởi điểm ngoại lệ ở rìa cụm.
3. **Average Linkage (`average` - UPGMA)**:
   - Khoảng cách giữa 2 cụm là trung bình cộng khoảng cách giữa tất cả các cặp điểm:
     $$D(A, B) = \frac{1}{|A| \cdot |B|} \sum_{x \in A} \sum_{y \in B} d(x, y)$$
   - Tính chất: Là sự dung hòa giữa Single Linkage và Complete Linkage, ít nhạy cảm với nhiễu hơn Complete Linkage. Tuy nhiên trên dữ liệu thực tế có phân phối mật độ không đồng đều, `average` rất dễ bị hiện tượng "chuỗi" (chaining) - liên tục gộp các điểm cô lập vào một cụm khổng lồ duy nhất.
4. **Single Linkage (`single` - Minimum linkage)**:
   - Khoảng cách nhỏ nhất giữa một cặp điểm: $D(A, B) = \min_{x \in A, y \in B} d(x, y)$. Thường gây ra hiện tượng chuỗi nặng nề nên ít được ưu tiên cho bài toán phân khúc thị trường.

### 1.3. Biểu đồ Dendrogram và cách chọn số cụm
- **Dendrogram** hiển thị lịch sử các phép gộp. Trục hoành biểu diễn các quan sát (hoặc chỉ số mẫu), trục tung biểu diễn khoảng cách (dissimilarity/linkage distance) tại thời điểm hai cụm được sáp nhập.
- **Quy tắc chọn số cụm ($k$)**: Tìm nhát cắt ngang (horizontal cut) đi qua các nhánh thẳng đứng có chiều dài lớn nhất (vertical distance) mà không bị các nhánh ngang khác cắt qua. Khoảng cách thẳng đứng lớn biểu thị rằng việc gộp hai cụm ở mức đó làm tăng đột biến độ dị biệt nội cụm, tức là cấu trúc $k$ cụm trước khi gộp là tự nhiên và tách bạch nhất.

### 1.4. Ưu - Nhược điểm & Độ phức tạp
- **Ưu điểm**:
  - Không yêu cầu chỉ định trước số cụm $k$ khi xây dựng cây phân cấp (có thể tùy chọn $k$ linh hoạt sau khi quan sát dendrogram).
  - Cung cấp cái nhìn phả hệ sâu sắc: cho phép doanh nghiệp phân khúc đa cấp độ (ví dụ: cấp vĩ mô $k=3$: Khách VIP / Phổ thông / Rời bỏ; cấp vi mô $k=6$: tách nhỏ nhóm Phổ thông thành Khách mới, Khách tiềm năng, Khách có nguy cơ).
  - Kết quả hoàn toàn xác định (deterministic), không phụ thuộc vào khởi tạo ngẫu nhiên như K-Means.
- **Nhược điểm**:
  - Khi hai cụm đã được gộp lại ở một bước, quyết định đó không thể đảo ngược (không có cơ chế tối ưu hóa cục bộ gán lại điểm như K-Means).
  - Rất tốn tài nguyên bộ nhớ và thời gian tính toán khi kích thước mẫu $N$ lớn.
- **Độ phức tạp tính toán**:
  - Thời gian: $O(N^2 \log N)$ đến $O(N^3)$ tùy cấu trúc dữ liệu lưu trữ ma trận khoảng cách.
  - Không gian (bộ nhớ): $O(N^2)$ để lưu ma trận khoảng cách đôi một giữa các điểm. Với $N = 3.360$ khách hàng trong giai đoạn quan sát của đồ án, ma trận khoảng cách cần $\approx 3360 \times 3360 \times 8 \text{ bytes} \approx 90\text{ MB}$, hoàn toàn phù hợp để tính toán trực tiếp trên RAM.

---

## 2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### 2.1. Nguyên lý hoạt động
DBSCAN là thuật toán phân cụm dựa trên mật độ không gian. Ý tưởng cốt lõi: một cụm là một vùng có mật độ điểm dữ liệu cao, được ngăn cách bởi các vùng có mật độ điểm dữ liệu thấp (được coi là nhiễu hoặc điểm ngoại lệ).

Thuật toán định nghĩa các khái niệm cơ bản dựa trên hai siêu tham số $\varepsilon$ (eps) và $MinPts$ (min_samples):
1. **$\varepsilon$-lân cận ($N_\varepsilon(p)$)**: Tập hợp các điểm $q$ trong không gian có khoảng cách tới $p$ không vượt quá $\varepsilon$:
   $$N_\varepsilon(p) = \{q \in D \mid d(p, q) \le \varepsilon\}$$
2. **Điểm lõi (Core point)**: Điểm $p$ có ít nhất $MinPts$ điểm nằm trong $\varepsilon$-lân cận của nó ($|N_\varepsilon(p)| \ge MinPts$). Điểm lõi nằm sâu bên trong vùng mật độ cao.
3. **Trực tiếp mật độ đạt được (Directly Density-Reachable)**: Điểm $q$ trực tiếp mật độ đạt được từ $p$ nếu $p$ là điểm lõi và $q \in N_\varepsilon(p)$.
4. **Mật độ đạt được (Density-Reachable)**: Điểm $q$ mật độ đạt được từ $p$ nếu tồn tại một chuỗi các điểm $p_1, p_2, \dots, p_n$ với $p_1 = p, p_n = q$ sao cho $p_{i+1}$ trực tiếp mật độ đạt được từ $p_i$. (Quan hệ này không có tính đối xứng nếu điểm đích là điểm biên).
5. **Liên thông mật độ (Density-Connected)**: Hai điểm $p$ và $q$ được gọi là liên thông mật độ nếu tồn tại điểm $o$ sao cho cả $p$ và $q$ đều mật độ đạt được từ $o$. Đây là quan hệ tương đương (đối xứng và bắc cầu giữa các điểm lõi).
6. **Điểm biên (Border point)**: Điểm không phải là điểm lõi ($|N_\varepsilon(p)| < MinPts$), nhưng nằm trong $\varepsilon$-lân cận của một điểm lõi. Điểm biên đóng vai trò làm ranh giới ngoài của cụm.
7. **Điểm nhiễu (Noise point)**: Điểm không phải điểm lõi và cũng không thuộc về bất kỳ $\varepsilon$-lân cận của điểm lõi nào. Điểm nhiễu được gán nhãn $-1$.

### 2.2. Các tham số chính và phương pháp xác định
- **Bán kính $\varepsilon$ (eps)**: Xác định phạm vi lân cận. Nếu $\varepsilon$ quá nhỏ, phần lớn dữ liệu bị coi là nhiễu; nếu $\varepsilon$ quá lớn, các cụm riêng biệt sẽ bị sáp nhập thành một cụm khổng lồ duy nhất.
- **Số điểm tối thiểu $MinPts$ (min_samples)**: Xác định ngưỡng mật độ tối thiểu để tạo thành cụm. Quy tắc kinh nghiệm (heuristic):
  - Thông thường chọn $MinPts \ge D + 1$ (với $D$ là số chiều không gian đặc trưng), hoặc $MinPts \approx 2 \times D$ để chống lại hiện tượng thưa thớt do số chiều cao (curse of dimensionality).
  - Với tập `RFM` ($D=3$), $MinPts \in [4, 10]$.
  - Với tập `RFM_PLUS` ($D=7$), $MinPts \in [8, 14]$.
- **Đồ thị $k$-distance plot (với $k = MinPts$)**:
  - Tính khoảng cách từ mỗi điểm dữ liệu tới láng giềng thứ $k$ gần nhất của nó.
  - Sắp xếp các khoảng cách này theo thứ tự tăng dần và vẽ đồ thị.
  - **Điểm gãy (Elbow/Knee point)**: Điểm mà đồ thị bắt đầu uốn cong vọt lên trên. Khoảng cách tại điểm gãy phản ánh ranh giới giữa các điểm thuộc vùng mật độ dày đặc (phần đồ thị nằm ngang thấp) và các điểm thưa thớt/ngoại lệ (phần đồ thị dựng đứng). Giá trị khoảng cách tại điểm gãy là ước lượng ban đầu tối ưu cho $\varepsilon$.

### 2.3. Hiện tượng cấu hình suy biến trong DBSCAN
Trên dữ liệu thực tế (đặc biệt là dữ liệu hành vi khách hàng bán lẻ), DBSCAN thường gặp các cấu hình suy biến:
1. **Suy biến 1 cụm khổng lồ (Over-merging)**: Khi $\varepsilon$ quá lớn, thuật toán kết nối tất cả các điểm thành 1 cụm duy nhất (chiếm > 95-99% dữ liệu). Cấu hình này không có giá trị phân khúc.
2. **Suy biến toàn bộ là nhiễu (Over-fragmentation / High Noise)**: Khi $\varepsilon$ quá nhỏ hoặc $MinPts$ quá lớn, thuật toán không tìm thấy đủ điểm lõi, tỷ lệ nhiễu vượt quá 80-90% hoặc phân mảnh thành hàng chục vi-cụm nhỏ.
- **Nguyên nhân cốt lõi**: Phân phối mật độ không đồng đều (non-uniform density). Vùng khách hàng phổ thông (Recency trung bình, Frequency thấp, Monetary thấp) có mật độ điểm cực kỳ dày đặc; trong khi vùng khách hàng giá trị cao (High Monetary, High Frequency) lại rất thưa thớt. Một giá trị $\varepsilon$ toàn cục duy nhất không thể vừa bao trọn vùng thưa thớt vừa phân tách được vùng dày đặc.

### 2.4. Đánh giá nội tại cho DBSCAN
- Vì nhãn $-1$ đại diện cho các quan sát ngoại lệ cô lập chứ không phải một phân khúc đồng nhất, các độ đo nội tại chuẩn như **Silhouette Score**, **Davies-Bouldin Index**, **Calinski-Harabasz Index** **chỉ được tính trên tập các điểm hợp lệ (nhãn $\neq -1$)**.
- Cần theo dõi đồng thời $n\_clusters$ và tỷ lệ nhiễu $noise\_frac$ để tránh chọn cấu hình có Silhouette ảo cao do loại bỏ hầu hết dữ liệu dưới dạng nhiễu.

### 2.5. Ưu - Nhược điểm & Độ phức tạp
- **Ưu điểm**:
  - Tự động xác định số lượng cụm dựa trên mật độ tự nhiên, không cần ấn định $k$.
  - Có khả năng phát hiện các cụm có hình dạng hình học phức tạp, phi cầu (arbitrary shapes).
  - Có cơ chế cô lập điểm ngoại lệ/nhiễu tự nhiên và mạnh mẽ, không bị các điểm cực đoan kéo lệch phân cụm như K-Means hay Ward Hierarchical.
- **Nhược điểm**:
  - Không hoạt động hiệu quả khi các cụm trong dữ liệu có mật độ chênh lệch quá lớn (variable density).
  - Rất nhạy cảm với việc chọn cặp tham số $(\varepsilon, MinPts)$.
  - Bị ảnh hưởng nặng nề bởi "lời nguyền số chiều" (Curse of Dimensionality): trong không gian nhiều chiều, khoảng cách giữa các điểm có xu hướng hội tụ, khiến khái niệm mật độ trở nên kém phân biệt.
- **Độ phức tạp tính toán**:
  - Thời gian: $O(N \log N)$ nếu sử dụng cấu trúc chỉ mục không gian như $k$-d tree hoặc ball tree; $O(N^2)$ trong trường hợp xấu nhất hoặc số chiều cao.
  - Không gian: $O(N)$.

---

## 3. So Sánh Bản Chất Gom Cụm Giữa Hierarchical, DBSCAN và K-Means

| Tiêu chí | K-Means | Hierarchical (Ward Linkage) | DBSCAN |
| :--- | :--- | :--- | :--- |
| **Bản chất thuật toán** | Phân hoạch (Partitioning) dựa trên trọng tâm (Centroid-based). | Phân cấp (Hierarchical) dựa trên khoảng cách liên kết (Linkage-based). | Mật độ (Density-based) dựa trên tính liên thông mật độ. |
| **Giả định hình học cụm** | Cụm dạng hình cầu lồi (convex spherical), kích thước và phương sai tương đối đồng đều. | Tương tự K-Means (với Ward), tạo cụm gọn, cân bằng; với Complete/Average tạo cụm theo chuỗi khoảng cách. | Bất kỳ hình dạng nào (arbitrary / non-convex shapes), không giả định hình cầu. |
| **Yêu cầu số cụm $k$** | Phải ấn định trước số cụm $k$. | Cắt cây dendrogram tại bất kỳ mức $k$ nào sau khi huấn luyện. | Tự động phát hiện số cụm dựa trên $\varepsilon$ và $MinPts$. |
| **Xử lý điểm ngoại lệ (Outliers / Noise)** | **Rất nhạy cảm**: Ép mọi điểm ngoại lệ vào cụm gần nhất, làm méo mó tâm cụm. | **Nhạy cảm**: Các điểm ngoại lệ thường bị gộp muộn nhất hoặc làm sai lệch liên kết. | **Tuyệt vời**: Chủ động cô lập ngoại lệ thành nhóm nhiễu (nhãn $-1$). |
| **Tính tất định (Determinism)** | Không tất định (phụ thuộc vào khởi tạo ngẫu nhiên `init='k-means++'`, hạt giống ngẫu nhiên). | Hoàn toàn tất định (cho cùng một ma trận khoảng cách và linkage). | Hoàn toàn tất định đối với các điểm lõi và nhiễu (điểm biên có thể phụ thuộc nhẹ vào thứ tự duyệt). |
| **Độ phức tạp thời gian** | $O(t \cdot k \cdot N \cdot D)$ (rất nhanh, xấp xỉ tuyến tính với $N$). | $O(N^2 \log N)$ đến $O(N^3)$ (chậm khi $N$ lớn). | $O(N \log N)$ với cây không gian, $O(N^2)$ số chiều cao. |
| **Ý nghĩa trong phân khúc khách hàng** | Phù hợp nhất để chia thị trường thành các nhóm quy mô cân đối, dễ đặt tên kinh doanh. | Cho phép nhìn thấy cấu trúc quan hệ phả hệ đa cấp độ của khách hàng. | Cực kỳ xuất sắc để sàng lọc khách hàng mua sỉ / giao dịch bất thường (nhóm ngoại lệ). |

---

## 4. Lý Do Áp Dụng Cho Bài Toán Phân Khúc Khách Hàng Bán Lẻ

1. **Bản chất phân phối dữ liệu bán lẻ (Skewed & Heavy-tailed)**:
   - Dữ liệu giao dịch bán lẻ thực tế luôn tồn tại quy luật Pareto (20% khách hàng đóng góp 80% doanh thu). Một số lượng nhỏ khách hàng mua sỉ (Wholesale) có Monetary và Frequency gấp hàng chục tới hàng trăm lần mức trung bình của tập khách lẻ.
   - Nếu chỉ áp dụng K-Means thuần túy, các khách hàng cực đoan này sẽ kéo lệch trọng tâm của phân khúc VIP, khiến phân khúc này trở nên quá rộng hoặc chia nhỏ sai lệch các phân khúc thông thường.
2. **Vai trò bổ trợ của DBSCAN**:
   - DBSCAN không cố gắng ép buộc mọi khách hàng phải thuộc về một cụm thông thường. Thuật toán tự động nhận diện và cô lập các khách hàng ngoại lệ vào nhóm nhiễu (`noise`, nhãn `-1`).
   - Kết quả thực nghiệm cho thấy nhóm nhiễu của DBSCAN có giá trị Monetary trung bình $> 15.000$ GBP và AvgOrderValue $> 2.300$ GBP (so với mức trung bình chung của khách hàng thường là $\approx 1.100$ GBP và $330$ GBP). Điều này chứng minh DBSCAN là công cụ định vị khách sỉ / khách siêu giá trị (Wholesale B2B) vô cùng chính xác.
3. **Vai trò bổ trợ của Hierarchical Clustering**:
   - Biểu đồ Dendrogram cung cấp bằng chứng trực quan và khoa học để thẩm định tính hợp lý của số lượng cụm $k$, bổ sung cho phương pháp Elbow và Silhouette của K-Means.
   - Cung cấp mô hình phân cấp khách hàng từ gốc đến ngọn: giúp nhà quản trị kinh doanh linh hoạt áp dụng chiến lược từ phân khúc tổng thể 3 nhóm lớn (Active High, Active Low, Inactive) đến chiến lược tiếp thị chi tiết 4-6 nhóm nhỏ.

