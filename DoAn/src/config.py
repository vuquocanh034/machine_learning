"""Hằng số dùng chung cho toàn bộ dự án. Mọi module khác import từ đây."""
from pathlib import Path

# ---- Tái lập ----
SEED = 42

# ---- Chia theo thời gian (chống data leakage) ----
# Giai đoạn quan sát: giao dịch TRƯỚC CUTOFF -> tạo đặc trưng, scaler, huấn luyện, chọn tham số.
# Giai đoạn kiểm chứng: từ CUTOFF đến hết dữ liệu -> CHỈ dùng đánh giá ngoài (RQ3).
CUTOFF = "2011-09-09"
REF_DATE = CUTOFF  # Recency tính từ mốc này

# ---- Đường dẫn ----
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "Online_Retail.xlsx"
DATA_INTERIM = ROOT / "data" / "interim"
RAW_CACHE = DATA_INTERIM / "online_retail_raw.pkl"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"

# ---- Bộ đặc trưng ----
RFM = ["Recency", "Frequency", "Monetary"]
RFM_PLUS = RFM + ["AvgOrderValue", "NUniqueProducts", "AvgInterPurchaseDays", "CancelRate"]
FEATURE_SETS = {"RFM": RFM, "RFM_PLUS": RFM_PLUS}

# ---- Schema chung cho mọi bảng tuning ----
TUNING_SCHEMA = [
    "algo", "feature_set", "params", "n_clusters", "noise_frac",
    "silhouette", "davies_bouldin", "calinski_harabasz", "inertia",
]
