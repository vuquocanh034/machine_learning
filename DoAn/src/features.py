"""Chia theo thời gian, tạo đặc trưng RFM / RFM_PLUS và chuẩn hóa (chỉ dùng giai đoạn quan sát).

Kiểm soát data leakage:
  * Mọi đặc trưng, ngưỡng lệch phải, scaler đều ước lượng CHỈ từ giao dịch trước CUTOFF (giai đoạn quan sát).
  * Giao dịch từ CUTOFF trở đi chỉ dùng để tạo nhãn kiểm chứng (FutureSpend, Returned) cho đánh giá ngoài RQ3.
    Hai cột này không bao giờ là đặc trưng và không dùng để chọn tham số/mô hình.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import CUTOFF, DATA_PROCESSED, MODELS_DIR, REF_DATE, RFM, RFM_PLUS

# Biến chỉ dùng để mô tả/profiling, không đưa vào ma trận đặc trưng.
AUX_COLS = ["SinglePurchase"]
# Ngưỡng |skewness| để coi một biến là lệch phải đáng kể (quy ước thường dùng: > 1).
SKEW_THRESHOLD = 1.0


# ---------------------------------------------------------------- chia theo thời gian
def split_by_time(sales: pd.DataFrame, cancel: pd.DataFrame, cutoff: str = CUTOFF):
    """Tách giao dịch bán/hủy thành giai đoạn quan sát (< cutoff) và kiểm chứng (>= cutoff).

    Trả về (sales_obs, cancel_obs, sales_future). Hóa đơn hủy giai đoạn kiểm chứng không dùng ở đâu cả.
    """
    cut = pd.Timestamp(cutoff)
    sales_obs = sales[sales["InvoiceDate"] < cut].copy()
    cancel_obs = cancel[cancel["InvoiceDate"] < cut].copy()
    sales_future = sales[sales["InvoiceDate"] >= cut].copy()
    # load_raw đọc CustomerID kiểu Int64 (nullable); sau clean_transactions không còn thiếu nên ép về int64
    # để index của mọi bảng đặc trưng/nhãn cùng kiểu, so sánh và nối bảng không lệch dtype.
    for d in (sales_obs, cancel_obs, sales_future):
        assert d["CustomerID"].notna().all(), "CustomerID còn giá trị thiếu: chạy clean_transactions trước"
        d["CustomerID"] = d["CustomerID"].astype("int64")
    return sales_obs, cancel_obs, sales_future


def future_outcome(sales_future: pd.DataFrame, customer_ids) -> pd.DataFrame:
    """Nhãn kiểm chứng cho các khách trong tập phân khúc (chỉ để đánh giá ngoài RQ3).

    FutureSpend = tổng Quantity*UnitPrice các dòng bán trong giai đoạn kiểm chứng (0 nếu không mua).
    Returned    = 1 nếu có ít nhất một hóa đơn bán trong giai đoạn kiểm chứng, ngược lại 0.
    Khách mới chỉ xuất hiện ở giai đoạn kiểm chứng không nằm trong tập phân khúc (không có đặc trưng).
    """
    spend = sales_future.groupby("CustomerID")["LineTotal"].sum()
    out = pd.DataFrame(index=pd.Index(customer_ids, name="CustomerID"))
    out["FutureSpend"] = spend.reindex(out.index).fillna(0.0)
    # Returned tính theo sự tồn tại của hóa đơn bán, không theo dấu của tổng tiền.
    n_inv = sales_future.groupby("CustomerID")["InvoiceNo"].nunique().reindex(out.index).fillna(0)
    out["Returned"] = (n_inv > 0).astype(int)
    return out


# ---------------------------------------------------------------- đặc trưng theo khách
def build_features(sales_obs: pd.DataFrame, cancel_obs: pd.DataFrame, ref_date: str = REF_DATE) -> pd.DataFrame:
    """Đặc trưng theo khách, chỉ từ giao dịch giai đoạn quan sát. Index = CustomerID.

    Recency              số ngày từ ngày mua cuối đến ref_date (tính theo ngày lịch).
    Frequency            số hóa đơn bán khác nhau.
    Monetary             tổng Quantity*UnitPrice của giao dịch bán.
    AvgOrderValue        Monetary / Frequency.
    NUniqueProducts      số StockCode khác nhau đã mua.
    AvgInterPurchaseDays khoảng cách trung bình (ngày) giữa hai hóa đơn liên tiếp = (hóa đơn cuối - đầu) / (F-1).
                         Khách chỉ có 1 hóa đơn thì không quan sát được khoảng cách: điền bằng độ dài giai đoạn
                         quan sát (từ giao dịch đầu tiên đến ref_date). Lý do: khoảng cách thật chỉ có thể
                         dài hơn khoảng thời gian họ đã "im lặng" (dữ liệu bị cắt phải), nên điền giá trị lớn
                         nhất có thể giữ đúng chiều "mua thưa"; điền trung vị sẽ xếp họ vào giữa, sai chiều.
                         Cột cờ SinglePurchase đánh dấu các khách này để profiling và kiểm tra độ nhạy.
    CancelRate           số hóa đơn hủy / (hóa đơn bán + hóa đơn hủy), trong [0, 1]; khách không hủy = 0.
    """
    ref = pd.Timestamp(ref_date)
    assert sales_obs["InvoiceDate"].max() < ref, "sales_obs chứa giao dịch từ ref_date trở đi (leakage)"
    assert cancel_obs.empty or cancel_obs["InvoiceDate"].max() < ref, "cancel_obs chứa giao dịch từ ref_date trở đi (leakage)"

    g = sales_obs.groupby("CustomerID")
    feats = pd.DataFrame(index=g.size().index)
    feats.index.name = "CustomerID"

    last_day = g["InvoiceDate"].max().dt.normalize()
    feats["Recency"] = (ref.normalize() - last_day).dt.days
    feats["Frequency"] = g["InvoiceNo"].nunique()
    feats["Monetary"] = g["LineTotal"].sum()
    feats["AvgOrderValue"] = feats["Monetary"] / feats["Frequency"]
    feats["NUniqueProducts"] = g["StockCode"].nunique()

    # Thời điểm từng hóa đơn (mỗi hóa đơn lấy thời điểm sớm nhất của các dòng).
    inv = sales_obs.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"].min().reset_index()
    gi = inv.groupby("CustomerID")["InvoiceDate"]
    span_days = (gi.max() - gi.min()).dt.total_seconds() / 86400
    window_days = float((ref - sales_obs["InvoiceDate"].min().normalize()).days)
    single = feats["Frequency"] == 1
    gap = span_days.reindex(feats.index) / (feats["Frequency"] - 1).where(~single)
    feats["AvgInterPurchaseDays"] = gap.where(~single, window_days)
    feats["SinglePurchase"] = single.astype(int)

    n_cancel = cancel_obs.groupby("CustomerID")["InvoiceNo"].nunique().reindex(feats.index).fillna(0)
    feats["CancelRate"] = n_cancel / (feats["Frequency"] + n_cancel)

    return feats[RFM_PLUS + AUX_COLS]


# ---------------------------------------------------------------- biến đổi + chuẩn hóa
def skewness_report(df: pd.DataFrame, cols, threshold: float = SKEW_THRESHOLD) -> pd.DataFrame:
    """Skewness trước/sau log1p và quyết định biến đổi cho từng cột.

    Quy tắc: chỉ log1p khi skew gốc > threshold (lệch phải rõ) VÀ log1p làm |skew| giảm ít nhất một nửa.
    Điều kiện thứ hai tránh áp dụng máy móc cho biến dồn nhiều số 0 trên thang nhỏ (như CancelRate trong [0, 1]),
    nơi log1p gần như không đổi dạng phân phối. log1p an toàn vì mọi đặc trưng >= 0 (xác nhận bằng assert).
    """
    rows = []
    for c in cols:
        assert (df[c] >= 0).all(), f"{c} có giá trị âm, không dùng được log1p"
        before = float(df[c].skew())
        after = float(np.log1p(df[c]).skew())
        rows.append({
            "feature": c, "skew_before": before, "skew_log1p": after,
            "log1p": bool(before > threshold and abs(after) <= 0.5 * abs(before)),
        })
    return pd.DataFrame(rows).set_index("feature")


def scaler_path(cols):
    """Đường dẫn file scaler theo bộ đặc trưng (RFM hoặc RFM_PLUS)."""
    name = "rfm" if list(cols) == RFM else "rfm_plus"
    return MODELS_DIR / f"scaler_{name}.joblib"


def transform_features(df: pd.DataFrame, cols, fit_scaler: bool = True, path=None) -> pd.DataFrame:
    """log1p cho biến lệch phải rồi StandardScaler. Trả về DataFrame đã scale (giữ index, tên cột).

    fit_scaler=True : quyết định biến nào cần log1p và fit scaler TRÊN df (phải là đặc trưng giai đoạn quan sát),
                      lưu {scaler, log_cols, cols} vào models/scaler_*.joblib.
    fit_scaler=False: nạp bundle đã lưu và chỉ transform (dùng cho dữ liệu mới), không fit lại.
    Cần log + scale vì K-Means/Hierarchical/DBSCAN dựa trên khoảng cách Euclid: biến lệch phải (Monetary, Frequency)
    có vài khách khổng lồ sẽ kéo khoảng cách, còn biến có đơn vị lớn (tiền, ngày) lấn át biến nhỏ (CancelRate).
    """
    cols = list(cols)
    path = path or scaler_path(cols)
    if fit_scaler:
        report = skewness_report(df, cols)
        log_cols = report.index[report["log1p"]].tolist()
        x = df[cols].astype(float).copy()
        x[log_cols] = np.log1p(x[log_cols])
        scaler = StandardScaler().fit(x)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": scaler, "log_cols": log_cols, "cols": cols}, path)
    else:
        bundle = joblib.load(path)
        scaler, log_cols = bundle["scaler"], bundle["log_cols"]
        assert bundle["cols"] == cols, "bộ cột không khớp với scaler đã lưu"
        x = df[cols].astype(float).copy()
        x[log_cols] = np.log1p(x[log_cols])
    return pd.DataFrame(scaler.transform(x), index=df.index, columns=cols)


def load_scaler_bundle(cols) -> dict:
    """Nạp {scaler, log_cols, cols} đã lưu cho bộ đặc trưng."""
    return joblib.load(scaler_path(cols))


# ---------------------------------------------------------------- tương quan / dư thừa
def correlation_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Các cặp đặc trưng có |tương quan| >= threshold (mỗi cặp một dòng, sắp giảm dần)."""
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = float(corr.loc[a, b])
            rows.append({"feature_a": a, "feature_b": b, "pearson_r": r, "abs_r": abs(r)})
    out = pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
    out["flag"] = out["abs_r"] >= threshold
    return out


def explained_by(x: pd.DataFrame, target: str, others) -> float:
    """R^2 khi hồi quy tuyến tính `target` theo các cột `others` (đo mức một đặc trưng bị các đặc trưng khác tái tạo)."""
    y = x[target].to_numpy(float)
    a = np.column_stack([x[list(others)].to_numpy(float), np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    ss_res = float(((y - a @ coef) ** 2).sum())
    return 1.0 - ss_res / float(((y - y.mean()) ** 2).sum())


def vif_table(x: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor của từng đặc trưng (đường chéo của nghịch đảo ma trận tương quan).

    VIF > 5 (nghiêm ngặt) hoặc > 10 (nới) cho thấy đặc trưng gần như là tổ hợp tuyến tính của các đặc trưng khác.
    """
    inv = np.linalg.inv(x.corr().to_numpy())
    return pd.Series(np.diag(inv), index=x.columns, name="VIF")


# ---------------------------------------------------------------- kiểm tra leakage
def assert_no_leakage(sales: pd.DataFrame, cancel: pd.DataFrame, cutoff: str = CUTOFF, ref_date: str = REF_DATE):
    """Kiểm tra bằng thực nghiệm: phá hỏng dữ liệu kiểm chứng thì đặc trưng/scaler không được đổi.

    Nhân đôi số lượng, đổi mã khách và xóa dữ liệu từ cutoff trở đi trong bản sao, rồi dựng lại đặc trưng.
    Nếu có bất kỳ đường nào rò thông tin tương lai vào đặc trưng, hai kết quả sẽ khác nhau.
    """
    cut = pd.Timestamp(cutoff)
    s0, c0, _ = split_by_time(sales, cancel, cutoff)
    f0 = build_features(s0, c0, ref_date)

    s2, c2 = sales.copy(), cancel.copy()
    fut_s, fut_c = s2["InvoiceDate"] >= cut, c2["InvoiceDate"] >= cut
    s2.loc[fut_s, "Quantity"] = s2.loc[fut_s, "Quantity"] * 7
    s2.loc[fut_s, "LineTotal"] = s2.loc[fut_s, "LineTotal"] * 7
    c2 = c2.loc[~fut_c]
    s3, c3, _ = split_by_time(s2, c2, cutoff)
    f3 = build_features(s3, c3, ref_date)
    pd.testing.assert_frame_equal(f0, f3)

    x0 = transform_features(f0, RFM_PLUS, fit_scaler=False)
    x3 = transform_features(f3, RFM_PLUS, fit_scaler=False)
    pd.testing.assert_frame_equal(x0, x3)
    return True


def save_features(feats: pd.DataFrame, outcome: pd.DataFrame, dir_=None):
    """Lưu đặc trưng quan sát và nhãn kiểm chứng ra data/processed/."""
    d = dir_ or DATA_PROCESSED
    d.mkdir(parents=True, exist_ok=True)
    feats.to_csv(d / "features_obs.csv")
    outcome.to_csv(d / "future_outcome.csv")
    return d / "features_obs.csv", d / "future_outcome.csv"
