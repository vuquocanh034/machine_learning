"""Baseline phân khúc theo luật RFM (điểm 1–5 theo phân vị) và đánh giá bằng cùng metric nội tại với mô hình học máy.

Baseline chỉ dùng đặc trưng giai đoạn quan sát; không nhìn nhãn kiểm chứng.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.config import FEATURE_SETS, TUNING_SCHEMA
from src.features import transform_features

# Luật ghép phân khúc, áp dụng THEO THỨ TỰ (dòng đầu thỏa thì lấy). FM = (F + M) / 2.
# Dùng R và FM thay vì cả ba chiều để mỗi phân khúc có nghĩa kinh doanh rõ: R = còn hoạt động không,
# FM = giá trị/độ trung thành. Bộ luật phủ kín mọi tổ hợp (R, FM) nên không có khách nào bị bỏ sót.
SEGMENT_RULES = [
    ("Champions", "R >= 4 và FM >= 4"),
    ("Loyal", "R >= 3 và FM >= 3 (phần còn lại sau Champions)"),
    ("Potential Loyalist", "R >= 4 và 2 <= FM < 3"),
    ("New", "R >= 4 và FM < 2"),
    ("Need Attention", "R == 3 và FM < 3"),
    ("At Risk", "R <= 2 và FM >= 3"),
    ("Hibernating", "R <= 2 và 2 <= FM < 3"),
    ("Lost", "R <= 2 và FM < 2"),
]
SEGMENT_ORDER = [name for name, _ in SEGMENT_RULES]


def quintile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Điểm 1–5 theo phân vị, các giá trị trùng nhau luôn nhận CÙNG điểm.

    Dùng hạng "min" (xếp hạng kiểu thi đấu: các giá trị trùng cùng nhận hạng thấp nhất của nhóm):
    pct_thấp = (hạng_min - 1) / n, điểm = floor(5 * pct_thấp) + 1.
    Không dùng pd.qcut trực tiếp vì Frequency có rất nhiều khách trùng giá trị (40% chỉ có 1 hóa đơn):
    qcut báo lỗi cạnh trùng hoặc, nếu rank(method='first'), chia đôi nhóm giống hệt nhau một cách tùy tiện theo thứ tự
    CustomerID. Hạng "average" cũng không ổn: nhóm thấp nhất 40% có hạng trung bình 0,20 nên rơi sang điểm 2 và
    không ai có điểm 1. Hạng "min" luôn cho nhóm thấp nhất điểm 1; đổi lại các mức điểm có thể không đều nhau.
    Recency thì nhỏ hơn là tốt hơn (higher_is_better=False): đảo dấu để khách mua gần đây nhất được 5 điểm.
    """
    v = values if higher_is_better else -values
    pct_low = (v.rank(method="min") - 1) / len(v)
    return (np.floor(5 * pct_low).astype(int) + 1).clip(1, 5)


def rfm_rule_segments(features: pd.DataFrame) -> pd.DataFrame:
    """Chấm điểm R, F, M (1–5) theo phân vị rồi ghép thành phân khúc có tên theo SEGMENT_RULES.

    Trả về DataFrame index CustomerID với R, F, M, Segment, Baseline_Label (số nguyên theo SEGMENT_ORDER).
    """
    s = pd.DataFrame(index=features.index)
    s["R"] = quintile_score(features["Recency"], higher_is_better=False)
    s["F"] = quintile_score(features["Frequency"])
    s["M"] = quintile_score(features["Monetary"])
    fm = (s["F"] + s["M"]) / 2
    r = s["R"]

    conds = [
        (r >= 4) & (fm >= 4),
        (r >= 3) & (fm >= 3),
        (r >= 4) & (fm >= 2) & (fm < 3),
        (r >= 4) & (fm < 2),
        (r == 3) & (fm < 3),
        (r <= 2) & (fm >= 3),
        (r <= 2) & (fm >= 2) & (fm < 3),
        (r <= 2) & (fm < 2),
    ]
    s["Segment"] = np.select(conds, SEGMENT_ORDER, default="UNASSIGNED")
    assert (s["Segment"] != "UNASSIGNED").all(), "luật ghép không phủ hết các tổ hợp (R, FM)"
    s["Baseline_Label"] = s["Segment"].map({n: i for i, n in enumerate(SEGMENT_ORDER)}).astype(int)
    return s


def internal_metrics(x: pd.DataFrame, labels) -> dict:
    """Silhouette, Davies–Bouldin, Calinski–Harabasz cho một phân hoạch (cùng hàm dùng cho mọi mô hình).

    Tính trên toàn bộ điểm (không lấy mẫu) nên kết quả xác định. Cần ít nhất 2 cụm.
    """
    lab = np.asarray(labels)
    if len(np.unique(lab)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": float(silhouette_score(x, lab)),
        "davies_bouldin": float(davies_bouldin_score(x, lab)),
        "calinski_harabasz": float(calinski_harabasz_score(x, lab)),
    }


def evaluate_baseline(features: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Đánh giá baseline trên không gian đặc trưng đã scale của từng feature set (scaler đã fit trên giai đoạn quan sát).

    Dùng đúng schema TUNING_SCHEMA để tuần sau gộp được với bảng tuning của mô hình. Inertia chỉ định nghĩa cho
    K-Means nên để NaN. noise_frac = 0 vì luật gán mọi khách vào một phân khúc.
    """
    labels = segments.loc[features.index, "Baseline_Label"]
    rows = []
    for name, cols in FEATURE_SETS.items():
        x = transform_features(features, cols, fit_scaler=False)
        m = internal_metrics(x, labels)
        rows.append({
            "algo": "RFM_rule", "feature_set": name,
            "params": json.dumps({"scoring": "quintile_rank_min_ties", "segments": len(SEGMENT_ORDER)}),
            "n_clusters": int(labels.nunique()), "noise_frac": 0.0, "inertia": np.nan, **m,
        })
    return pd.DataFrame(rows)[TUNING_SCHEMA]
