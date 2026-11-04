"""Mô đun phân cụm (Hierarchical Agglomerative Clustering & DBSCAN).

Tuân thủ quy ước:
  - Chỉ nhận ma trận đặc trưng đã scale từ giai đoạn quan sát (không nhìn nhãn kiểm chứng).
  - Không sửa/xóa các hàm đã có (để dành chỗ cho run_kmeans của Huy).
  - Trả về cấu trúc kết quả và metric nội tại theo TUNING_SCHEMA chuẩn của dự án.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from src.config import TUNING_SCHEMA


# ---------------------------------------------------------------- Đánh giá nội tại
def evaluate_clustering(x: pd.DataFrame | np.ndarray, labels: Sequence[int] | np.ndarray) -> dict[str, Any]:
    """Tính toán các chỉ số nội tại cho một kết quả phân cụm.

    Quy ước đặc biệt với DBSCAN / thuật toán có nhiễu:
      - Cụm nhiễu mang nhãn -1.
      - Tỷ lệ nhiễu: noise_frac = (số điểm có nhãn -1) / (tổng số điểm).
      - Các chỉ số nội tại (Silhouette, Davies-Bouldin, Calinski-Harabasz) CHỈ ĐƯỢC TÍNH
        trên tập các điểm hợp lệ không phải nhiễu (labels != -1).
      - Nếu số cụm hợp lệ (khác -1) < 2 hoặc số điểm không phải nhiễu < 2 cụm,
        các chỉ số nội tại sẽ trả về np.nan.
    """
    x_arr = np.asarray(x)
    labels_arr = np.asarray(labels)

    total_points = len(labels_arr)
    noise_mask = labels_arr == -1
    noise_count = int(np.sum(noise_mask))
    noise_frac = float(noise_count / total_points) if total_points > 0 else 0.0

    valid_mask = ~noise_mask
    valid_labels = labels_arr[valid_mask]
    valid_x = x_arr[valid_mask]

    unique_clusters = np.unique(valid_labels)
    n_clusters = int(len(unique_clusters))

    if n_clusters < 2 or len(valid_x) <= n_clusters:
        return {
            "n_clusters": n_clusters,
            "noise_frac": noise_frac,
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    sil = float(silhouette_score(valid_x, valid_labels))
    db = float(davies_bouldin_score(valid_x, valid_labels))
    ch = float(calinski_harabasz_score(valid_x, valid_labels))

    return {
        "n_clusters": n_clusters,
        "noise_frac": noise_frac,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
    }


# ---------------------------------------------------------------- Hierarchical Clustering
def run_hierarchical(
    x: pd.DataFrame | np.ndarray,
    n_clusters: int = 4,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> tuple[AgglomerativeClustering, np.ndarray]:
    """Huấn luyện mô hình AgglomerativeClustering.

    Tham số:
      - x: ma trận dữ liệu đã scale (DataFrame hoặc numpy array).
      - n_clusters: số lượng cụm cần chia (k >= 2).
      - linkage: phương pháp liên kết ('ward', 'complete', 'average', 'single').
                 Lưu ý: linkage='ward' bắt buộc metric='euclidean'.
      - metric: khoảng cách ('euclidean', 'cosine', 'manhattan',...).
    Trả về:
      - (model, labels): mô hình đã fit và mảng nhãn cụm (0 đến n_clusters - 1).
    """
    # scikit-learn >= 1.2 dùng metric thay cho affinity
    if linkage == "ward":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)

    labels = model.fit_predict(x)
    return model, labels


def tune_hierarchical(
    x: pd.DataFrame,
    feature_set_name: str,
    k_range: Iterable[int] = range(2, 11),
    linkages: Iterable[str] = ("ward", "average", "complete"),
) -> pd.DataFrame:
    """Thực nghiệm lưới siêu tham số cho Hierarchical Clustering.

    Lưới:
      - k: số lượng cụm (mặc định 2..10).
      - linkage: 'ward', 'average', 'complete'.
    Trả về DataFrame theo đúng TUNING_SCHEMA chuẩn của dự án (cột inertia để NaN).
    """
    rows: list[dict[str, Any]] = []

    for link in linkages:
        for k in k_range:
            _, labels = run_hierarchical(x, n_clusters=k, linkage=link)
            metrics = evaluate_clustering(x, labels)
            params_dict = {"n_clusters": k, "linkage": link, "metric": "euclidean"}

            rows.append({
                "algo": "Hierarchical",
                "feature_set": feature_set_name,
                "params": json.dumps(params_dict),
                "n_clusters": metrics["n_clusters"],
                "noise_frac": metrics["noise_frac"],
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "inertia": np.nan,
            })

    df_res = pd.DataFrame(rows)
    return df_res[TUNING_SCHEMA]


# ---------------------------------------------------------------- DBSCAN
def compute_k_distances(x: pd.DataFrame | np.ndarray, k: int = 5) -> np.ndarray:
    """Tính khoảng cách từ mỗi điểm đến láng giềng thứ k gần nhất của nó.

    Dùng để vẽ đồ thị k-distance plot hỗ trợ việc tìm điểm gãy (elbow/knee)
    nhằm chọn giá trị eps phù hợp cho DBSCAN với min_samples = k.
    Trả về mảng khoảng cách đã được sắp xếp tăng dần.
    """
    x_arr = np.asarray(x)
    nbrs = NearestNeighbors(n_neighbors=k).fit(x_arr)
    distances, _ = nbrs.kneighbors(x_arr)
    k_dist = distances[:, -1]
    k_dist.sort()
    return k_dist


def run_dbscan(
    x: pd.DataFrame | np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> tuple[DBSCAN, np.ndarray]:
    """Huấn luyện mô hình DBSCAN trên dữ liệu đã scale.

    Tham số:
      - x: ma trận đặc trưng.
      - eps: bán kính lân cận (epsilon).
      - min_samples: số điểm tối thiểu trong lân cận eps để cấu thành điểm lõi (core point).
      - metric: độ đo khoảng cách (mặc định 'euclidean').
    Trả về:
      - (model, labels): mô hình và mảng nhãn phân cụm (điểm nhiễu có nhãn -1).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(x)
    return model, labels


def tune_dbscan(
    x: pd.DataFrame,
    feature_set_name: str,
    eps_grid: Iterable[float],
    min_samples_grid: Iterable[int],
) -> pd.DataFrame:
    """Thực nghiệm lưới siêu tham số (eps, min_samples) cho DBSCAN.

    Tự động ghi nhận số cụm, tỷ lệ nhiễu và các metric nội tại tính trên tập nhãn != -1.
    Trả về DataFrame theo đúng TUNING_SCHEMA chuẩn của dự án.
    """
    rows: list[dict[str, Any]] = []

    for ms in min_samples_grid:
        for eps in eps_grid:
            _, labels = run_dbscan(x, eps=eps, min_samples=ms)
            metrics = evaluate_clustering(x, labels)
            params_dict = {"eps": round(float(eps), 3), "min_samples": int(ms), "metric": "euclidean"}

            rows.append({
                "algo": "DBSCAN",
                "feature_set": feature_set_name,
                "params": json.dumps(params_dict),
                "n_clusters": metrics["n_clusters"],
                "noise_frac": metrics["noise_frac"],
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "inertia": np.nan,
            })

    df_res = pd.DataFrame(rows)
    return df_res[TUNING_SCHEMA]

