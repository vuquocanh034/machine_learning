"""Hình cho notebook 03 (K-Means): đường tuning theo k, tâm cụm, kích thước cụm."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clustering import elbow_knee
from src.config import FIGURES_DIR
from src.viz_features import C_ALT, C_INK, C_MAIN, C_MUTED, DIVERGING, _clean

# Mỗi bộ đặc trưng một màu; mỗi cách khởi tạo một kiểu nét (hai đường gần trùng nhau là bằng chứng init ít ảnh hưởng).
SET_COLOR = {"RFM": C_MAIN, "RFM_PLUS": C_ALT}
INIT_STYLE = {"k-means++": dict(marker="o", ls="-", ms=5), "random": dict(marker="s", ls="--", ms=4, mfc="white")}


def _save(fig, path):
    if path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")


def _tuning_panels(t: pd.DataFrame, panels, title: str, path=None, mark_knee=False):
    """Lưới (hàng = bộ đặc trưng, cột = metric) theo k. `t` cần cột feature_set, k, init và các metric."""
    sets = list(t["feature_set"].unique())
    fig, axes = plt.subplots(len(sets), len(panels), figsize=(4.6 * len(panels), 3.4 * len(sets)), squeeze=False)
    for i, fs in enumerate(sets):
        d = t[t["feature_set"] == fs]
        for j, (col, label, note) in enumerate(panels):
            ax = axes[i, j]
            for init, g in d.groupby("init"):
                g = g.sort_values("k")
                ax.plot(g["k"], g[col], color=SET_COLOR.get(fs, C_MAIN), label=init, **INIT_STYLE.get(init, {}))
            if mark_knee and col == "inertia":
                best = d.sort_values("inertia").groupby("k", as_index=False).head(1).sort_values("k")
                kk = elbow_knee(best["k"], best["inertia"])
                ax.axvline(kk, color=C_MUTED, ls=":", lw=1)
                ax.text(kk + 0.1, ax.get_ylim()[1] * 0.97, f"khuỷu k={kk}", fontsize=8, color=C_INK, va="top")
            ax.set_xticks(sorted(d["k"].unique()))
            _clean(ax)
            ax.set_xlabel("k", fontsize=8, color=C_INK)
            ax.set_title(f"{fs}: {label} {note}", fontsize=9, color=C_INK)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, title="init", title_fontsize=7)
    fig.suptitle(title, fontsize=11, color=C_INK, y=1.01)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_elbow_silhouette(t: pd.DataFrame, path=None):
    """Elbow (inertia) và Silhouette theo k cho từng bộ đặc trưng."""
    return _tuning_panels(
        t, [("inertia", "Inertia", "(càng thấp càng khớp, luôn giảm theo k)"), ("silhouette", "Silhouette", "(càng cao càng tốt)")],
        "K-Means: Elbow và Silhouette theo k (giai đoạn quan sát)", path, mark_knee=True)


def plot_db_ch(t: pd.DataFrame, path=None):
    """Davies–Bouldin và Calinski–Harabasz theo k cho từng bộ đặc trưng."""
    return _tuning_panels(
        t, [("davies_bouldin", "Davies–Bouldin", "(càng thấp càng tốt)"), ("calinski_harabasz", "Calinski–Harabasz", "(càng cao càng tốt)")],
        "K-Means: Davies–Bouldin và Calinski–Harabasz theo k", path)


def plot_centroid_heatmaps(items, path=None, vlim: float = 2.0):
    """Heatmap tâm cụm (z-score trong không gian đã scale), mỗi cấu hình một ô.

    `items`: danh sách (tiêu đề, DataFrame tâm cụm [cụm x đặc trưng], Series số khách mỗi cụm).
    Màu = trung bình z-score của cụm; nhãn trục tung ghi cả số khách và tỷ lệ.
    """
    # constrained layout để nhãn trục tung (có ghi số khách) của ô sau không đè lên số liệu của ô trước
    cols = [c.shape[1] for _, c, _ in items]
    fig, axes = plt.subplots(1, len(items), figsize=(sum(0.6 * n + 3.0 for n in cols) + 1.2,
                                                     0.62 * max(c.shape[0] for _, c, _ in items) + 2.4),
                             gridspec_kw={"width_ratios": cols}, squeeze=False, layout="constrained")
    im = None
    for ax, (title, cent, sizes) in zip(axes[0], items):
        im = ax.imshow(cent.to_numpy(), cmap=DIVERGING, vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_xticks(range(cent.shape[1]), cent.columns, rotation=40, ha="right", fontsize=8)
        ylab = [f"cụm {c}  (n={int(sizes[c]):,}, {sizes[c] / sizes.sum():.0%})" for c in cent.index]
        ax.set_yticks(range(cent.shape[0]), ylab, fontsize=8)
        for i in range(cent.shape[0]):
            for j in range(cent.shape[1]):
                v = cent.iloc[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 1.2 else C_INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=9, color=C_INK)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="z-score (đã log1p + scale)")
    fig.suptitle("Tâm cụm K-Means trong không gian đã chuẩn hóa", fontsize=11, color=C_INK, y=1.02)
    _save(fig, path)
    return fig
