"""Hình cho notebook 02: phân phối trước/sau log1p, ma trận tương quan, kích thước phân khúc baseline."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import FIGURES_DIR

# Cùng bảng màu với notebook EDA của nhóm.
C_MAIN, C_ALT, C_MUTED, C_INK = "#257c89", "#dc9343", "#8a8a8a", "#333333"
DIVERGING = LinearSegmentedColormap.from_list("alt_neutral_main", [C_ALT, "#f2f2f2", C_MAIN])


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_INK, labelsize=8)


def plot_skew_before_after(df: pd.DataFrame, report: pd.DataFrame, path=None):
    """Histogram từng đặc trưng: hàng trên là giá trị gốc, hàng dưới là log1p. Tiêu đề ghi skewness và quyết định."""
    cols = list(report.index)
    fig, axes = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 5.2))
    for j, c in enumerate(cols):
        applied = bool(report.loc[c, "log1p"])
        for i, (vals, skew) in enumerate([(df[c], report.loc[c, "skew_before"]),
                                          (np.log1p(df[c]), report.loc[c, "skew_log1p"])]):
            ax = axes[i, j]
            color = C_MAIN if (i == 0 or applied) else C_MUTED
            ax.hist(vals, bins=40, color=color, edgecolor="white", linewidth=0.3)
            _clean(ax)
            ax.set_yticks([])
            label = "gốc" if i == 0 else ("log1p (áp dụng)" if applied else "log1p (không áp dụng)")
            ax.set_title(f"{c}\n{label}\nskew = {skew:.2f}", fontsize=8, color=C_INK)
    fig.suptitle("Phân phối đặc trưng giai đoạn quan sát: gốc và sau log1p", fontsize=11, color=C_INK, y=1.02)
    fig.tight_layout()
    if path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_corr_heatmap(corr: pd.DataFrame, title: str, path=None):
    """Heatmap tương quan Pearson có ghi số; thang màu phân kỳ hai đầu, giữa là xám trung tính."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.95 * n + 1.5, 0.85 * n + 1))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else C_INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(title, fontsize=10, color=C_INK)
    fig.tight_layout()
    if path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_segment_sizes(counts: pd.Series, path=None):
    """Số khách theo phân khúc baseline (thanh ngang, ghi số khách và %)."""
    counts = counts[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(counts) + 1.2))
    ax.barh(counts.index, counts.to_numpy(), color=C_MAIN, height=0.6)
    total = counts.sum()
    for y, v in enumerate(counts.to_numpy()):
        ax.text(v + total * 0.008, y, f"{v:,} ({v / total:.1%})", va="center", fontsize=8, color=C_INK)
    ax.set_xlim(0, counts.max() * 1.25)
    _clean(ax)
    ax.set_xlabel("Số khách", fontsize=9, color=C_INK)
    ax.set_title("Phân khúc baseline luật RFM (giai đoạn quan sát)", fontsize=10, color=C_INK)
    fig.tight_layout()
    if path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
