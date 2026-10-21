"""Đọc dữ liệu gốc Online Retail."""
import pandas as pd

from src.config import DATA_RAW, DATA_INTERIM, RAW_CACHE


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Đọc Online_Retail.xlsx (lần đầu chậm), cache sang pickle để lần sau đọc nhanh.

    Kiểu dữ liệu: InvoiceNo, StockCode -> str; CustomerID -> Int64 nullable
    (CustomerID thiếu giữ nguyên là <NA>, không bị ép thành float).
    """
    if use_cache and RAW_CACHE.exists():
        return pd.read_pickle(RAW_CACHE)

    if not DATA_RAW.exists():
        raise FileNotFoundError(
            f"Không thấy {DATA_RAW}. Tải Online_Retail.xlsx từ "
            "https://archive.ics.uci.edu/dataset/352/online+retail và đặt vào data/raw/."
        )

    df = pd.read_excel(DATA_RAW, dtype={"InvoiceNo": str, "StockCode": str}, engine="openpyxl")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_pickle(RAW_CACHE)
    return df
