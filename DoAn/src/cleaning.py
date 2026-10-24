"""Làm sạch dữ liệu giao dịch Online Retail.

Mọi quy tắc ở đây là quy tắc logic/định dạng (không dựa ngưỡng thống kê tính từ dữ liệu),
nên không dùng thông tin từ giai đoạn kiểm chứng để ra quyết định (không gây data leakage).
"""
from __future__ import annotations

import pandas as pd

from src.config import TABLES_DIR

# Mã StockCode KHÔNG phải sản phẩm (phí vận chuyển, thủ công, chiết khấu, hoa hồng, phí ngân hàng).
# Danh sách lấy từ dữ liệu thật (xem notebook 01, mục B5): đây là các mã xuất hiện ở dòng có CustomerID.
# Các mã không phải sản phẩm khác (AMAZONFEE, S, B, gift_0001_*, ...) chỉ xuất hiện ở dòng thiếu
# CustomerID nên đã bị loại ở bước trước.
NON_PRODUCT_CODES = ["POST", "DOT", "M", "m", "C2", "D", "CRUK", "BANK CHARGES"]


def clean_transactions(df: pd.DataFrame):
    """Làm sạch giao dịch và tách giao dịch bán / hóa đơn hủy.

    Thứ tự bước (số dòng bị loại được đếm TUẦN TỰ: một dòng chỉ tính vào lý do đầu tiên nó dính):
      1. Loại dòng trùng lặp hoàn toàn.
      2. Loại dòng thiếu CustomerID (không gắn được vào khách để phân khúc).
      3. Tách hóa đơn hủy (InvoiceNo bắt đầu bằng 'C') sang cancel_df (giữ lại để tính CancelRate).
      4. Với giao dịch bán: loại Quantity <= 0 và UnitPrice <= 0 (không phải doanh thu bán thật).
      5. Loại các dòng StockCode không phải sản phẩm ở cả sales_df và cancel_df.

    Trả về: (sales_df, cancel_df, log_dict). Cả hai DataFrame có thêm cột LineTotal = Quantity * UnitPrice.
    Ngoại lệ về Quantity/UnitPrice KHÔNG bị xóa ở bước này (xử lý ở mức khách hàng khi tạo đặc trưng).
    """
    log = {"rows_raw": len(df)}

    # 1. Trùng lặp
    n0 = len(df)
    df = df.drop_duplicates()
    log["removed_duplicates"] = n0 - len(df)

    # 2. Thiếu CustomerID
    n0 = len(df)
    df = df[df["CustomerID"].notna()].copy()
    log["removed_missing_customer"] = n0 - len(df)

    # 3. Tách hóa đơn hủy
    is_cancel = df["InvoiceNo"].astype(str).str.startswith("C")
    cancel_df = df[is_cancel].copy()
    sales_df = df[~is_cancel].copy()
    log["cancel_rows_split"] = len(cancel_df)

    # 4. Giao dịch bán không hợp lệ
    n0 = len(sales_df)
    sales_df = sales_df[sales_df["Quantity"] > 0]
    log["removed_nonpositive_quantity"] = n0 - len(sales_df)
    n0 = len(sales_df)
    sales_df = sales_df[sales_df["UnitPrice"] > 0]
    log["removed_nonpositive_price"] = n0 - len(sales_df)

    # 5. Mã không phải sản phẩm
    n0 = len(sales_df)
    sales_df = sales_df[~sales_df["StockCode"].isin(NON_PRODUCT_CODES)]
    log["removed_nonproduct_sales"] = n0 - len(sales_df)
    n0 = len(cancel_df)
    cancel_df = cancel_df[~cancel_df["StockCode"].isin(NON_PRODUCT_CODES)]
    log["removed_nonproduct_cancel"] = n0 - len(cancel_df)

    for d in (sales_df, cancel_df):
        d["LineTotal"] = d["Quantity"] * d["UnitPrice"]

    sales_df = sales_df.reset_index(drop=True)
    cancel_df = cancel_df.reset_index(drop=True)
    log["rows_sales"] = len(sales_df)
    log["rows_cancel"] = len(cancel_df)
    log["customers_sales"] = int(sales_df["CustomerID"].nunique())
    log["customers_cancel"] = int(cancel_df["CustomerID"].nunique())
    return sales_df, cancel_df, log


def save_cleaning_log(log: dict, path=None):
    """Lưu log làm sạch ra CSV (cột: step, rows)."""
    path = path or (TABLES_DIR / "cleaning_log.csv")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"step": list(log.keys()), "rows": list(log.values())}).to_csv(path, index=False)
    return path
