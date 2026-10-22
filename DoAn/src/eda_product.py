"""EDA sản phẩm/quốc gia; không thay đổi dữ liệu đầu vào hay pipeline RFM."""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CUTOFF, FIGURES_DIR


def observation_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Chỉ dùng giai đoạn quan sát cho các nhận định tiền xử lý."""
    return raw.loc[raw['InvoiceDate'] < pd.Timestamp(CUTOFF)].copy()


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Các cờ độc lập có thể chồng lấp, không cộng để tính số dòng loại."""
    flags = {
        'Description thiếu/rỗng': df.Description.fillna('').str.strip().eq(''),
        'CustomerID thiếu': df.CustomerID.isna(),
        'Quantity âm': df.Quantity.lt(0),
        'Quantity bằng 0': df.Quantity.eq(0),
        'UnitPrice âm': df.UnitPrice.lt(0),
        'UnitPrice bằng 0': df.UnitPrice.eq(0),
        'Hóa đơn bắt đầu C': df.InvoiceNo.str.upper().str.startswith('C'),
        'Dòng trùng hoàn toàn (sau dòng đầu)': df.duplicated(),
    }
    return pd.DataFrame([
        {'issue': name, 'rows': int(mask.sum()), 'pct_rows': 100 * mask.mean()}
        for name, mask in flags.items()
    ])


def stockcode_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra mã khác mẫu phổ biến; mẫu ký tự chỉ là cờ rà soát."""
    candidate = ~df.StockCode.str.fullmatch(r'[0-9]{5}[A-Za-z]*', na=False)
    work = df.loc[candidate].assign(line_value=lambda x: x.Quantity * x.UnitPrice)
    return work.groupby('StockCode').agg(
        rows=('StockCode', 'size'),
        descriptions=('Description', lambda s: ' | '.join(sorted(set(s.dropna().str.strip())))),
        quantity=('Quantity', 'sum'), signed_value=('line_value', 'sum'),
        negative_qty_rows=('Quantity', lambda s: int(s.lt(0).sum())),
        max_price=('UnitPrice', 'max'),
    ).sort_values('rows', ascending=False)


def positive_product_sales(df: pd.DataFrame, excluded_codes: dict) -> tuple:
    """Bộ lọc nhẹ chỉ phục vụ EDA hàng hóa; giữ thiếu ID/mô tả và ngoại lệ lớn."""
    work = df.copy()
    steps = [{'step': 'Giai đoạn quan sát', 'remaining_rows': len(work), 'removed_rows': 0}]
    rules = [
        ('Bỏ hóa đơn C: không phải lượt bán dương', lambda x: ~x.InvoiceNo.str.upper().str.startswith('C')),
        ('Quantity > 0: tách trả hàng/điều chỉnh', lambda x: x.Quantity.gt(0)),
        ('UnitPrice > 0: tách miễn phí/giá lỗi', lambda x: x.UnitPrice.gt(0)),
        ('Bỏ mã phí/điều chỉnh đã kiểm tra mô tả', lambda x: ~x.StockCode.isin(excluded_codes)),
    ]
    for name, rule in rules:
        before = len(work)
        work = work.loc[rule(work)].copy()
        steps.append({'step': name, 'remaining_rows': len(work), 'removed_rows': before - len(work)})
    work['line_value'] = work.Quantity * work.UnitPrice
    work['is_non_uk'] = work.Country.ne('United Kingdom')
    return work, pd.DataFrame(steps)


def product_ranking(sales: pd.DataFrame) -> pd.DataFrame:
    """Gom theo mã, tránh tách một sản phẩm vì mô tả thay đổi."""
    return sales.groupby('StockCode').agg(
        description=('Description', lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else '(thiếu mô tả)'),
        quantity=('Quantity', 'sum'), revenue_gbp=('line_value', 'sum'),
        rows=('StockCode', 'size'),
    )


def country_profile(sales: pd.DataFrame) -> pd.DataFrame:
    """Mẫu số từng tỷ trọng được tách rõ; khách có thể xuất hiện ở nhiều nước."""
    result = sales.groupby('Country', dropna=False).agg(
        rows=('Country', 'size'), invoices=('InvoiceNo', 'nunique'),
        customers=('CustomerID', 'nunique'), revenue_gbp=('line_value', 'sum'),
    )
    result['row_pct'] = result.rows / len(sales) * 100
    result['revenue_pct'] = result.revenue_gbp / sales.line_value.sum() * 100
    return result.sort_values('rows', ascending=False)


def save_figure(fig, name: str):
    """Chỉ ghi hình thuộc phần việc Thành."""
    if not re.fullmatch(r'eda_prod_[a-z0-9_]+', name):
        raise ValueError('Tên hình phải có tiền tố eda_prod_.')
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_product_ranking(ranking: pd.DataFrame, top_n: int = 10):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, column, title, xlabel in zip(
        axes, ['quantity', 'revenue_gbp'], ['Top sản phẩm theo số lượng', 'Top sản phẩm theo doanh thu bán dương'],
        ['Số đơn vị hàng', 'GBP (chưa trừ trả hàng)'],
    ):
        top = ranking.nlargest(top_n, column).sort_values(column)
        labels = [f'{code} — {desc[:27]}' for code, desc in zip(top.index, top.description)]
        ax.barh(labels, top[column], color='#257c89')
        ax.set(title=title, xlabel=xlabel)
        ax.tick_params(axis='y', labelsize=8)
    save_figure(fig, 'eda_prod_top_products')


def plot_distributions(obs: pd.DataFrame) -> pd.DataFrame:
    """Giữ dấu ở toàn miền; p99 chỉ giới hạn hình phụ, không cắt dữ liệu."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    notes = []
    for row, column in enumerate(['Quantity', 'UnitPrice']):
        values = obs[column].dropna()
        signed_log = np.sign(values) * np.log1p(np.abs(values))
        axes[row, 0].hist(signed_log, bins=70, color='#257c89')
        axes[row, 0].set(title=f'{column}: toàn miền, giữ giá trị âm', xlabel='sign(x) × log1p(|x|)', ylabel='Số dòng')
        positive = values[values > 0]
        limit = positive.quantile(.99)
        axes[row, 1].hist(positive[positive <= limit], bins=50, color='#dc9343')
        axes[row, 1].set(title=f'{column}: phần dương đến p99 = {limit:,.2f}', xlabel=column, ylabel='Số dòng')
        notes.append({'variable': column, 'min': values.min(), 'max': values.max(),
                      'positive_p99': limit, 'positive_rows_above_p99': int((positive > limit).sum())})
    save_figure(fig, 'eda_prod_distributions')
    return pd.DataFrame(notes)


def plot_special_codes(audit: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5))
    audit.head(15).sort_values('rows').rows.plot.barh(ax=ax, color='#93629e')
    ax.set(title='Các mã cần rà soát: số dòng trước bộ lọc', xlabel='Số dòng (giai đoạn quan sát)', ylabel='StockCode')
    save_figure(fig, 'eda_prod_stockcodes')


def description_profile(obs: pd.DataFrame) -> pd.DataFrame:
    missing = obs.Description.fillna('').str.strip().eq('')
    statuses = {'Thiếu/rỗng': missing, 'Có mô tả': ~missing}
    return pd.DataFrame([
        {'description_status': name, 'rows': int(mask.sum()),
         'nonpositive_price_pct': 100 * obs.loc[mask, 'UnitPrice'].le(0).mean(),
         'negative_quantity_pct': 100 * obs.loc[mask, 'Quantity'].lt(0).mean(),
         'missing_customer_pct': 100 * obs.loc[mask, 'CustomerID'].isna().mean()}
        for name, mask in statuses.items()
    ]).set_index('description_status')


def plot_description(profile: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    profile.rows.plot.bar(ax=axes[0], rot=0, color='#257c89')
    axes[0].set(title='Description thiếu/rỗng và có mô tả', ylabel='Số dòng')
    for i, value in enumerate(profile.rows):
        axes[0].text(i, value, f'{value:,}', ha='center', va='bottom')
    profile.drop(columns='rows').plot.bar(ax=axes[1], rot=0)
    axes[1].set(title='Dấu hiệu chất lượng theo nhóm mô tả', ylabel='% dòng trong từng nhóm')
    axes[1].legend(['Giá ≤ 0', 'Số lượng âm', 'Thiếu CustomerID'], fontsize=8)
    save_figure(fig, 'eda_prod_description_missing')


def plot_countries(profile: pd.DataFrame, top_n: int = 10):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, column, title in zip(axes, ['row_pct', 'revenue_pct'], ['Tỷ trọng dòng bán dương', 'Tỷ trọng doanh thu bán dương']):
        top = profile.nlargest(top_n, column)[column].copy()
        top.loc['Các nước còn lại'] = profile[column].sum() - top.sum()
        top.sort_values().plot.barh(ax=ax, color='#257c89')
        ax.set(title=title, xlabel='% tổng tương ứng', ylabel='')
    save_figure(fig, 'eda_prod_countries')
