"""EDA theo thời gian: doanh thu, hóa đơn, khách theo tháng/tuần/giờ/thứ và so sánh hai giai đoạn quanh CUTOFF.

Chỉ mô tả. Không ước lượng scaler, ngưỡng hay tham số nào từ dữ liệu; dữ liệu kiểm chứng ở đây chỉ dùng
để kiểm tra hai giai đoạn có tương đương hay không, KHÔNG dùng để chọn đặc trưng hay mô hình.
"""
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from src.config import CUTOFF, FIGURES_DIR

OBS, VAL, MIX = 'Quan sát', 'Kiểm chứng', 'Chứa cutoff'
C_OBS, C_VAL, C_MIX, C_LINE = '#257c89', '#dc9343', '#8a8a8a', '#333333'
PHASE_COLOR = {OBS: C_OBS, VAL: C_VAL, MIX: C_MIX}
DOW_LABELS = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']


# ---------------------------------------------------------------- bộ lọc nhẹ
def light_sales(raw: pd.DataFrame):
    """Bộ lọc nhẹ chỉ để đo hoạt động bán hàng theo thời gian (không phải làm sạch cho RFM).

    Giữ dòng thiếu CustomerID (vẫn là doanh thu/hóa đơn thật; khi đếm khách thì tự bỏ qua <NA>),
    giữ dòng trùng và ngoại lệ lớn (chưa có căn cứ để coi là lỗi). Trả về (sales, log_df).
    """
    work = raw.copy()
    steps = [{'step': 'Toàn file', 'remaining_rows': len(work), 'removed_rows': 0}]
    rules = [
        ('Bỏ hóa đơn hủy (InvoiceNo bắt đầu C)', lambda x: ~x.InvoiceNo.str.upper().str.startswith('C')),
        ('Giữ Quantity > 0 (bỏ trả hàng/điều chỉnh)', lambda x: x.Quantity.gt(0)),
        ('Giữ UnitPrice > 0 (bỏ hàng miễn phí/giá lỗi)', lambda x: x.UnitPrice.gt(0)),
    ]
    for name, rule in rules:
        before = len(work)
        work = work.loc[rule(work)].copy()
        steps.append({'step': name, 'remaining_rows': len(work), 'removed_rows': before - len(work)})
    work['LineTotal'] = work.Quantity * work.UnitPrice
    work['Date'] = work.InvoiceDate.dt.normalize()
    work['GiaiDoan'] = np.where(work.InvoiceDate < pd.Timestamp(CUTOFF), OBS, VAL)
    return work.reset_index(drop=True), pd.DataFrame(steps)


def cancel_value_by_month(raw: pd.DataFrame) -> pd.Series:
    """Giá trị tuyệt đối trên hóa đơn hủy theo tháng (gồm cả mã phí/thủ công), để so với doanh thu gộp."""
    cancel = raw[raw.InvoiceNo.str.upper().str.startswith('C')]
    value = (cancel.Quantity * cancel.UnitPrice).abs()
    return value.groupby(cancel.InvoiceDate.dt.to_period('M')).sum()


# ---------------------------------------------------------------- tổng hợp theo thời gian
def aggregate_time(sales: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
    """Gộp theo tháng ('M') hoặc tuần thứ 2–CN ('W'): doanh thu, hóa đơn, khách, số ngày có giao dịch."""
    per = sales.InvoiceDate.dt.to_period('M' if freq == 'M' else 'W-SUN')
    out = sales.groupby(per).agg(
        revenue=('LineTotal', 'sum'), lines=('LineTotal', 'size'), invoices=('InvoiceNo', 'nunique'),
        customers=('CustomerID', 'nunique'), trading_days=('Date', 'nunique'),
    )
    no_cust = sales.LineTotal.where(sales.CustomerID.isna(), 0).groupby(per).sum()
    out['rev_per_trading_day'] = out.revenue / out.trading_days
    out['rev_no_customer_pct'] = no_cust / out.revenue * 100
    out['start'] = out.index.start_time
    out['end'] = out.index.end_time.normalize()
    cut, dmin, dmax = pd.Timestamp(CUTOFF), sales.Date.min(), sales.Date.max()
    # partial: kỳ kéo dài ra ngoài khoảng dữ liệu (tuần/tháng đầu và cuối)
    out['partial'] = (out.start < dmin) | (out.end > dmax)
    out['phase'] = np.select([out.end < cut, out.start >= cut], [OBS, VAL], default=MIX)
    out.insert(0, 'period', out.index.astype(str))
    return out.reset_index(drop=True)


def no_trading_runs(sales: pd.DataFrame, min_days: int = 1) -> pd.DataFrame:
    """Chuỗi ngày giao dịch kỳ vọng liên tiếp không có hóa đơn nào.

    Ngày kỳ vọng = mọi ngày lịch trừ thứ chưa từng có giao dịch trong dữ liệu (thứ 7), nên thứ 7 không bị tính là 'nghỉ'.
    """
    present_dow = set(sales.Date.dt.dayofweek.unique())
    days = pd.date_range(sales.Date.min(), sales.Date.max())
    expected = days[days.dayofweek.isin(present_dow)]
    missing = expected.difference(pd.DatetimeIndex(sales.Date.unique()))
    if len(missing) == 0:
        return pd.DataFrame(columns=['start', 'end', 'trading_days_missing', 'calendar_days'])
    pos = pd.Series(np.arange(len(expected)), index=expected).loc[missing].to_numpy()
    group = (np.diff(pos, prepend=pos[0] - 1) != 1).cumsum()
    runs = pd.DataFrame({'date': missing, 'group': group}).groupby('group').date.agg(['min', 'max', 'size'])
    runs.columns = ['start', 'end', 'trading_days_missing']
    runs['calendar_days'] = (runs.end - runs.start).dt.days + 1
    return runs[runs.trading_days_missing >= min_days].reset_index(drop=True)


# ---------------------------------------------------------------- giờ trong ngày, thứ trong tuần
def invoice_table(sales: pd.DataFrame) -> pd.DataFrame:
    """Mỗi hóa đơn một dòng (lấy thời điểm sớm nhất) để đếm theo giờ/thứ không bị lặp theo số dòng."""
    inv = sales.groupby('InvoiceNo').agg(
        InvoiceDate=('InvoiceDate', 'min'), revenue=('LineTotal', 'sum'),
        CustomerID=('CustomerID', 'first'), GiaiDoan=('GiaiDoan', 'first'),
    )
    inv['hour'] = inv.InvoiceDate.dt.hour
    inv['dow'] = inv.InvoiceDate.dt.dayofweek
    return inv


def hour_profile(inv: pd.DataFrame) -> pd.DataFrame:
    """Số hóa đơn, doanh thu và tỷ trọng doanh thu theo giờ (kèm tỷ trọng riêng từng giai đoạn)."""
    out = inv.groupby('hour').agg(invoices=('revenue', 'size'), revenue=('revenue', 'sum'))
    out['revenue_pct'] = out.revenue / out.revenue.sum() * 100
    by_phase = inv.groupby(['hour', 'GiaiDoan']).revenue.sum().unstack(fill_value=0)
    for phase, tag in ((OBS, 'obs'), (VAL, 'val')):
        out[f'revenue_pct_{tag}'] = (by_phase[phase] / by_phase[phase].sum() * 100).reindex(out.index).fillna(0)
    return out


def weekday_profile(inv: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Theo thứ: chia cho số ngày giao dịch của thứ đó để không lệch do số tuần mỗi thứ khác nhau."""
    out = pd.DataFrame(index=range(7))
    out['label'] = DOW_LABELS
    out['trading_days'] = sales.groupby(sales.Date.dt.dayofweek).Date.nunique().reindex(range(7), fill_value=0)
    out['invoices'] = inv.groupby('dow').size().reindex(range(7), fill_value=0)
    out['revenue'] = inv.groupby('dow').revenue.sum().reindex(range(7), fill_value=0)
    days = out.trading_days.replace(0, np.nan)
    out['invoices_per_day'] = (out.invoices / days).fillna(0)
    out['revenue_per_day'] = (out.revenue / days).fillna(0)
    out['revenue_pct'] = out.revenue / out.revenue.sum() * 100
    return out


# ---------------------------------------------------------------- hóa đơn lớn, tháng 12, hai giai đoạn
def largest_invoices(sales: pd.DataFrame, raw: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Top hóa đơn theo giá trị, kèm hóa đơn hủy khớp (cùng CustomerID, StockCode, Quantity đối nhau) nếu có."""
    top = sales.groupby('InvoiceNo').agg(
        InvoiceDate=('InvoiceDate', 'min'), CustomerID=('CustomerID', 'first'),
        lines=('LineTotal', 'size'), revenue=('LineTotal', 'sum'),
    ).nlargest(n, 'revenue')
    cancel = raw[raw.InvoiceNo.str.upper().str.startswith('C') & raw.CustomerID.notna()]
    cancel = pd.DataFrame({
        'CustomerID': cancel.CustomerID, 'StockCode': cancel.StockCode, 'Quantity': -cancel.Quantity,
        'cancel_invoice': cancel.InvoiceNo, 'cancel_date': cancel.InvoiceDate,
        'cancel_value': (cancel.Quantity * cancel.UnitPrice).abs(),
    })
    rows = []
    for no, r in top.iterrows():
        lines = sales.loc[sales.InvoiceNo == no, ['CustomerID', 'StockCode', 'Quantity', 'LineTotal']]
        match = lines.merge(cancel, on=['CustomerID', 'StockCode', 'Quantity'], how='inner')
        # chỉ nhận hóa đơn hủy xảy ra sau hóa đơn gốc và có đúng giá trị dòng gốc (tránh khớp nhầm mã thủ công 'M')
        match = match[(match.cancel_date >= r.InvoiceDate) & ((match.cancel_value - match.LineTotal).abs() < 0.01)]
        rows.append({
            'InvoiceNo': no, 'InvoiceDate': r.InvoiceDate, 'CustomerID': r.CustomerID,
            'lines': r.lines, 'revenue': r.revenue,
            'cancel_invoice': match.cancel_invoice.iloc[0] if len(match) else None,
            'cancel_minutes_later': (match.cancel_date.iloc[0] - r.InvoiceDate).total_seconds() / 60 if len(match) else np.nan,
            'cancelled_value': float(match.cancel_value.sum()),
        })
    return pd.DataFrame(rows)


def december_like_for_like(sales: pd.DataFrame, days: int = 9):
    """So 1–9/12/2010 với 1–9/12/2011 (cùng số ngày lịch) để biết tháng 12/2011 cao hơn thật hay do cắt ngắn."""
    daily, rows = {}, []
    top_invoice = None
    for year in (2010, 2011):
        x = sales[(sales.InvoiceDate >= f'{year}-12-01') & (sales.InvoiceDate < f'{year}-12-{days + 1:02d}')]
        daily[year] = x.groupby(x.Date.dt.day).LineTotal.sum().reindex(range(1, days + 1), fill_value=0)
        rows.append({'case': f'1–{days}/12/{year}', 'trading_days': x.Date.nunique(), 'invoices': x.InvoiceNo.nunique(),
                     'customers': x.CustomerID.nunique(), 'revenue': x.LineTotal.sum()})
        if year == 2011:
            by_inv = x.groupby('InvoiceNo').LineTotal.sum()
            top_invoice = by_inv.idxmax()
            rest = x[x.InvoiceNo != top_invoice]
            rows.append({'case': f'1–{days}/12/2011, bỏ hóa đơn lớn nhất ({top_invoice})',
                         'trading_days': rest.Date.nunique(), 'invoices': rest.InvoiceNo.nunique(),
                         'customers': rest.CustomerID.nunique(), 'revenue': rest.LineTotal.sum()})
    return pd.DataFrame(daily), pd.DataFrame(rows).set_index('case'), top_invoice


def window_rev_per_day(sales: pd.DataFrame, start: str, end: str) -> float:
    """Doanh thu bình quân mỗi ngày có giao dịch trong [start, end)."""
    x = sales[(sales.InvoiceDate >= start) & (sales.InvoiceDate < end)]
    return float(x.LineTotal.sum() / x.Date.nunique())


def compare_periods(raw: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Bảng so sánh giai đoạn quan sát và kiểm chứng; cột cuối là tỷ lệ Kiểm chứng / Quan sát."""
    cut = pd.Timestamp(CUTOFF)
    dmin, dmax = sales.Date.min(), sales.Date.max()
    daily_customers = sales.groupby(['GiaiDoan', 'Date']).CustomerID.nunique().groupby(level=0).mean()
    spans = {OBS: (dmin, cut - pd.Timedelta(days=1)), VAL: (cut, dmax)}
    cols = {}
    for name, (a, b) in spans.items():
        s = sales[sales.GiaiDoan == name]
        r = raw[raw.InvoiceDate < cut] if name == OBS else raw[raw.InvoiceDate >= cut]
        days = s.Date.nunique()
        cols[name] = {
            'Ngày lịch trong giai đoạn': (b - a).days + 1,
            'Ngày có giao dịch bán': days,
            'Dòng gốc (chưa lọc)': len(r),
            'Dòng bán hợp lệ (sau bộ lọc nhẹ)': len(s),
            'Hóa đơn bán': s.InvoiceNo.nunique(),
            'Khách trong dữ liệu gốc (kể cả chỉ có hóa đơn hủy)': r.CustomerID.nunique(),
            'Khách có mua (CustomerID không thiếu)': s.CustomerID.nunique(),
            'Doanh thu gộp (GBP, chưa trừ hủy)': s.LineTotal.sum(),
            'Doanh thu / ngày giao dịch (GBP)': s.LineTotal.sum() / days,
            'Hóa đơn / ngày giao dịch': s.InvoiceNo.nunique() / days,
            'Khách khác nhau / ngày giao dịch (TB)': daily_customers[name],
            'Doanh thu thuộc dòng thiếu CustomerID (%)': s.LineTotal.where(s.CustomerID.isna(), 0).sum() / s.LineTotal.sum() * 100,
        }
    table = pd.DataFrame(cols)
    table['Kiểm chứng / Quan sát'] = table[VAL] / table[OBS]
    return table


def format_periods(table: pd.DataFrame) -> pd.DataFrame:
    """Bản chữ để hiển thị bảng so sánh: số đếm nguyên, tiền/tỷ lệ có dấu phân cách; bản số vẫn dùng để lưu CSV."""
    def one(label, value):
        if '(%)' in label:
            return f'{value:,.1f}'
        if '/ ngày' in label:
            return f'{value:,.2f}'
        return f'{value:,.0f}'

    out = pd.DataFrame({col: [one(label, v) for label, v in table[col].items()] for col in (OBS, VAL)}, index=table.index)
    out['Kiểm chứng / Quan sát'] = [f'{v:.2f}' for v in table['Kiểm chứng / Quan sát']]
    return out


def customer_overlap(sales: pd.DataFrame) -> dict:
    """Chỉ đếm khách giao nhau giữa hai giai đoạn (mô tả, không dùng chọn mô hình)."""
    obs_ids = set(sales.loc[sales.GiaiDoan == OBS, 'CustomerID'].dropna())
    val_ids = set(sales.loc[sales.GiaiDoan == VAL, 'CustomerID'].dropna())
    return {'obs_customers': len(obs_ids), 'val_customers': len(val_ids),
            'in_both': len(obs_ids & val_ids), 'only_val': len(val_ids - obs_ids), 'only_obs': len(obs_ids - val_ids)}


# ---------------------------------------------------------------- vẽ hình
def save_figure(fig, name: str):
    """Chỉ ghi hình thuộc phần việc EDA thời gian (tiền tố eda_time_)."""
    if not re.fullmatch(r'eda_time_[a-z0-9_]+', name):
        raise ValueError('Tên hình phải có tiền tố eda_time_.')
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def _num(ts):
    return mdates.date2num(pd.Timestamp(ts))


def _thousands(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:,.0f}k'))


def _cutoff_line(ax):
    ax.axvline(_num(CUTOFF), color=C_LINE, ls='--', lw=1.3)


def _period_bars(ax, agg, column, gap_days):
    for _, r in agg.iterrows():
        ax.bar(_num(r.start), r[column], width=(r.end - r.start).days + 1 - gap_days, align='edge',
               color=PHASE_COLOR[r.phase], hatch='///' if r.partial else None, edgecolor='white', linewidth=0.4)
    ax.xaxis_date()
    ax.set_ylim(0, agg[column].max() * 1.25)


def _month_ticks(ax, agg):
    """Nhãn tháng đặt giữa cột để khỏi đọc lệch sang tháng trước."""
    mids = [_num(r.start) + ((r.end - r.start).days + 1) / 2 for _, r in agg.iterrows()]
    ax.set_xticks(mids, [f'{r.start:%m/%y}' for _, r in agg.iterrows()])


def _phase_legend(ax, partial_label, loc='upper left', ncol=2):
    handles = [Patch(color=C_OBS, label='Giai đoạn quan sát'), Patch(color=C_MIX, label='Kỳ chứa CUTOFF'),
               Patch(color=C_VAL, label='Giai đoạn kiểm chứng'),
               Patch(facecolor='#bbbbbb', edgecolor='white', hatch='///', label=partial_label),
               plt.Line2D([], [], color=C_LINE, ls='--', label=f'CUTOFF {CUTOFF}')]
    ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=8, frameon=False)


def plot_monthly(monthly: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    specs = [('revenue', 'Doanh thu gộp theo tháng (nghìn GBP)', True),
             ('invoices', 'Số hóa đơn theo tháng', False),
             ('customers', 'Số khách (CustomerID) hoạt động theo tháng', False),
             ('rev_per_trading_day', 'Doanh thu / ngày giao dịch (nghìn GBP)', True)]
    for ax, (col, title, k) in zip(axes.ravel(), specs):
        _period_bars(ax, monthly, col, gap_days=2)
        _cutoff_line(ax)
        ax.set_title(title, fontsize=11)
        if k:
            _thousands(ax)
    for _, r in monthly[monthly.partial].iterrows():
        axes[1, 1].annotate(f'{r.trading_days} ngày', (_num(r.start) + 12, r.rev_per_trading_day), ha='center',
                            va='bottom', fontsize=8)
    _phase_legend(axes[0, 0], 'Tháng chưa đủ dữ liệu (đến 09/12/2011)')
    _month_ticks(axes[1, 0], monthly)
    save_figure(fig, 'eda_time_monthly')


def plot_weekly(weekly: pd.DataFrame, gaps: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    specs = [('revenue', 'Doanh thu gộp theo tuần T2–CN (nghìn GBP)', True),
             ('invoices', 'Số hóa đơn theo tuần', False),
             ('customers', 'Số khách hoạt động theo tuần', False)]
    for ax, (col, title, k) in zip(axes, specs):
        _period_bars(ax, weekly, col, gap_days=1)
        _cutoff_line(ax)
        ax.set_title(title, fontsize=11)
        if k:
            _thousands(ax)
        for _, g in gaps.iterrows():
            ax.axvspan(_num(g.start), _num(g.end) + 1, color='#bbbbbb', alpha=0.35, lw=0)
    for _, g in gaps.iterrows():
        axes[0].text(_num(g.start) + (g.calendar_days / 2), axes[0].get_ylim()[1] * 0.88,
                     f'không có GD\n{g.start:%d/%m}–{g.end:%d/%m}', ha='center', va='top', fontsize=7.5, color=C_LINE)
    _phase_legend(axes[0], 'Tuần chưa đủ (đầu/cuối dữ liệu)', loc='upper center', ncol=5)
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    save_figure(fig, 'eda_time_weekly')


def plot_hour_dow(hourly: pd.DataFrame, weekday: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]
    ax.bar(hourly.index, hourly.invoices, color=C_OBS)
    ax.set(title='Số hóa đơn theo giờ trong ngày (cả hai giai đoạn)', xlabel='Giờ (theo giờ ghi trong dữ liệu)',
           ylabel='Số hóa đơn')
    ax = axes[0, 1]
    ax.plot(hourly.index, hourly['revenue_pct_obs'], marker='o', color=C_OBS, label='Quan sát')
    ax.plot(hourly.index, hourly['revenue_pct_val'], marker='o', color=C_VAL, label='Kiểm chứng')
    ax.set(title='Tỷ trọng doanh thu theo giờ, so hai giai đoạn', xlabel='Giờ', ylabel='% doanh thu của giai đoạn')
    ax.legend(frameon=False)
    for ax, col, title, ylabel in [(axes[1, 0], 'invoices_per_day', 'Hóa đơn / ngày giao dịch theo thứ', 'Số hóa đơn'),
                                   (axes[1, 1], 'revenue_per_day', 'Doanh thu / ngày giao dịch theo thứ', 'nghìn GBP')]:
        ax.bar(weekday.label, weekday[col], color=C_OBS)
        ax.set(title=title, ylabel=ylabel)
        empty = weekday[weekday.trading_days == 0]
        for _, r in empty.iterrows():
            ax.text(r.label, ax.get_ylim()[1] * 0.04, 'không có\ngiao dịch', ha='center', fontsize=8, color=C_LINE)
    _thousands(axes[1, 1])
    save_figure(fig, 'eda_time_hour_dow')


def plot_dec_compare(daily: pd.DataFrame, monthly: pd.DataFrame, top_invoice_value: float, top_invoice: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(daily))
    n = len(daily)
    ax.bar(x - 0.2, daily[2010], width=0.4, color=C_OBS, label=f'01–{n:02d}/12/2010')
    ax.bar(x + 0.2, daily[2011], width=0.4, color=C_VAL, label=f'01–{n:02d}/12/2011')
    ax.set_xticks(x, [str(d) for d in daily.index])
    ax.set(title=f'Doanh thu theo ngày trong tháng 12: cùng {n} ngày đầu',
           xlabel='Ngày trong tháng 12 (ngày không có cột là thứ Bảy, không giao dịch)', ylabel='nghìn GBP')
    _thousands(ax)
    ax.annotate(f'HĐ {top_invoice}\n£{top_invoice_value / 1000:,.0f}k', (x[-1] + 0.2, daily[2011].iloc[-1]),
                xytext=(x[-1] - 2.6, daily[2011].max() * 0.9), fontsize=8,
                arrowprops={'arrowstyle': '->', 'color': C_LINE})
    ax.legend(frameon=False, loc='upper left')
    ax = axes[1]
    for _, r in monthly.iterrows():
        ax.bar(_num(r.start), r.trading_days, width=(r.end - r.start).days - 1, align='edge',
               color=PHASE_COLOR[r.phase], hatch='///' if r.partial else None, edgecolor='white', linewidth=0.4)
        ax.text(_num(r.start) + 14, r.trading_days + 0.4, str(r.trading_days), ha='center', fontsize=8)
    ax.xaxis_date()
    _month_ticks(ax, monthly)
    _cutoff_line(ax)
    ax.set(title='Số ngày có giao dịch mỗi tháng', ylabel='Ngày')
    ax.set_ylim(0, monthly.trading_days.max() * 1.2)
    save_figure(fig, 'eda_time_dec_compare')


def plot_periods(table: pd.DataFrame):
    keys = ['Ngày có giao dịch bán', 'Doanh thu / ngày giao dịch (GBP)', 'Hóa đơn / ngày giao dịch',
            'Khách khác nhau / ngày giao dịch (TB)']
    titles = ['Số ngày có giao dịch', 'Doanh thu / ngày giao dịch (GBP)', 'Hóa đơn / ngày giao dịch',
              'Khách khác nhau / ngày giao dịch']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, key, title in zip(axes, keys, titles):
        vals = [table.loc[key, OBS], table.loc[key, VAL]]
        bars = ax.bar(['Quan sát', 'Kiểm chứng'], vals, color=[C_OBS, C_VAL])
        ax.bar_label(bars, labels=[f'{v:,.0f}' for v in vals], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    save_figure(fig, 'eda_time_periods')
