"""
backtester.py
Cross-sectional IC evaluation và portfolio metrics.

Quá trình tính return:
  1. Mỗi ngày t trong OOS period:
     - Long leg  = các ticker có signal[t] > median → equal weight
     - Short leg = các ticker có signal[t] <= median → equal weight
     - daily_pnl[t] = mean(long leg returns[t]) - mean(short leg returns[t])
       = long-short spread return (dollar-neutral, equal-weight)

  2. Annualized return = geometric:
     total_return = prod(1 + daily_pnl) - 1
     ann_return   = (1 + total_return)^(252/n_days) - 1

  Ý nghĩa: spread return chưa trừ transaction cost, chưa tính leverage.

Sharpe:
  Sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252)
  Đây là Sharpe của long-short spread portfolio.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from config import DEFAULT_CONFIG


def _is_constant_series(s: pd.Series) -> bool:
    """True nếu series không có đủ biến thiên để tính correlation."""
    if s is None:
        return True
    x = s.dropna()
    if len(x) < 2:
        return True
    return x.nunique() < 2


# ── Cross-sectional IC ────────────────────────────────────────────────

def compute_ic_cross_sectional(
    signal_df: pd.DataFrame,
    fwd_ret_df: pd.DataFrame,
) -> float:
    """
    Tính Spearman IC cross-sectional.
    Tại mỗi ngày t: corr(signal[t, :], fwd_ret[t, :]) across tickers.
    Returns: (mean_ic, ic_ir, ic_series)
      mean_ic:   E[IC_t]
      ic_ir:     mean_ic / std_ic  — đo tính ổn định của alpha
      ic_series: Series IC theo ngày
    """
    common_dates   = signal_df.index.intersection(fwd_ret_df.index)
    common_tickers = signal_df.columns.intersection(fwd_ret_df.columns)

    sig = signal_df.loc[common_dates, common_tickers]
    fwd = fwd_ret_df.loc[common_dates, common_tickers]

    # Adaptive min_tickers: 30% universe, floor tại 10
    n_universe = len(common_tickers)
    min_tickers_for_ic = max(10, int(n_universe * 0.30))

    ic_list = []
    for date in common_dates:
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        common_t = row_sig.index.intersection(row_fwd.index)
        if len(common_t) < min_tickers_for_ic:
            continue
        s_sig = row_sig[common_t]
        s_fwd = row_fwd[common_t]
        if _is_constant_series(s_sig) or _is_constant_series(s_fwd):
            continue
        ic = s_sig.corr(s_fwd, method="spearman")
        if not np.isnan(ic):
            ic_list.append((date, ic))

    if not ic_list:
        return np.nan, np.nan, pd.Series(dtype=float)

    ic_series = pd.Series({d: v for d, v in ic_list}, name="ic")
    mean_ic   = float(ic_series.mean())

    return mean_ic


def compute_ic_cross_sectional_oos(
    signal_df: pd.DataFrame,
    fwd_ret_df: pd.DataFrame,
    test_ratio: float = None,
) -> Tuple[float, float]:
    """
    Walk-forward cross-sectional IC, tách IS/OOS theo thời gian.
    Returns: (mean_ic_is, mean_ic_oos, ic_ir_is, ic_ir_oos)
    """
    if test_ratio is None:
        test_ratio = DEFAULT_CONFIG.test_ratio

    common_dates = sorted(signal_df.index.intersection(fwd_ret_df.index))
    if len(common_dates) < 60:
        ic = compute_ic_cross_sectional(signal_df, fwd_ret_df)
        return ic, ic

    split_idx   = int(len(common_dates) * (1 - test_ratio))
    train_dates = common_dates[:split_idx]
    test_dates  = common_dates[split_idx:]

    ic_is = compute_ic_cross_sectional(
        signal_df.loc[train_dates], fwd_ret_df.loc[train_dates]
    )
    ic_oos = compute_ic_cross_sectional(
        signal_df.loc[test_dates],  fwd_ret_df.loc[test_dates]
    )
    return ic_is, ic_oos


# ── Portfolio daily PnL helper ────────────────────────────────────────

def _build_daily_pnl(
    signal_df: pd.DataFrame,
    fwd_ret_df: pd.DataFrame,
    test_dates: list,
) -> np.ndarray:
    """
    Tính daily portfolio return với rank-based continuous positions.

    Mỗi ngày t:
      1. Rank signal[t] across tickers: rank 1..n
      2. position_i = (rank_i - (n+1)/2) / ((n-1)/2)
         → top ticker:    position = +1.0  (mua nhiều nhất)
         → median ticker: position =  0.0  (không giao dịch)
         → bottom ticker: position = -1.0  (bán nhiều nhất)
      3. Normalize: pos = pos / sum(|pos|)  → sum(|pos|) = 1
      4. daily_pnl[t] = sum(pos_i * fwd_ret_i)

    Tính chất:
      - Dollar-neutral: sum(pos) = 0
      - Signal mạnh → position lớn, signal yếu → position nhỏ
      - Liên tục, không bị mất thông tin như binary +1/-1
    """
    common_tickers = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_tickers]
    fwd = fwd_ret_df[common_tickers]

    daily_pnl = []
    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        common_t = row_sig.index.intersection(row_fwd.index)
        n = len(common_t)
        if n < 4:
            continue

        # Rank-based continuous positions
        ranks = row_sig[common_t].rank(ascending=True)        # rank 1..n
        pos   = (ranks - (n + 1) / 2) / ((n - 1) / 2 + 1e-9) # center, scale [-1,1]

        abs_sum = pos.abs().sum()
        if abs_sum < 1e-9:
            continue
        pos = pos / abs_sum  # normalize: sum(|pos|) = 1

        pnl = float((pos * row_fwd[common_t]).sum())
        daily_pnl.append(pnl)

    return np.array(daily_pnl)


# ── Sharpe ratio ──────────────────────────────────────────────────────

def compute_sharpe_oos(
    signal_df: pd.DataFrame,
    fwd_ret_df: pd.DataFrame,
    test_ratio: float = None,
) -> float:
    """
    Annualized Sharpe của long-short portfolio trên OOS period.
    Sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252)
    """
    if test_ratio is None:
        test_ratio = DEFAULT_CONFIG.test_ratio

    common_dates = sorted(signal_df.index.intersection(fwd_ret_df.index))
    if len(common_dates) < 60:
        return np.nan

    split_idx  = int(len(common_dates) * (1 - test_ratio))
    test_dates = common_dates[split_idx:]

    arr = _build_daily_pnl(signal_df, fwd_ret_df, test_dates)
    if len(arr) < 20:
        return np.nan

    std = arr.std()
    if std < 1e-9:
        return np.nan
    return float(arr.mean() / std * np.sqrt(252))


# ── Return ────────────────────────────────────────────────────────────

def compute_return_oos(
    signal_df: pd.DataFrame,
    fwd_ret_df: pd.DataFrame,
    test_ratio: float = None,
) -> float:
    """
    Annualized return của long-short portfolio trên OOS period.

    Geometric annualization:
      total_return = prod(1 + daily_pnl) - 1
      ann_return   = (1 + total_return)^(252/n_days) - 1

    Fallback về arithmetic nếu total_return < -0.99 (tránh domain error).
    """
    if test_ratio is None:
        test_ratio = DEFAULT_CONFIG.test_ratio

    common_dates = sorted(signal_df.index.intersection(fwd_ret_df.index))
    if len(common_dates) < 60:
        return np.nan

    split_idx  = int(len(common_dates) * (1 - test_ratio))
    test_dates = common_dates[split_idx:]

    arr = _build_daily_pnl(signal_df, fwd_ret_df, test_dates)
    if len(arr) < 20:
        return np.nan

    n_days = len(arr)

    arr_clipped = np.clip(arr, -0.5, 0.5)
    total_return = float(np.prod(1.0 + arr_clipped) - 1.0)

    if total_return <= -0.99:
        return float(arr.mean() * 252)

    ann_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    return float(ann_return)


# ── Turnover ──────────────────────────────────────────────────────────

def compute_turnover(signal_df: pd.DataFrame) -> float:
    """
    Tốc độ thay đổi signal — proxy cho transaction costs.
    Turnover = mean(|signal[t] - signal[t-1]|) / mean(|signal[t]|)
    """
    diffs = signal_df.diff().abs().mean(axis=1)
    scale = signal_df.abs().mean(axis=1)
    return float((diffs / (scale + 1e-9)).mean())


# ── Legacy single-stock helpers (GP fast path) ───────────────────────

def compute_ic_single(alpha: pd.Series, fwd_ret: pd.Series) -> float:
    c = pd.concat([alpha, fwd_ret], axis=1).dropna()
    if len(c) < 30:
        return np.nan
    if _is_constant_series(c.iloc[:, 0]) or _is_constant_series(c.iloc[:, 1]):
        return np.nan
    return float(c.iloc[:, 0].corr(c.iloc[:, 1], method="spearman"))


def compute_ic_oos_single(
    alpha: pd.Series,
    fwd_ret: pd.Series,
    test_ratio: float = None,
) -> Tuple[float, float]:
    if test_ratio is None:
        test_ratio = DEFAULT_CONFIG.test_ratio
    merged = pd.concat([alpha, fwd_ret], axis=1).dropna()
    n = len(merged)
    if n < 60:
        ic = compute_ic_single(merged.iloc[:, 0], merged.iloc[:, 1])
        return ic, ic
    split = int(n * (1 - test_ratio))
    train, test = merged.iloc[:split], merged.iloc[split:]
    return (compute_ic_single(train.iloc[:, 0], train.iloc[:, 1]),
            compute_ic_single(test.iloc[:, 0],  test.iloc[:, 1]))