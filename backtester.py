"""
backtester.py
Walk-forward IC/Sharpe evaluation — dùng cho cả GP fitness và full review.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ── Helpers ───────────────────────────────────────────────────────────
 
def _train_test_split(alpha: pd.Series, fwd_ret: pd.Series,
                      test_ratio: float = 0.3):
    """Chronological split. Trả về (a_train, a_test, r_train, r_test)."""
    merged = pd.concat([alpha, fwd_ret], axis=1).dropna()
    n = len(merged)
    split = int(n * (1 - test_ratio))
    train = merged.iloc[:split]
    test  = merged.iloc[split:]
    return (train.iloc[:, 0], test.iloc[:, 0],
            train.iloc[:, 1], test.iloc[:, 1])


# ── IC ────────────────────────────────────────────────────────────────
 
def compute_ic(alpha: pd.Series, fwd_ret: pd.Series) -> float:
    """Spearman IC trên toàn bộ data được truyền vào."""
    c = pd.concat([alpha, fwd_ret], axis=1).dropna()
    if len(c) < 30:
        return np.nan
    return float(c.iloc[:, 0].corr(c.iloc[:, 1], method="spearman"))

def compute_ic_oos(alpha: pd.Series, fwd_ret: pd.Series,
                   test_ratio: float = 0.3) -> Tuple[float, float]:
    """
    Walk-forward IC.
    Returns (ic_is, ic_oos).
    """
    merged = pd.concat([alpha, fwd_ret], axis=1).dropna()
    n = len(merged)
    if n < 60:
        ic = compute_ic(merged.iloc[:, 0], merged.iloc[:, 1])
        return ic, ic
    split = int(n * (1 - test_ratio))
    train = merged.iloc[:split]
    test  = merged.iloc[split:]
    return (compute_ic(train.iloc[:, 0], train.iloc[:, 1]),
            compute_ic(test.iloc[:, 0],  test.iloc[:, 1]))


# ── Sharpe ratio ──────────────────────────────────────────────────────

def compute_sharpe_oos(alpha: pd.Series, fwd_ret: pd.Series,
                       test_ratio: float = 0.3) -> float:
    merged = pd.concat([alpha, fwd_ret], axis=1).dropna()
    n = len(merged)
    if n < 60:
        return np.nan
    split = int(n * (1 - test_ratio))
    test = merged.iloc[split:]
    a, r = test.iloc[:, 0], test.iloc[:, 1]
    pos = np.where(a > a.median(), 1.0, -1.0)
    pnl = pos * r
    if pnl.std() < 1e-9:
        return np.nan
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


# ── Backtest return ───────────────────────────────────────────────────
 
def compute_return_oos(alpha: pd.Series, fwd_ret: pd.Series,
                       test_ratio: float = 0.3) -> Tuple[float, float]:
    """
    Annualized return và Max Drawdown trên OOS period.
    Đây là metric thứ ba trong paper Section 2.3: "backtest returns".
 
    Cách tính:
    1. Tạo position: long (+1) khi signal > median, short (-1) khi <= median
    2. Daily P&L = position × forward_return
    3. Annualized return = mean(daily_pnl) × 252
    4. Max drawdown = max peak-to-trough trên cumulative P&L
 
    Returns: (annualized_return, max_drawdown)
      annualized_return: float, ví dụ 0.12 = 12%/năm
      max_drawdown: float dương, ví dụ 0.05 = drawdown tối đa 5%
    """
    merged = pd.concat([alpha, fwd_ret], axis=1).dropna()
    n = len(merged)
    if n < 60:
        return np.nan, np.nan
 
    split = int(n * (1 - test_ratio))
    test  = merged.iloc[split:]
    a, r  = test.iloc[:, 0], test.iloc[:, 1]
 
    # Long/short position dựa trên median của OOS period
    pos     = np.where(a > a.median(), 1.0, -1.0)
    daily   = pos * r.values
 
    # Annualized return: giả sử 252 trading days/năm
    ann_ret = float(np.mean(daily) * 252)
 
    # Max Drawdown: tính trên cumulative P&L
    cum = np.cumsum(daily)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    mdd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
 
    return ann_ret, mdd
 
 
# ── Turnover ──────────────────────────────────────────────────────────
 
def compute_turnover(alpha: pd.Series) -> float:
    """
    Tốc độ thay đổi position — proxy cho transaction costs.
    Turnover cao → chi phí giao dịch lớn → return thực tế thấp hơn.
    """
    scale = alpha.abs().mean()
    if scale < 1e-9:
        return np.nan
    return float(alpha.diff().abs().mean() / scale)