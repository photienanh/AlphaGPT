import numpy as np
import pandas as pd
from typing import Optional

def shift(series: pd.Series, period: int) -> pd.Series:
    return series.shift(period)

def ts_corr(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    return s1.rolling(window).corr(s2)

def ts_cov(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    return s1.rolling(window).cov(s2)

def ts_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def ts_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()

def ts_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).sum()

def ts_product(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(np.prod, raw=True)

def ts_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).min()

def ts_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).max()

def ts_argmax(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(np.argmax, raw=True)

def ts_argmin(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(np.argmin, raw=True)

def ts_argmaxmin_diff(series: pd.Series, window: int) -> pd.Series:
    return ts_argmax(series, window) - ts_argmin(series, window)

def ts_max_diff(series: pd.Series, window: int) -> pd.Series:
    return series - ts_max(series, window)

def ts_min_diff(series: pd.Series, window: int) -> pd.Series:
    return series - ts_min(series, window)

def ts_median(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).median()

def ts_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

def ts_zscore_scale(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mu) / (std + 1e-9)

def ts_maxmin_scale(series: pd.Series, window: int) -> pd.Series:
    mn = series.rolling(window).min()
    mx = series.rolling(window).max()
    return (series - mn) / (mx - mn + 1e-9)

def ts_skew(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).skew()

def ts_kurt(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).kurt()

def ts_delta(series: pd.Series, period: int) -> pd.Series:
    return series.diff(period)

def ts_delta_ratio(series: pd.Series, period: int) -> pd.Series:
    prev = series.shift(period)
    return (series - prev) / (prev.abs() + 1e-9)

def ts_ir(series: pd.Series, window: int) -> pd.Series:
    ret = series.pct_change()
    return ret.rolling(window).mean() / (ret.rolling(window).std() + 1e-9)

def ts_decayed_linear(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return series.rolling(window).apply(
        lambda x: (x * weights[-len(x):]).sum(), raw=True
    )

def ts_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def ts_percentile(series: pd.Series, window: int, pct: float = 0.5) -> pd.Series:
    return series.rolling(window).quantile(pct)

def ts_linear_reg(series: pd.Series, window: int) -> pd.Series:
    def _slope(x):
        if len(x) < 2:
            return np.nan
        t = np.arange(len(x), dtype=float)
        slope, _ = np.polyfit(t, x, 1)
        return slope
    return series.rolling(window).apply(_slope, raw=True)

# ── Cross-sectional ─────────────────────────────────────────────────

def zscore_scale(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    if window:
        mu  = series.rolling(window).mean()
        std = series.rolling(window).std()
    else:
        mu  = series.expanding().mean()
        std = series.expanding().std()
    return (series - mu) / (std + 1e-9)

def winsorize_scale(series: pd.Series, limits: float = 0.05) -> pd.Series:
    lo = series.quantile(limits)
    hi = series.quantile(1 - limits)
    clipped = series.clip(lower=lo, upper=hi)
    return 2 * (clipped - lo) / (hi - lo + 1e-9) - 1

def normed_rank(series: pd.Series) -> pd.Series:
    return series.expanding().rank(pct=True)

def cwise_max(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return pd.concat([s1, s2], axis=1).max(axis=1)

def cwise_min(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return pd.concat([s1, s2], axis=1).min(axis=1)

# ── Group-wise ──────────────────────────────────────────────────────
# LƯU Ý: grouped_* chỉ nhận pd.Series, KHÔNG nhận DataFrame

def grouped_mean(series: pd.Series, window: int) -> pd.Series:
    assert isinstance(series, pd.Series), "grouped_mean chỉ nhận pd.Series"
    return series.rolling(window).mean()

def grouped_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()

def grouped_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).max()

def grouped_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).min()

def grouped_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).sum()

def grouped_demean(series: pd.Series, window: int) -> pd.Series:
    return series - series.rolling(window).mean()

def grouped_zscore_scale(series: pd.Series, window: int) -> pd.Series:
    return ts_zscore_scale(series, window)

def grouped_winsorize_scale(series: pd.Series, window: int,
                            limits: float = 0.05) -> pd.Series:
    def _wins(x):
        lo = np.percentile(x, limits * 100)
        hi = np.percentile(x, (1 - limits) * 100)
        clipped = np.clip(x[-1], lo, hi)
        return 2 * (clipped - lo) / (hi - lo + 1e-9) - 1
    return series.rolling(window).apply(_wins, raw=True)

# ── Element-wise ────────────────────────────────────────────────────

def relu(series: pd.Series) -> pd.Series:
    return series.clip(lower=0)

def neg(series: pd.Series) -> pd.Series:
    return -series

def abso(series: pd.Series) -> pd.Series:
    return series.abs()

def log(series: pd.Series) -> pd.Series:
    out = np.log(series.clip(lower=1e-9))
    return pd.Series(out, index=series.index, name=series.name)

def log1p(series: pd.Series) -> pd.Series:
    out = np.log1p(series.clip(lower=-0.9999))
    return pd.Series(out, index=series.index, name=series.name)

def sign(series: pd.Series) -> pd.Series:
    out = np.sign(series)
    return pd.Series(out, index=series.index, name=series.name)

def pow_op(series: pd.Series, exp: float) -> pd.Series:
    return series.pow(exp)

def pow_sign(series: pd.Series, exp: float) -> pd.Series:
    out = np.sign(series) * series.abs().pow(exp)
    return pd.Series(out, index=series.index, name=series.name)

def round_op(series: pd.Series, decimals: int = 2) -> pd.Series:
    return series.round(decimals)

def add(s1, s2) -> pd.Series:
    return s1 + s2

def minus(s1, s2) -> pd.Series:
    return s1 - s2

def div(s1, s2) -> pd.Series:
    """Chia 2 series hoặc scalar/series. Tránh chia cho 0."""
    if isinstance(s2, pd.Series):
        denom = s2.replace(0, np.nan).fillna(1e-9)
    else:
        denom = s2 if abs(s2) > 1e-9 else 1e-9
    return s1 / denom

def greater(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return (s1 > s2).astype(float)

def less(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return (s1 < s2).astype(float)

def cwise_mul(s1, s2) -> pd.Series:
    return s1 * s2

def normed_rank_diff(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return normed_rank(s1) - normed_rank(s2)

def tanh(series: pd.Series) -> pd.Series:
    out = np.tanh(series)
    return pd.Series(out, index=series.index, name=series.name)

def clip(series: pd.Series, lower=None, upper=None) -> pd.Series:
    return series.clip(lower=lower, upper=upper)