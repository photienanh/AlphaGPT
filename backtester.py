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
from copy import deepcopy
from typing import Dict, Any
import logging

import alpha_operators as op
from validators import validate_expression
from config import DEFAULT_CONFIG

log = logging.getLogger(__name__)

DATA_FIELDS = list({
    "open", "high", "low", "close", "volume",
    "vwap", "adv20", "returns",
    "sma_5", "sma_20", "ema_10",
    "momentum_3", "momentum_10",
    "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_middle", "bb_lower",
    "obv",
})

IC_SIGNAL_THRESHOLD = DEFAULT_CONFIG.ic_signal_threshold
SHARPE_MIN_THRESHOLD = DEFAULT_CONFIG.sharpe_min_threshold
RETURN_MIN_THRESHOLD = DEFAULT_CONFIG.return_min_threshold


def _is_constant_series(s: pd.Series) -> bool:
    """True nếu series không có đủ biến thiên để tính correlation."""
    if s is None:
        return True
    x = s.dropna()
    if len(x) < 2:
        return True
    return x.nunique() < 2


# ── Cross-sectional IC ────────────────────────────────────────────────

def compute_ic(
    signal_all: pd.DataFrame,
    forward_return: pd.DataFrame,
) -> float:
    """
    Tính Spearman IC cross-sectional.
    Tại mỗi ngày t: corr(signal[t, :], fwd_ret[t, :]) across tickers.
    Returns: (mean_ic, ic_ir, ic_series)
      mean_ic:   E[IC_t]
    """
    common_dates   = signal_all.index.intersection(forward_return.index)
    common_tickers = signal_all.columns.intersection(forward_return.columns)

    sig = signal_all.loc[common_dates, common_tickers]
    fwd = forward_return.loc[common_dates, common_tickers]

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
        return float("nan")

    ic_series = pd.Series({d: v for d, v in ic_list}, name="ic")
    mean_ic   = float(ic_series.mean())

    return mean_ic


def _build_namespace(df: pd.DataFrame, industry=None) -> dict:
    ns = {name: getattr(op, name) for name in dir(op) if not name.startswith("_")}
    ns.update({"df": df, "np": np, "pd": pd})
    for col in DATA_FIELDS:
        col_lower = col.lower()
        if col_lower in df.columns:
            ns[col_lower] = df[col_lower]
            ns[col] = df[col_lower]
        elif col in df.columns:
            ns[col] = df[col]
    if industry is not None:
        ns["industry"] = industry
    return ns


def _is_valid_signal(series: pd.Series, min_valid_ratio: float = 0.5) -> tuple:
    if series is None:
        return False, "None"
    n = len(series)
    if n == 0:
        return False, "empty"
    n_valid = series.dropna().shape[0]
    if n_valid / n < min_valid_ratio:
        return False, f"too many NaN ({n - n_valid}/{n})"
    s = series.dropna()
    if s.std() < 1e-9:
        return False, "constant"
    if (s == 0).mean() > 0.65:
        return False, "too sparse"
    return True, "OK"


def _exec_on_ticker(expression: str, ticker_df: pd.DataFrame) -> Optional[pd.Series]:
    ns = _build_namespace(ticker_df)
    exec(expression, ns)
    signal = ns.get("alpha")
    if not isinstance(signal, pd.Series):
        return None
    signal = signal.replace([np.inf, -np.inf], np.nan)
    return signal


def eval_alpha(
    alpha_def: Dict[str, Any],
    df_by_ticker: Dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Full cross-sectional evaluation trên toàn universe.
    IC_OOS được tính trên 30% ngày cuối.
    IC_IS lấy từ GP fitness (alpha_def['ic_is']).
    """
    result = deepcopy(alpha_def)
    result.update({
        "ic_is": alpha_def.get("ic_is"),
        "ic_oos": None,
        "sharpe_oos": None,
        "return_oos": None,
        "turnover": None,
        "status": "EVAL_ERROR",
        "series": None,
    })

    expr = alpha_def.get("expression", "")
    is_valid, err_msg = validate_expression(expr)
    if not is_valid:
        result["error"] = f"validation: {err_msg}"
        return result

    try:
        signal_all = {}
        skip_count = 0
        for ticker, ticker_df in df_by_ticker.items():
            try:
                signal_by_ticker = _exec_on_ticker(expr, ticker_df)
                if signal_by_ticker is not None:
                    valid, _ = _is_valid_signal(signal_by_ticker)
                    if valid:
                        signal_all[ticker] = signal_by_ticker
                    else:
                        skip_count += 1
            except Exception:
                skip_count += 1

        if len(signal_all) < 3:
            result["error"] = (
                f"chỉ có {len(signal_all)} tickers có signal hợp lệ "
                f"(bỏ qua {skip_count})"
            )
            return result

        signal_all_df = pd.DataFrame(signal_all)
        signal_all_df.index = pd.to_datetime(signal_all_df.index)
        signal_all_df.index.name = "date"

        fwd = forward_return.copy()
        fwd.index = pd.to_datetime(fwd.index)

        signal_all_normalized = signal_all_df.apply(
            lambda row: (row - row.mean()) / (row.std() + 1e-9),
            axis=1,
        )

        common_dates = signal_all_normalized.index.intersection(fwd.index)
        common_tickers = signal_all_normalized.columns.intersection(fwd.columns)
        log.debug(
            f"[Eval:{alpha_def.get('id','?')}] "
            f"{len(common_tickers)} tickers x {len(common_dates)} dates "
            f"(valid={len(signal_all)}, skipped={skip_count})"
        )

        n_universe = len(signal_all_normalized.columns)
        min_tickers_eval = max(10, int(n_universe * 0.30))

        if len(common_tickers) < min_tickers_eval or len(common_dates) < 60:
            result["error"] = (
                f"không đủ overlap: {len(common_tickers)} tickers "
                f"(cần {min_tickers_eval}), {len(common_dates)} dates"
            )
            return result

    except Exception as e:
        result["error"] = str(e)[:120]
        return result

    test_ratio = DEFAULT_CONFIG.test_ratio
    common_dates_sorted = sorted(signal_all_normalized.index.intersection(fwd.index))
    split_idx = int(len(common_dates_sorted) * (1 - test_ratio))
    test_dates = common_dates_sorted[split_idx:]

    ic_oos = compute_ic(
        signal_all_normalized.loc[test_dates],
        fwd.loc[test_dates],
    )
    sharpe_oos = compute_sharpe_oos(signal_all_normalized, fwd)
    ann_return = compute_return_oos(signal_all_normalized, fwd)
    turnover = compute_turnover(signal_all_normalized)

    ic_oos_val = ic_oos if (ic_oos is not None and np.isfinite(ic_oos)) else 0.0
    sharpe_val = sharpe_oos if (sharpe_oos is not None and np.isfinite(sharpe_oos)) else None
    return_val = ann_return if (ann_return is not None and np.isfinite(ann_return)) else None

    if ic_oos_val <= 0:
        status = "WEAK"
        weak_reason = f"IC_OOS={ic_oos_val:+.4f} <= 0: signal sai chiều"
    else:
        reasons = []
        if ic_oos_val < IC_SIGNAL_THRESHOLD:
            reasons.append(f"IC_OOS={ic_oos_val:+.4f} < {IC_SIGNAL_THRESHOLD} (IC dương nhưng yếu)")
        if sharpe_val is None or sharpe_val <= SHARPE_MIN_THRESHOLD:
            sv = 0.0 if sharpe_val is None else sharpe_val
            reasons.append(f"Sharpe_OOS={sv:+.4f} <= {SHARPE_MIN_THRESHOLD}")
        if return_val is None or return_val <= RETURN_MIN_THRESHOLD:
            rv = 0.0 if return_val is None else return_val
            reasons.append(f"Return_OOS={rv:+.4f} <= {RETURN_MIN_THRESHOLD}")
        status = "WEAK" if reasons else "OK"
        weak_reason = "; ".join(reasons) if reasons else None

    result.update({
        "ic_is": _r(result.get("ic_is"), 6),
        "ic_oos": _r(ic_oos, 6),
        "sharpe_oos": _r(sharpe_oos, 4),
        "return_oos": _r(ann_return, 4),
        "turnover": _r(turnover, 4),
        "status": status,
        "signal": signal_all_normalized,
    })

    if weak_reason is not None:
        result["weak_reason"] = weak_reason
    return result


def _r(val, decimals):
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), decimals)


# ── Portfolio daily PnL helper ────────────────────────────────────────

def _build_daily_pnl(
    signal_all: pd.DataFrame,
    forward_return: pd.DataFrame,
    test_dates: list,
    cost_per_turnover: float = 0.0015,
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
    common_tickers = signal_all.columns.intersection(forward_return.columns)
    sig = signal_all[common_tickers]
    fwd = forward_return[common_tickers]

    daily_pnl = []
    prev_pos = None

    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        common_t = row_sig.index.intersection(row_fwd.index)
        n = len(common_t)
        if n < 4:
            continue

        ranks = row_sig[common_t].rank(ascending=True)
        pos   = (ranks - (n + 1) / 2) / ((n - 1) / 2 + 1e-9)
        abs_sum = pos.abs().sum()
        if abs_sum < 1e-9:
            continue
        pos = pos / abs_sum

        gross_pnl = float((pos * row_fwd[common_t]).sum())

        # Turnover = sum(|pos[t] - pos[t-1]|) / 2
        if prev_pos is not None:
            prev_aligned = prev_pos.reindex(common_t).fillna(0)
            turnover = float((pos - prev_aligned).abs().sum()) / 2
        else:
            turnover = float(pos.abs().sum()) / 2

        net_pnl = gross_pnl - cost_per_turnover * turnover
        daily_pnl.append(net_pnl)
        prev_pos = pos

    return np.array(daily_pnl)


# ── Sharpe ratio ──────────────────────────────────────────────────────

def compute_sharpe_oos(
    signal_all: pd.DataFrame,
    forward_return: pd.DataFrame,
    cost_per_turnover: float = 0.0015,
    test_ratio: float = None,
) -> float:
    """
    Sharpe sau khi trừ transaction cost ước tính.
    cost_per_turnover: chi phí mỗi đơn vị turnover (mặc định 0.15% = 15bps mỗi chiều).
    Công thức: net_pnl[t] = gross_pnl[t] - cost_per_turnover * turnover[t]
    """
    if test_ratio is None:
        test_ratio = DEFAULT_CONFIG.test_ratio

    common_dates = sorted(signal_all.index.intersection(forward_return.index))
    if len(common_dates) < 60:
        return np.nan

    split_idx  = int(len(common_dates) * (1 - test_ratio))
    test_dates = common_dates[split_idx:]

    common_tickers = signal_all.columns.intersection(forward_return.columns)
    sig = signal_all[common_tickers]
    fwd = forward_return[common_tickers]

    daily_net_pnl = _build_daily_pnl(sig, fwd, test_dates, cost_per_turnover)

    arr = np.array(daily_net_pnl)
    if len(arr) < 20:
        return np.nan
    std = arr.std()
    if std < 1e-9:
        return np.nan
    return float(arr.mean() / std * np.sqrt(252))

# ── Return ────────────────────────────────────────────────────────────

def compute_return_oos(
    signal_all: pd.DataFrame,
    forward_return: pd.DataFrame,
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

    common_dates = sorted(signal_all.index.intersection(forward_return.index))
    if len(common_dates) < 60:
        return np.nan

    split_idx  = int(len(common_dates) * (1 - test_ratio))
    test_dates = common_dates[split_idx:]

    arr = _build_daily_pnl(signal_all, forward_return, test_dates)
    if len(arr) < 20:
        return np.nan

    n_days = len(arr)

    total_return = float(np.prod(1.0 + arr) - 1.0)

    if total_return <= -0.99:
        return float(arr.mean() * 252)

    ann_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    return float(ann_return)


# ── Turnover ──────────────────────────────────────────────────────────

def compute_turnover(signal_al: pd.DataFrame) -> float:
    """
    Tốc độ thay đổi signal — proxy cho transaction costs.
    Turnover = mean(|signal[t] - signal[t-1]|) / mean(|signal[t]|)
    """
    diffs = signal_al.diff().abs().mean(axis=1)
    scale = signal_al.abs().mean(axis=1)
    return float((diffs / (scale + 1e-9)).mean())
