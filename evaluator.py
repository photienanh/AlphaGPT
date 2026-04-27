"""
evaluator.py
"""
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, Any, Optional

import alpha_operators as op
from backtester import (
    compute_ic_cross_sectional_oos,
    compute_sharpe_oos,
    compute_return_oos,
    compute_turnover,
)
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

IC_SIGNAL_THRESHOLD  = DEFAULT_CONFIG.ic_signal_threshold
SHARPE_MIN_THRESHOLD = DEFAULT_CONFIG.sharpe_min_threshold
RETURN_MIN_THRESHOLD = DEFAULT_CONFIG.return_min_threshold


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


def _is_valid(series: pd.Series, min_valid_ratio: float = 0.5) -> tuple:
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
    result = deepcopy(alpha_def)
    result.update({
        "ic_is": None, "ic_oos": None,
        "sharpe_oos": None, "return_oos": None,
        "turnover": None,
        "status": "EVAL_ERROR", "series": None,
    })

    expr = alpha_def.get("expression", "")
    is_valid, err_msg = validate_expression(expr)
    if not is_valid:
        result["error"] = f"validation: {err_msg}"
        return result

    # ── Full eval: cross-sectional trên toàn universe ─────────────────

    try:
        signal_all = {}
        skip_count   = 0
        for ticker, ticker_df in df_by_ticker.items():
            try:
                signal_by_ticker = _exec_on_ticker(expr, ticker_df)
                if signal_by_ticker is not None:
                    valid, _ = _is_valid(signal_by_ticker)
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

        forward_return = forward_return.copy()
        forward_return.index = pd.to_datetime(forward_return.index)

        # Cross-sectional normalize: mỗi ngày zscore across tickers
        signal_all_normalized = signal_all_df.apply(
            lambda row: (row - row.mean()) / (row.std() + 1e-9),
            axis=1,
        )

        common_dates   = signal_all_normalized.index.intersection(forward_return.index)
        common_tickers = signal_all_normalized.columns.intersection(forward_return.columns)
        log.debug(
            f"[Eval:{alpha_def.get('id','?')}] "
            f"{len(common_tickers)} tickers × {len(common_dates)} dates "
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

    ic_is, ic_oos       = compute_ic_cross_sectional_oos(signal_all_normalized, forward_return)
    sharpe_oos          = compute_sharpe_oos(signal_all_normalized, forward_return)
    ann_return          = compute_return_oos(signal_all_normalized, forward_return)
    turnover            = compute_turnover(signal_all_normalized)

    ic_oos_val = ic_oos     if (ic_oos     is not None and np.isfinite(ic_oos))     else 0.0
    sharpe_val = sharpe_oos if (sharpe_oos is not None and np.isfinite(sharpe_oos)) else None
    return_val = ann_return if (ann_return is not None and np.isfinite(ann_return)) else None

    if ic_oos_val <= 0:
        status      = "WEAK"
        weak_reason = f"IC_OOS={ic_oos_val:+.4f} ≤ 0: signal sai chiều"
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
        status      = "WEAK" if reasons else "OK"
        weak_reason = "; ".join(reasons) if reasons else None

    result.update({
        "ic_is":       _r(ic_is,      6),
        "ic_oos":      _r(ic_oos,     6),
        "sharpe_oos":  _r(sharpe_oos, 4),
        "return_oos":  _r(ann_return, 4),
        "turnover":    _r(turnover,   4),
        "status":      status,
        "signal":      signal_all_normalized,
    })

    if weak_reason is not None:
        result["weak_reason"] = weak_reason
    return result


def _r(val, decimals):
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), decimals)