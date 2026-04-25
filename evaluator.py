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
    compute_ic_oos_single,
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


def _exec_on_ticker(expression: str, df_ticker: pd.DataFrame) -> Optional[pd.Series]:
    ns = _build_namespace(df_ticker)
    exec(expression, ns)
    series = ns.get("alpha")
    if not isinstance(series, pd.Series):
        return None
    series = series.replace([np.inf, -np.inf], np.nan)
    exp_mu  = series.expanding(min_periods=20).mean()
    exp_std = series.expanding(min_periods=20).std()
    return ((series - exp_mu) / (exp_std + 1e-9)).clip(-5, 5)


def eval_alpha(
    alpha_def: Dict[str, Any],
    df_or_ticker_dfs,
    fwd_ret,
    full: bool = True,
) -> Dict[str, Any]:
    result = deepcopy(alpha_def)
    result.update({
        "ic_is": None, "ic_oos": None,
        "sharpe_oos": None, "return_oos": None,
        "turnover": None,
        "status": "EVAL_ERROR", "series": None,
        "gp_enhanced": False,
    })

    expr = alpha_def.get("expression", "")
    is_valid, err_msg = validate_expression(expr)
    if not is_valid:
        result["error"] = f"validation: {err_msg}"
        return result

    # ── GP fast path: single ticker ───────────────────────────────────
    if not full:
        try:
            norm = _exec_on_ticker(expr, df_or_ticker_dfs)
            if norm is None:
                result["error"] = "expression did not produce pd.Series named 'alpha'"
                return result
            valid, reason = _is_valid(norm)
            if not valid:
                result["error"] = reason
                return result
            ic_is, ic_oos = compute_ic_oos_single(norm, fwd_ret)
            result.update({"ic_is": _r(ic_is, 6), "ic_oos": _r(ic_oos, 6),
                           "status": "OK", "series": norm})
        except Exception as e:
            result["error"] = str(e)[:120]
        return result

    # ── Full eval: cross-sectional trên toàn universe ─────────────────
    ticker_dfs    = df_or_ticker_dfs
    fwd_ret_multi = fwd_ret

    try:
        signal_parts = {}
        skip_count   = 0
        for ticker, df_t in ticker_dfs.items():
            try:
                norm = _exec_on_ticker(expr, df_t)
                if norm is not None:
                    valid, _ = _is_valid(norm)
                    if valid:
                        signal_parts[ticker] = norm
                    else:
                        skip_count += 1
            except Exception:
                skip_count += 1

        if len(signal_parts) < 3:
            result["error"] = (
                f"chỉ có {len(signal_parts)} tickers có signal hợp lệ "
                f"(bỏ qua {skip_count})"
            )
            return result

        signal_df = pd.DataFrame(signal_parts)
        signal_df.index = pd.to_datetime(signal_df.index)
        signal_df.index.name = "date"

        fwd_ret_multi = fwd_ret_multi.copy()
        fwd_ret_multi.index = pd.to_datetime(fwd_ret_multi.index)

        # Cross-sectional normalize: mỗi ngày zscore across tickers
        signal_norm = signal_df.apply(
            lambda row: (row - row.mean()) / (row.std() + 1e-9),
            axis=1,
        )

        common_dates   = signal_norm.index.intersection(fwd_ret_multi.index)
        common_tickers = signal_norm.columns.intersection(fwd_ret_multi.columns)
        log.debug(
            f"[Eval:{alpha_def.get('id','?')}] "
            f"{len(common_tickers)} tickers × {len(common_dates)} dates "
            f"(valid={len(signal_parts)}, skipped={skip_count})"
        )

        n_universe = len(signal_norm.columns)
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

    ic_is, ic_oos       = compute_ic_cross_sectional_oos(signal_norm, fwd_ret_multi)
    sharpe_oos          = compute_sharpe_oos(signal_norm, fwd_ret_multi)
    ann_return          = compute_return_oos(signal_norm, fwd_ret_multi)
    turnover            = compute_turnover(signal_norm)

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
        "series":      signal_norm,
    })

    if weak_reason is not None:
        result["weak_reason"] = weak_reason
    return result


def _r(val, decimals):
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), decimals)