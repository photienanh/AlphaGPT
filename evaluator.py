"""
evaluator.py
Eval một alpha expression string dùng alpha_operators.
Dùng cho cả GP fitness (nhanh) và full review (đầy đủ).
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, Any

import alpha_operators as op
from backtester import (
    compute_ic, compute_ic_oos,
    compute_sharpe_oos, compute_return_oos,
    compute_turnover
)

DATA_FIELDS = [
    "open", "high", "low", "close", "volume",
    "SMA_5", "SMA_20", "EMA_10",
    "RSI_14", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "OBV", "Momentum_3", "Momentum_10",
]

IC_SIGNAL_THRESHOLD = 0.01  # dưới này là noise với daily VN30 data


def _build_namespace(df: pd.DataFrame) -> dict:
    """Tạo namespace cho exec() với tất cả operators và data fields."""
    ns = {name: getattr(op, name)
          for name in dir(op) if not name.startswith("_")}
    ns.update({"df": df, "np": np, "pd": pd})
    # Shorthand: close, volume, ... truy cập trực tiếp
    for col in DATA_FIELDS:
        if col in df.columns:
            ns[col] = df[col]
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


def eval_alpha(alpha_def: Dict[str, Any],
               df: pd.DataFrame,
               fwd_ret: pd.Series,
               full: bool = True) -> Dict[str, Any]:
    """
    Eval một alpha definition dict.

    full=False → chỉ tính IC_IS (nhanh, dùng cho GP fitness).
    full=True  → tính đầy đủ IC_IS/OOS, Sharpe_OOS, turnove.
    """
    result = deepcopy(alpha_def)
    result.update({
        "ic_is": None, "ic_oos": None,
        "sharpe_oos": None,
        "return_oos": None,
        "mdd": None,          # max drawdown
        "turnover": None,
        "status": "EVAL_ERROR",
        "series": None,
        "gp_enhanced": False,
    })

    # ── Thực thi expression ──────────────────────────────────────────
    try:
        ns = _build_namespace(df)
        exec(alpha_def["expression"], ns)
        series = ns.get("alpha")
        if not isinstance(series, pd.Series):
            result["error"] = "expression did not produce pd.Series named 'alpha'"
            return result
        series = series.replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        result["error"] = str(e)[:120]
        return result

    valid, reason = _is_valid(series)
    if not valid:
        result["error"] = reason
        return result

    # ── Normalize (expanding, không dùng future data) ────────────────
    exp_mu  = series.expanding(min_periods=20).mean()
    exp_std = series.expanding(min_periods=20).std()
    norm = ((series - exp_mu) / (exp_std + 1e-9)).clip(-5, 5)

    # ── Fast path: chỉ IC_IS cho GP ─────────────────────────────────
    if not full:
        ic_is = compute_ic(norm, fwd_ret)
        result.update({
            "ic_is": round(ic_is, 6) if not np.isnan(ic_is) else None,
            "status": "OK",
            "series": norm,
        })
        return result

    # ── Full eval ────────────────────────────────────────────────────
    # Bỏ warm-up NaN đầu kỳ
    mask = norm.notna() & fwd_ret.notna()
    if not mask.any():
        result["error"] = "no valid overlap after warm-up"
        return result
    start = mask[mask].index[0]
    norm_eval = norm.loc[start:]
    fwd_eval  = fwd_ret.loc[start:]

    ic_is, ic_oos = compute_ic_oos(norm_eval, fwd_eval)

    ic_oos_val = ic_oos if not np.isnan(ic_oos) else 0.0

    if ic_oos_val <= 0:
        status = "WEAK"
        weak_reason = f"IC_OOS={ic_oos_val:+.4f} ≤ 0: signal sai chiều"
    elif ic_oos_val < IC_SIGNAL_THRESHOLD:
        status = "MARGINAL"
        weak_reason = (
            f"IC_OOS={ic_oos_val:+.4f} < {IC_SIGNAL_THRESHOLD}: "
            f"dương nhưng trong vùng noise thống kê"
        )
    else:
        status = "OK"
        weak_reason = None
 
    sharpe_oos          = compute_sharpe_oos(norm_eval, fwd_eval)
    ann_return, mdd     = compute_return_oos(norm_eval, fwd_eval)
    turnover            = compute_turnover(norm_eval)
    result.update({
        "ic_is":      _r(ic_is,      6),
        "ic_oos":     _r(ic_oos,     6),
        "sharpe_oos": _r(sharpe_oos, 4),
        "return_oos": _r(ann_return, 4),   # annualized return
        "mdd":        _r(mdd,        4),   # max drawdown
        "turnover":   _r(turnover,   4),
        "status":     status,
        "series":     norm_eval,
    })
    if weak_reason is not None:
        result["weak_reason"] = weak_reason
    return result

def _r(val, decimals):
    """Round nếu finite, None nếu không."""
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), decimals)