"""
core/backtester.py — Alpha evaluation & signal generation
"""
import numpy as np
import pandas as pd
from typing import Optional


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_ic(alpha: pd.Series, fwd_ret: pd.Series) -> float:
    c = pd.concat([alpha, fwd_ret], axis=1).dropna()
    if len(c) < 30:
        return np.nan
    return float(c.iloc[:, 0].corr(c.iloc[:, 1], method="spearman"))


def compute_icir(alpha: pd.Series, fwd_ret: pd.Series, window: int = 20) -> float:
    """Rolling IC IR = mean(IC_rolling) / std(IC_rolling)"""
    merged = pd.concat([alpha, fwd_ret], axis=1).dropna()
    merged.columns = ["a", "r"]
    if len(merged) < window + 10:
        return np.nan
    rolling_ic = merged["a"].rolling(window).corr(merged["r"])
    ic_mean = rolling_ic.mean()
    ic_std  = rolling_ic.std()
    return float(ic_mean / (ic_std + 1e-9))


def compute_sharpe(alpha: pd.Series, fwd_ret: pd.Series) -> float:
    c = pd.concat([alpha, fwd_ret], axis=1).dropna()
    c.columns = ["a", "r"]
    if len(c) < 30:
        return np.nan
    pos = np.where(c["a"] > c["a"].median(), 1.0, -1.0)
    pnl = pos * c["r"]
    if pnl.std() < 1e-9:
        return np.nan
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def compute_turnover(alpha: pd.Series) -> float:
    scale = alpha.abs().mean()
    if scale < 1e-9:
        return np.nan
    return float(alpha.diff().abs().mean() / scale)


def compute_max_drawdown(alpha: pd.Series, fwd_ret: pd.Series) -> float:
    c = pd.concat([alpha, fwd_ret], axis=1).dropna()
    c.columns = ["a", "r"]
    pos = np.where(c["a"] > c["a"].median(), 1.0, -1.0)
    pnl = (pos * c["r"]).cumsum()
    rolling_max = pnl.cummax()
    drawdown = pnl - rolling_max
    return float(drawdown.min())


def composite_score(ic: Optional[float], sharpe: Optional[float]) -> float:
    _ic = abs(ic) if ic is not None and not np.isnan(ic) else 0.0
    _sh = sharpe if sharpe is not None and not np.isnan(sharpe) else 0.0
    return round(0.6 * _ic + 0.4 * max(_sh / 5.0, 0.0), 6)


def rolling_ic_series(alpha: pd.Series, fwd_ret: pd.Series, window: int = 20) -> pd.Series:
    """Return rolling IC series for chart display."""
    merged = pd.concat([alpha, fwd_ret], axis=1).dropna()
    merged.columns = ["a", "r"]
    return merged["a"].rolling(window).corr(merged["r"]).rename("rolling_ic")


# ─── Composite Signal ────────────────────────────────────────────────────────

def build_composite_signal(
    alpha_values: pd.DataFrame,
    scores: list[float],
    alpha_ids: list[int],
) -> pd.Series:
    """
    Weighted average of alpha columns by score.
    Returns z-score normalized composite signal.
    """
    total_score = sum(scores)
    if total_score < 1e-9:
        weights = [1.0 / len(scores)] * len(scores)
    else:
        weights = [s / total_score for s in scores]

    signal = pd.Series(0.0, index=alpha_values.index)
    for i, aid in enumerate(alpha_ids):
        col = f"alpha_{aid}"
        if col in alpha_values.columns:
            a = alpha_values[col].fillna(0.0)
            signal += weights[i] * a

    # Z-score normalize
    mu  = signal.rolling(60, min_periods=20).mean()
    std = signal.rolling(60, min_periods=20).std()
    return ((signal - mu) / (std + 1e-9)).rename("composite_signal")


def generate_trade_signals(
    composite: pd.Series,
    top_n: int = 5,
    threshold_std: float = 1.0,
) -> dict:
    """
    Given composite signal for ONE ticker, return:
      - latest_signal: float
      - action: BUY / SELL / HOLD
      - strength: 0-100
    """
    if composite.empty:
        return {"latest_signal": 0.0, "action": "HOLD", "strength": 0}

    latest = float(composite.iloc[-1])
    if np.isnan(latest):
        return {"latest_signal": 0.0, "action": "HOLD", "strength": 0}

    strength = min(100, int(abs(latest) / threshold_std * 50))

    if latest > threshold_std:
        action = "BUY"
    elif latest < -threshold_std:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "latest_signal": round(latest, 4),
        "action": action,
        "strength": strength,
    }


# ─── Alpha Decay Detection ───────────────────────────────────────────────────

def detect_decay(
    alpha: pd.Series,
    fwd_ret: pd.Series,
    window: int = 20,
    decay_threshold: float = 0.30,
) -> dict:
    """
    Compare recent IC vs historical IC.
    Returns decay info dict.
    """
    merged = pd.concat([alpha, fwd_ret], axis=1)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
    merged.columns = ["a", "r"]
    if len(merged) < window * 2:
        return {"decaying": False, "reason": "insufficient data"}

    rolling = merged["a"].rolling(window).corr(merged["r"])
    rolling = rolling.replace([np.inf, -np.inf], np.nan).dropna()
    if len(rolling) < window * 2:
        return {"decaying": False, "reason": "insufficient rolling data"}

    hist_part = rolling.iloc[:-window]
    recent_part = rolling.iloc[-window:]
    if hist_part.empty or recent_part.empty:
        return {"decaying": False, "reason": "insufficient rolling split"}

    hist_ic = float(hist_part.mean())
    recent_ic = float(recent_part.mean())

    if abs(hist_ic) < 1e-4:
        return {"decaying": False, "hist_ic": hist_ic, "recent_ic": recent_ic}

    drop_pct = (abs(hist_ic) - abs(recent_ic)) / (abs(hist_ic) + 1e-9)
    decaying = drop_pct > decay_threshold

    return {
        "decaying": decaying,
        "hist_ic": round(hist_ic, 4),
        "recent_ic": round(recent_ic, 4),
        "drop_pct": round(drop_pct * 100, 1),
        "reason": f"IC dropped {drop_pct*100:.1f}% (threshold {decay_threshold*100:.0f}%)" if decaying else "OK",
    }
