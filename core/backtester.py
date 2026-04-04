"""
core/backtester.py
Walk-forward IC evaluation + signal generation.
Key fix vs original: IC computed on OUT-OF-SAMPLE split to avoid data leakage.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ── Walk-forward split ────────────────────────────────────────────────

def train_test_split_time(
    alpha: pd.Series,
    fwd_ret: pd.Series,
    test_ratio: float = 0.3,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Chronological train/test split.
    Returns (alpha_train, alpha_test, ret_train, ret_test).
    """
    merged = pd.concat([alpha, fwd_ret], axis=1).dropna()
    n = len(merged)
    split = int(n * (1 - test_ratio))
    train = merged.iloc[:split]
    test  = merged.iloc[split:]
    return (
        train.iloc[:, 0], test.iloc[:, 0],
        train.iloc[:, 1], test.iloc[:, 1],
    )


# ── Metrics ───────────────────────────────────────────────────────────

def compute_ic(alpha: pd.Series, fwd_ret: pd.Series) -> float:
    c = pd.concat([alpha, fwd_ret], axis=1).dropna()
    if len(c) < 30:
        return np.nan
    return float(c.iloc[:, 0].corr(c.iloc[:, 1], method="spearman"))


def compute_ic_oos(alpha: pd.Series, fwd_ret: pd.Series,
                   test_ratio: float = 0.3) -> Tuple[float, float]:
    """
    Returns (ic_in_sample, ic_out_of_sample).
    ic_oos is the honest metric — no lookahead.

    Fix: gọi train_test_split_time đúng 1 lần, unpack đủ 4 giá trị.
    Trước đây gọi 2 lần riêng biệt — dư thừa và dễ nhầm thứ tự unpack.
    """
    a_train, a_test, r_train, r_test = train_test_split_time(alpha, fwd_ret, test_ratio)
    ic_is  = compute_ic(a_train, r_train)
    ic_oos = compute_ic(a_test,  r_test)
    return ic_is, ic_oos


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


def compute_sharpe_oos(alpha: pd.Series, fwd_ret: pd.Series,
                       test_ratio: float = 0.3) -> float:
    """Sharpe computed on out-of-sample split only."""
    _, a_test, _, r_test = train_test_split_time(alpha, fwd_ret, test_ratio)
    return compute_sharpe(a_test, r_test)


def compute_turnover(alpha: pd.Series) -> float:
    scale = alpha.abs().mean()
    if scale < 1e-9:
        return np.nan
    return float(alpha.diff().abs().mean() / scale)


def composite_score(ic_oos: float | None, sharpe_oos: float | None,
                    ic_is: float | None = None) -> float:
    """
    Scoring based on OOS metrics primarily.
    ic_is used as sanity check: if IS >> OOS, penalise (overfit signal).

    Scale chuẩn hoá:
      IC_OOS   range thực tế [0, 0.10] → chia 0.10 → [0, 1]
      Sharpe   range thực tế [0, 2.0]  → chia 2.0  → [0, 1]
    → Sau normalize, weight 60/40 mới thực sự là 60/40.

    Overfit penalty: nếu IC_IS > 3× IC_OOS → score × 0.5
    """
    _ic = abs(ic_oos)  if ic_oos  is not None and np.isfinite(ic_oos)  else 0.0
    _sh = sharpe_oos   if sharpe_oos is not None and np.isfinite(sharpe_oos) else 0.0

    # Normalize về [0, 1] dựa trên khoảng thực tế của mỗi metric
    IC_MAX     = 0.10   # IC_OOS > 0.10 là xuất sắc, hiếm gặp
    SHARPE_MAX = 2.0    # Sharpe_OOS > 2.0 là xuất sắc, hiếm gặp
    ic_norm = min(_ic / IC_MAX, 1.0)                        # cap tại 1.0
    sh_norm = min(max(_sh, 0.0) / SHARPE_MAX, 1.0)         # chỉ phần dương

    # Overfit penalty
    overfit_penalty = 1.0
    if ic_is is not None and _ic > 1e-4:
        ratio = abs(ic_is) / (_ic + 1e-9)
        if ratio > 3.0:
            overfit_penalty = 0.5

    return round((0.6 * ic_norm + 0.4 * sh_norm) * overfit_penalty, 6)


def rolling_ic_series(alpha: pd.Series, fwd_ret: pd.Series,
                      window: int = 20) -> pd.Series:
    merged = pd.concat([alpha, fwd_ret], axis=1).dropna()
    merged.columns = ["a", "r"]
    return merged["a"].rolling(window).corr(merged["r"]).rename("rolling_ic")


# ── Composite Signal ──────────────────────────────────────────────────

def build_composite_signal(
    alpha_values: pd.DataFrame,
    scores: list[float],
    alpha_ids: list[int],
) -> pd.Series:
    total = sum(scores)
    weights = [s / total for s in scores] if total > 1e-9 else [1.0 / len(scores)] * len(scores)
    signal = pd.Series(0.0, index=alpha_values.index)
    for i, aid in enumerate(alpha_ids):
        col = f"alpha_{aid}"
        if col in alpha_values.columns:
            signal += weights[i] * alpha_values[col].fillna(0.0)
    mu  = signal.rolling(60, min_periods=20).mean()
    std = signal.rolling(60, min_periods=20).std()
    return ((signal - mu) / (std + 1e-9)).rename("composite_signal")


def generate_trade_signals(composite: pd.Series,
                           threshold_std: float = 1.0) -> dict:
    if composite.empty:
        return {"latest_signal": 0.0, "action": "HOLD", "strength": 0}
    latest = float(composite.iloc[-1])
    if np.isnan(latest):
        return {"latest_signal": 0.0, "action": "HOLD", "strength": 0}
    strength = min(100, int(abs(latest) / threshold_std * 50))
    action = "BUY" if latest > threshold_std else ("SELL" if latest < -threshold_std else "HOLD")
    return {"latest_signal": round(latest, 4), "action": action, "strength": strength}


# ── Alpha Decay Detection ─────────────────────────────────────────────

def detect_decay(alpha: pd.Series, fwd_ret: pd.Series,
                 window: int = 20, decay_threshold: float = 0.30) -> dict:
    merged = pd.concat([alpha, fwd_ret], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    merged.columns = ["a", "r"]
    if len(merged) < window * 2:
        return {"decaying": False, "reason": "insufficient data"}
    rolling = merged["a"].rolling(window).corr(merged["r"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rolling) < window * 2:
        return {"decaying": False, "reason": "insufficient rolling data"}
    hist_ic   = float(rolling.iloc[:-window].mean())
    recent_ic = float(rolling.iloc[-window:].mean())
    if abs(hist_ic) < 1e-4:
        return {"decaying": False, "hist_ic": hist_ic, "recent_ic": recent_ic}
    drop_pct = (abs(hist_ic) - abs(recent_ic)) / (abs(hist_ic) + 1e-9)
    decaying = drop_pct > decay_threshold
    return {
        "decaying": decaying,
        "hist_ic":   round(hist_ic, 4),
        "recent_ic": round(recent_ic, 4),
        "drop_pct":  round(drop_pct * 100, 1),
        "reason": f"IC dropped {drop_pct*100:.1f}%" if decaying else "OK",
    }