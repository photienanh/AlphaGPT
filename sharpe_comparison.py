"""
sharpe_comparison.py
So sánh 5 cách tính Sharpe trên alpha từ alpha_library.json.

Cách chạy:
    python sharpe_comparison.py
    python sharpe_comparison.py --library path/to/alpha_library.json
    python sharpe_comparison.py --data-dir ./data --cost 0.002
"""

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Thêm project root vào sys.path để import được các module ──────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR  # điều chỉnh nếu script nằm ngoài project root
sys.path.insert(0, PROJECT_DIR)

# ─────────────────────────────────────────────────────────────────────
# PHẦN 1 — Load data
# ─────────────────────────────────────────────────────────────────────

def _load_data(data_dir: str):
    """Load ticker data từ project, dùng make_sample_data_multi nếu không có."""
    try:
        from data_loader import load_multi_stock, make_sample_data_multi
        if os.path.isdir(data_dir) and os.listdir(data_dir):
            _, ticker_dfs, fwd_ret_multi = load_multi_stock(data_dir, min_history_days=60)
            if ticker_dfs:
                print(f"  Loaded {len(ticker_dfs)} tickers từ {data_dir}")
                return ticker_dfs, fwd_ret_multi
        print("  Không tìm thấy data thực, dùng synthetic data (500 ngày, 50 tickers)")
        ticker_dfs, fwd_ret_multi = make_sample_data_multi(n_days=500, n_tickers=50, seed=42)
        return ticker_dfs, fwd_ret_multi
    except ImportError as e:
        print(f"  Import error: {e}\n  Tạo synthetic data nội bộ...")
        return _make_synthetic_data()


def _make_synthetic_data(n_days=500, n_tickers=50, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    tickers = [f"SYM{i:02d}" for i in range(n_tickers)]

    ticker_dfs = {}
    for t in tickers:
        returns = rng.normal(0.0005, 0.015, n_days)
        close   = 100.0 * np.exp(np.cumsum(returns))
        volume  = rng.lognormal(15, 0.5, n_days)
        df = pd.DataFrame({
            "open": close * (1 + rng.normal(0, 0.005, n_days)),
            "high": close * (1 + rng.uniform(0.005, 0.015, n_days)),
            "low":  close * (1 - rng.uniform(0.005, 0.015, n_days)),
            "close": close,
            "volume": volume,
            "returns": np.concatenate([[np.nan], np.diff(np.log(close))]),
            "vwap": close,
            "adv20": pd.Series(close * volume).rolling(20).mean().values,
        }, index=dates)
        df.index.name = "date"
        ticker_dfs[t] = df

    fwd_parts = [
        df["close"].pct_change(1).shift(-1).rename(t)
        for t, df in ticker_dfs.items()
    ]
    fwd_ret_multi = pd.concat(fwd_parts, axis=1)
    return ticker_dfs, fwd_ret_multi


# ─────────────────────────────────────────────────────────────────────
# PHẦN 2 — Eval alpha expression → signal DataFrame
# ─────────────────────────────────────────────────────────────────────

def _exec_alpha(expression: str, ticker_dfs: dict) -> Optional[pd.DataFrame]:
    """Chạy expression trên từng ticker, trả về signal_df [dates × tickers]."""
    try:
        import alpha_operators as op_module
    except ImportError:
        op_module = None

    def _exec_one(expr, df_t):
        ns = {"df": df_t, "np": np, "pd": pd}
        if op_module:
            ns.update({n: getattr(op_module, n)
                       for n in dir(op_module) if not n.startswith("_")})
        else:
            # fallback operators tối thiểu
            ns.update({
                "ts_maxmin_scale": lambda s, w: (
                    s - s.rolling(w).min()) / (s.rolling(w).max() - s.rolling(w).min() + 1e-9),
                "ts_delta": lambda s, p: s.diff(p),
                "rank": lambda s: s.expanding().rank(pct=True),
                "ts_zscore_scale": lambda s, w: (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9),
                "neg": lambda s: -s,
            })
        for col in df_t.columns:
            ns[col] = df_t[col]
        exec(expr, ns)
        series = ns.get("alpha")
        if not isinstance(series, pd.Series):
            return None
        series = series.replace([np.inf, -np.inf], np.nan)
        mu  = series.expanding(min_periods=20).mean()
        std = series.expanding(min_periods=20).std()
        return ((series - mu) / (std + 1e-9)).clip(-5, 5)

    parts = {}
    for ticker, df_t in ticker_dfs.items():
        try:
            s = _exec_one(expression, df_t)
            if s is not None and s.dropna().std() > 1e-9:
                parts[ticker] = s
        except Exception:
            pass

    if len(parts) < 5:
        return None

    signal_df = pd.DataFrame(parts)
    signal_df.index = pd.to_datetime(signal_df.index)
    # Cross-sectional zscore mỗi ngày
    signal_norm = signal_df.apply(
        lambda row: (row - row.mean()) / (row.std() + 1e-9), axis=1
    )
    return signal_norm


# ─────────────────────────────────────────────────────────────────────
# PHẦN 3 — 5 cách tính Sharpe
# ─────────────────────────────────────────────────────────────────────

TEST_RATIO = 0.3

def _split_oos(signal_df, fwd_ret_df):
    common = sorted(signal_df.index.intersection(fwd_ret_df.index))
    split  = int(len(common) * (1 - TEST_RATIO))
    return common[split:]


# ── Cách 0 (baseline): rank-based, toàn universe, KHÔNG cost ─────────
def method_baseline(signal_df: pd.DataFrame, fwd_ret_df: pd.DataFrame) -> dict:
    """
    Baseline: current implementation.
    rank-based continuous positions, toàn bộ tickers, không transaction cost.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]
    fwd = fwd_ret_df[common_t]

    daily_pnl = []
    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        ct = row_sig.index.intersection(row_fwd.index)
        n  = len(ct)
        if n < 4:
            continue
        ranks = row_sig[ct].rank(ascending=True)
        pos   = (ranks - (n + 1) / 2) / ((n - 1) / 2 + 1e-9)
        abs_s = pos.abs().sum()
        if abs_s < 1e-9:
            continue
        pos = pos / abs_s
        daily_pnl.append(float((pos * row_fwd[ct]).sum()))

    return _stats(np.array(daily_pnl), "Baseline (rank, no cost, full universe)")


# ── Cách 1: rank-based + transaction cost ────────────────────────────
def method_with_cost(signal_df: pd.DataFrame, fwd_ret_df: pd.DataFrame,
                     cost_per_turnover: float = 0.001) -> dict:
    """
    Thêm transaction cost 10bps mỗi đơn vị turnover.
    cost_per_turnover = 0.001 tương đương 10 bps.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]
    fwd = fwd_ret_df[common_t]

    daily_pnl = []
    prev_pos  = None
    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        ct = row_sig.index.intersection(row_fwd.index)
        n  = len(ct)
        if n < 4:
            continue
        ranks = row_sig[ct].rank(ascending=True)
        pos   = (ranks - (n + 1) / 2) / ((n - 1) / 2 + 1e-9)
        abs_s = pos.abs().sum()
        if abs_s < 1e-9:
            continue
        pos = pos / abs_s

        if prev_pos is not None:
            common_prev = pos.index.intersection(prev_pos.index)
            delta = pos.reindex(common_prev).fillna(0) - prev_pos.reindex(common_prev).fillna(0)
            turnover = float(delta.abs().sum()) / 2.0
            cost = turnover * cost_per_turnover
        else:
            cost = 0.0
        prev_pos = pos.copy()

        pnl = float((pos * row_fwd[ct]).sum()) - cost
        daily_pnl.append(pnl)

    return _stats(np.array(daily_pnl),
                  f"With cost ({cost_per_turnover*10000:.0f}bps/turnover)")


# ── Helper: tính cost thực tế HOSE ───────────────────────────────────

def _hose_cost(delta: pd.Series,
               brokerage: float = 0.0015,
               tax_sell: float = 0.001) -> float:
    """
    Tính transaction cost theo cấu trúc phí thực tế HOSE.
      buy_value  = sum(max(delta,  0))
      sell_value = sum(max(-delta, 0))
      cost = (buy + sell) * brokerage + sell * tax_sell
    """
    buy_value  = float(delta.clip(lower=0).sum())
    sell_value = float((-delta).clip(lower=0).sum())
    return (buy_value + sell_value) * brokerage + sell_value * tax_sell


def _threshold_rebalance(target_pos: pd.Series,
                         prev_pos: pd.Series,
                         min_trade: float = 0.001) -> pd.Series:
    """
    Threshold rebalancing: chỉ thực hiện trade nếu |delta[i]| > min_trade.
    Ticker có delta nhỏ hơn ngưỡng → giữ nguyên position cũ, không tính cost.

    min_trade = 0.001 tương đương ~0.1% portfolio value mỗi ticker.
    Với 401 tickers, position trung bình ~0.25% → ngưỡng 0.1% là hợp lý.
    """
    all_t    = target_pos.index.union(prev_pos.index)
    tgt_full = target_pos.reindex(all_t).fillna(0)
    prv_full = prev_pos.reindex(all_t).fillna(0)
    delta    = tgt_full - prv_full

    # Chỉ giữ delta vượt ngưỡng, phần còn lại = 0 (không trade)
    delta_filtered = delta.where(delta.abs() > min_trade, other=0.0)
    # Actual executed position = prev + filtered_delta
    actual_pos = prv_full + delta_filtered
    return actual_pos, delta_filtered


def _debug_turnover(signal_df: pd.DataFrame,
                    fwd_ret_df: pd.DataFrame,
                    n_sample: int = 5) -> None:
    """
    In thống kê turnover để hiểu tại sao cost cao.
    Chỉ dùng để debug, không ảnh hưởng kết quả.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]

    turnovers = []
    prev_pos  = None
    for date in test_dates[:60]:  # chỉ sample 60 ngày đầu OOS
        if date not in sig.index:
            continue
        row_sig = sig.loc[date].dropna()
        n = len(row_sig)
        if n < 4:
            continue
        ranks = row_sig.rank(ascending=True)
        pos   = (ranks - (n + 1) / 2) / ((n - 1) / 2 + 1e-9)
        pos   = pos / pos.abs().sum()

        if prev_pos is not None:
            cp    = pos.index.intersection(prev_pos.index)
            delta = pos.reindex(cp).fillna(0) - prev_pos.reindex(cp).fillna(0)
            turnovers.append(float(delta.abs().sum()) / 2.0)
        prev_pos = pos.copy()

    if turnovers:
        arr = np.array(turnovers)
        print(f"\n  [DEBUG Turnover] mean={arr.mean():.4f}  "
              f"median={np.median(arr):.4f}  "
              f"max={arr.max():.4f}  "
              f"→ daily cost≈{arr.mean()*0.004*100:.3f}%  "
              f"(×252 = {arr.mean()*0.004*252*100:.1f}%/năm)")


# ── Cách 5: signal-weighted, toàn universe ───────────────────────────
def method_signal_weighted_full(signal_df: pd.DataFrame,
                                fwd_ret_df: pd.DataFrame,
                                brokerage: float = 0.0015,
                                tax_sell: float = 0.001) -> dict:
    """
    Position tỉ lệ trực tiếp với signal (không rank-transform), toàn universe.
    pos[i] = signal[i] / sum(|signal|)
    Cost dùng cấu trúc phí HOSE thực tế.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]
    fwd = fwd_ret_df[common_t]

    daily_pnl = []
    prev_pos  = None
    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        ct = row_sig.index.intersection(row_fwd.index)
        if len(ct) < 4:
            continue

        pos   = row_sig[ct].copy()
        abs_s = pos.abs().sum()
        if abs_s < 1e-9:
            continue
        pos = pos / abs_s

        if prev_pos is not None:
            all_t    = pos.index.union(prev_pos.index)
            delta    = pos.reindex(all_t).fillna(0) - prev_pos.reindex(all_t).fillna(0)
            cost     = _hose_cost(delta, brokerage, tax_sell)
        else:
            cost = 0.0
        prev_pos = pos.copy()

        pnl = float((pos * row_fwd[ct]).sum()) - cost
        daily_pnl.append(pnl)

    return _stats(np.array(daily_pnl), "Signal-weighted full (HOSE cost)")


# ── Cách 6: signal-weighted, chỉ vùng quá mua/bán (|s| > 1) ─────────
def method_signal_weighted_extreme(signal_df: pd.DataFrame,
                                   fwd_ret_df: pd.DataFrame,
                                   threshold: float = 1.0,
                                   brokerage: float = 0.0015,
                                   tax_sell: float = 0.001) -> dict:
    """
    Chỉ giao dịch tickers có |signal| > threshold.
    Position tỉ lệ chính xác theo signal trong vùng đó.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]
    fwd = fwd_ret_df[common_t]

    daily_pnl = []
    prev_pos  = None
    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        ct = row_sig.index.intersection(row_fwd.index)
        if len(ct) < 4:
            continue

        extreme_sig = row_sig[ct][row_sig[ct].abs() > threshold]

        if extreme_sig.empty:
            if prev_pos is not None and len(prev_pos) > 0:
                delta = -prev_pos
                cost  = _hose_cost(delta, brokerage, tax_sell)
                daily_pnl.append(-cost)
            else:
                daily_pnl.append(0.0)
            prev_pos = pd.Series(dtype=float)
            continue

        pos   = extreme_sig.copy()
        abs_s = pos.abs().sum()
        if abs_s < 1e-9:
            continue
        pos = pos / abs_s

        if prev_pos is not None and len(prev_pos) > 0:
            all_t = pos.index.union(prev_pos.index)
            delta = pos.reindex(all_t).fillna(0) - prev_pos.reindex(all_t).fillna(0)
            cost  = _hose_cost(delta, brokerage, tax_sell)
        else:
            cost = _hose_cost(pos, brokerage, tax_sell)
        prev_pos = pos.copy()

        pnl = float((pos * row_fwd[ct].reindex(pos.index).fillna(0)).sum()) - cost
        daily_pnl.append(pnl)

    return _stats(np.array(daily_pnl),
                  f"Signal-weighted extreme (|s|>{threshold}, HOSE cost)")


# ── Cách 7: rank-based + threshold rebalancing ───────────────────────
def method_threshold_rebalance(signal_df: pd.DataFrame,
                               fwd_ret_df: pd.DataFrame,
                               min_trade: float = 0.001,
                               brokerage: float = 0.0015,
                               tax_sell: float = 0.001) -> dict:
    """
    Rank-based như Baseline, nhưng chỉ execute trade khi |delta[i]| > min_trade.
    Giảm số lượt trade nhỏ vụn → turnover thực tế thấp hơn nhiều.

    min_trade=0.001: chỉ rebalance ticker khi position thay đổi > 0.1% portfolio.
    Với 401 tickers, avg position ~0.25% → lọc được ~40-60% trade nhỏ.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]
    fwd = fwd_ret_df[common_t]

    daily_pnl = []
    actual_pos = None   # position thực tế đang giữ (sau threshold filter)

    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        ct = row_sig.index.intersection(row_fwd.index)
        n  = len(ct)
        if n < 4:
            continue

        # Target position (rank-based)
        ranks      = row_sig[ct].rank(ascending=True)
        target_pos = (ranks - (n + 1) / 2) / ((n - 1) / 2 + 1e-9)
        abs_s      = target_pos.abs().sum()
        if abs_s < 1e-9:
            continue
        target_pos = target_pos / abs_s

        if actual_pos is not None and len(actual_pos) > 0:
            executed_pos, delta = _threshold_rebalance(
                target_pos, actual_pos, min_trade
            )
            cost = _hose_cost(delta, brokerage, tax_sell)
        else:
            executed_pos = target_pos.copy()
            cost = _hose_cost(target_pos, brokerage, tax_sell)
        actual_pos = executed_pos.copy()

        hold_ct = executed_pos.index.intersection(row_fwd.index)
        pnl = float((executed_pos[hold_ct] * row_fwd[hold_ct]).sum()) - cost
        daily_pnl.append(pnl)

    return _stats(np.array(daily_pnl),
                  f"Threshold rebalance (min_trade={min_trade*100:.1f}%, HOSE cost)")


# ── Cách 8: signal-weighted + threshold rebalancing ──────────────────
def method_signal_weighted_threshold(signal_df: pd.DataFrame,
                                     fwd_ret_df: pd.DataFrame,
                                     min_trade: float = 0.001,
                                     brokerage: float = 0.0015,
                                     tax_sell: float = 0.001) -> dict:
    """
    Signal-weighted + threshold rebalancing.
    Kết hợp: position tỉ lệ theo signal + chỉ trade khi delta vượt ngưỡng.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]
    fwd = fwd_ret_df[common_t]

    daily_pnl = []
    actual_pos = None

    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        ct = row_sig.index.intersection(row_fwd.index)
        if len(ct) < 4:
            continue

        # Signal-weighted target
        target_pos = row_sig[ct].copy()
        abs_s      = target_pos.abs().sum()
        if abs_s < 1e-9:
            continue
        target_pos = target_pos / abs_s

        if actual_pos is not None and len(actual_pos) > 0:
            executed_pos, delta = _threshold_rebalance(
                target_pos, actual_pos, min_trade
            )
            cost = _hose_cost(delta, brokerage, tax_sell)
        else:
            executed_pos = target_pos.copy()
            cost = _hose_cost(target_pos, brokerage, tax_sell)
        actual_pos = executed_pos.copy()

        hold_ct = executed_pos.index.intersection(row_fwd.index)
        pnl = float((executed_pos[hold_ct] * row_fwd[hold_ct]).sum()) - cost
        daily_pnl.append(pnl)

    return _stats(np.array(daily_pnl),
                  f"Signal-weighted + threshold (min={min_trade*100:.1f}%, HOSE cost)")




# ── Cách 2: concentrated portfolio (top/bottom N tickers) ────────────
def method_concentrated(signal_df: pd.DataFrame, fwd_ret_df: pd.DataFrame,
                         top_n: int = 20) -> dict:
    """
    Chỉ giữ top N long + top N short thay vì toàn bộ universe.
    Std(daily_pnl) tăng lên → Sharpe về range realistic hơn.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]
    fwd = fwd_ret_df[common_t]

    daily_pnl = []
    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        ct = row_sig.index.intersection(row_fwd.index)
        if len(ct) < top_n * 2:
            continue
        sorted_sig = row_sig[ct].sort_values()
        short_idx  = sorted_sig.iloc[:top_n].index
        long_idx   = sorted_sig.iloc[-top_n:].index
        selected   = short_idx.append(long_idx)

        pos = row_sig[selected].copy()
        pos[short_idx] = -1.0 / top_n
        pos[long_idx]  =  1.0 / top_n

        abs_s = pos.abs().sum()
        if abs_s < 1e-9:
            continue
        pos = pos / abs_s

        pnl = float((pos * row_fwd[selected]).sum())
        daily_pnl.append(pnl)

    return _stats(np.array(daily_pnl), f"Concentrated (top/bottom {top_n})")


# ── Cách 3: signal clipping — |signal| > 1 = overbought/oversold ─────
def method_clipped_signal(signal_df: pd.DataFrame,
                           fwd_ret_df: pd.DataFrame) -> dict:
    """
    Đề xuất từ user: signal > 1 → overbought (short), signal < -1 → oversold (long).
    Position = clip(signal, -1, 1), sau đó normalize.
    Ý nghĩa: signal càng cực đoan càng bị giảm weight về ±1,
    tập trung vào tín hiệu moderate thay vì outlier.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]
    fwd = fwd_ret_df[common_t]

    daily_pnl = []
    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        ct = row_sig.index.intersection(row_fwd.index)
        if len(ct) < 4:
            continue

        # Clip signal về [-1, 1] — extreme signal bị cap
        pos = row_sig[ct].clip(-1.0, 1.0)
        abs_s = pos.abs().sum()
        if abs_s < 1e-9:
            continue
        pos = pos / abs_s

        pnl = float((pos * row_fwd[ct]).sum())
        daily_pnl.append(pnl)

    return _stats(np.array(daily_pnl), "Clipped signal (|s|>1 capped)")


# ── Cách 4: binary long/short (signal != 0) ──────────────────────────
def method_binary(signal_df: pd.DataFrame, fwd_ret_df: pd.DataFrame,
                  threshold: float = 0.5) -> dict:
    """
    Binary: pos = +1 nếu signal > threshold, -1 nếu < -threshold, 0 otherwise.
    Xem std có về mức hợp lý không.
    """
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df[common_t]
    fwd = fwd_ret_df[common_t]

    daily_pnl = []
    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        row_sig = sig.loc[date].dropna()
        row_fwd = fwd.loc[date].dropna()
        ct = row_sig.index.intersection(row_fwd.index)
        if len(ct) < 4:
            continue

        pos = pd.Series(0.0, index=ct)
        pos[row_sig[ct] >  threshold] =  1.0
        pos[row_sig[ct] < -threshold] = -1.0

        n_active = (pos != 0).sum()
        if n_active == 0:
            continue
        pos = pos / pos.abs().sum()

        pnl = float((pos * row_fwd[ct]).sum())
        daily_pnl.append(pnl)

    return _stats(np.array(daily_pnl),
                  f"Binary (|signal| > {threshold})")


# ─────────────────────────────────────────────────────────────────────
# PHẦN 4 — Tính thống kê
# ─────────────────────────────────────────────────────────────────────

def _stats(arr: np.ndarray, label: str) -> dict:
    if len(arr) < 20:
        return {"label": label, "error": "không đủ dữ liệu"}

    mean_d  = float(arr.mean())
    std_d   = float(arr.std())
    sharpe  = float(mean_d / (std_d + 1e-12) * np.sqrt(252))

    # Geometric annualized return
    arr_c        = np.clip(arr, -0.5, 0.5)
    total_ret    = float(np.prod(1.0 + arr_c) - 1.0)
    n_days       = len(arr)
    if total_ret > -0.99:
        ann_ret = (1.0 + total_ret) ** (252.0 / n_days) - 1.0
    else:
        ann_ret = mean_d * 252  # fallback arithmetic

    # Max drawdown
    cum   = np.cumprod(1.0 + arr_c)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak) / (peak + 1e-9)
    mdd   = float(dd.min())

    # Win rate
    win_rate = float((arr > 0).mean())

    return {
        "label":        label,
        "n_days":       n_days,
        "mean_daily":   round(mean_d * 100, 4),    # %
        "std_daily":    round(std_d  * 100, 4),    # %
        "sharpe":       round(sharpe, 3),
        "ann_return":   round(ann_ret * 100, 2),   # %/năm
        "max_drawdown": round(mdd * 100, 2),        # %
        "win_rate":     round(win_rate * 100, 1),   # %
    }


def _compute_ic_oos(signal_df: pd.DataFrame,
                    fwd_ret_df: pd.DataFrame) -> float:
    """IC OOS để cross-check."""
    test_dates = _split_oos(signal_df, fwd_ret_df)
    common_t   = signal_df.columns.intersection(fwd_ret_df.columns)
    sig = signal_df.loc[test_dates, common_t]
    fwd = fwd_ret_df.loc[test_dates, common_t]

    ic_list = []
    for date in test_dates:
        if date not in sig.index or date not in fwd.index:
            continue
        rs = sig.loc[date].dropna()
        rf = fwd.loc[date].dropna()
        ct = rs.index.intersection(rf.index)
        if len(ct) < 10:
            continue
        ic = rs[ct].corr(rf[ct], method="spearman")
        if not np.isnan(ic):
            ic_list.append(ic)
    return round(float(np.mean(ic_list)), 6) if ic_list else np.nan


# ─────────────────────────────────────────────────────────────────────
# PHẦN 5 — In kết quả
# ─────────────────────────────────────────────────────────────────────

def _verdict(results: List[dict], ic_oos: float) -> dict:
    """
    Đưa ra verdict tự động. Primary metric: Signal-weighted full (HOSE cost).
    Fallback: With cost nếu không có signal-weighted.
    """
    primary = next((r for r in results
                    if "Signal-weighted full" in r.get("label", "")), None)
    if primary is None:
        primary = next((r for r in results
                        if "With cost" in r.get("label", "")), None)
    base_r  = next((r for r in results
                    if "Baseline" in r.get("label", "")), None)
    extreme = next((r for r in results
                    if "extreme" in r.get("label", "")), None)

    if primary is None or "error" in primary:
        return {"text": "Không tính được verdict", "viable": False}

    sharpe_p    = primary["sharpe"]
    ret_p       = primary["ann_return"]
    sharpe_base = base_r["sharpe"] if base_r and "error" not in base_r else 0
    inflate_ratio = sharpe_base / (sharpe_p + 1e-9) if sharpe_p > 0 else float("inf")

    lines = []

    # 1. Viable?
    if sharpe_p > 1.5 and ret_p > 5:
        viability, viable = "VIABLE ✓", True
    elif sharpe_p > 0:
        viability, viable = "MARGINAL ~", True
    else:
        viability, viable = "NOT VIABLE ✗", False
    lines.append(f"Viability (signal-weighted, HOSE cost): {viability}  "
                 f"[Sharpe={sharpe_p:.2f}, Ann={ret_p:.1f}%/năm]")

    # 2. So sánh full vs extreme
    if extreme and "error" not in extreme:
        diff = extreme["sharpe"] - sharpe_p
        if diff > 0.3:
            lines.append(f"Extreme-only tốt hơn full ({extreme['sharpe']:.2f} vs {sharpe_p:.2f})"
                         f" → signal mạnh nhất tập trung ở vùng |s|>1")
        elif diff < -0.3:
            lines.append(f"Full tốt hơn extreme ({sharpe_p:.2f} vs {extreme['sharpe']:.2f})"
                         f" → signal moderate cũng đóng góp, không nên bỏ")
        else:
            lines.append(f"Full ≈ Extreme ({sharpe_p:.2f} vs {extreme['sharpe']:.2f})"
                         f" → có thể dùng extreme để tiết kiệm cost")

    # 3. Inflate check
    if inflate_ratio > 3:
        lines.append(f"Inflate ratio: {inflate_ratio:.1f}x  ← turnover cao, cost ăn nhiều")
    elif inflate_ratio > 1.5:
        lines.append(f"Inflate ratio: {inflate_ratio:.1f}x  ← moderate")
    else:
        lines.append(f"Inflate ratio: {inflate_ratio:.1f}x  ← ổn")

    # 4. IC consistency
    if ic_oos > 0.04 and viable:
        lines.append(f"IC_OOS={ic_oos:+.4f} consistent → alpha signal thực")
    elif ic_oos > 0.04 and not viable:
        lines.append(f"IC_OOS={ic_oos:+.4f} tốt nhưng Sharpe âm → cost quá cao "
                     f"cho daily rebalancing, cân nhắc weekly")
    else:
        lines.append(f"IC_OOS={ic_oos:+.4f} thấp → alpha yếu")

    # 5. Recommendation
    if viable and inflate_ratio > 3:
        lines.append("→ Deploy với signal-weighted. Xem xét giảm rebalance frequency.")
    elif viable:
        lines.append("→ Alpha production-ready với phí HOSE thực tế.")
    else:
        lines.append("→ Loại hoặc cải thiện alpha trước khi deploy.")

    return {"text": "\n    ".join(lines), "viable": viable}


def _print_results(alpha_id: str, ic_oos: float,
                   results: List[dict], library_sharpe: float) -> dict:
    """In kết quả và trả về verdict dict."""
    SEP = "─" * 100

    print(f"\n{'═'*100}")
    print(f"  Alpha: {alpha_id}   |   IC_OOS (cross-sectional): {ic_oos:+.4f}"
          f"   |   Sharpe trong library: {library_sharpe:.4f}")
    print('═'*100)

    # Header
    print(f"  {'Method':<45} {'N days':>7} {'Mean%/d':>8} {'Std%/d':>8} "
          f"{'Sharpe':>8} {'Ann Ret%':>9} {'MaxDD%':>8} {'WinRate%':>9}")
    print(f"  {SEP}")

    for r in results:
        if "error" in r:
            print(f"  {r['label']:<45}  ERROR: {r['error']}")
            continue

        flag = ""
        if r["sharpe"] > 5:
            flag = " ⚠ inflated"
        elif r["sharpe"] > 3:
            flag = " ✓ high"
        elif r["sharpe"] > 1:
            flag = " ✓ ok"
        elif r["sharpe"] > 0:
            flag = " ~ marginal"
        else:
            flag = " ✗ negative"

        # Đánh dấu With cost là primary
        prefix = "→ " if "With cost" in r["label"] else "  "
        print(
            f"{prefix}{r['label']:<45} "
            f"{r['n_days']:>7} "
            f"{r['mean_daily']:>7.4f}% "
            f"{r['std_daily']:>7.4f}% "
            f"{r['sharpe']:>8.3f}{flag:<14} "
            f"{r['ann_return']:>8.2f}% "
            f"{r['max_drawdown']:>7.2f}% "
            f"{r['win_rate']:>8.1f}%"
        )

    # Verdict
    v = _verdict(results, ic_oos)
    print(f"\n  VERDICT:")
    print(f"    {v['text']}")

    return v


# ─────────────────────────────────────────────────────────────────────
# PHẦN 6 — Main
# ─────────────────────────────────────────────────────────────────────

def run_comparison(library_path: str, data_dir: str,
                   top_n: int = 20,
                   brokerage: float = 0.0015,
                   tax_sell: float = 0.001) -> None:

    round_trip_bps = (brokerage * 2 + tax_sell) * 10000
    print(f"\n{'='*100}")
    print(f"  SHARPE COMPARISON — Alpha Library: {library_path}")
    print(f"  Phí HOSE: môi giới {brokerage*100:.2f}%/chiều, thuế bán {tax_sell*100:.2f}%"
          f"  →  round-trip ~{round_trip_bps:.0f}bps")
    print(f"{'='*100}")

    with open(library_path, "r", encoding="utf-8") as f:
        library = json.load(f)
    print(f"\n  Số alpha trong library: {len(library)}")

    print("\n  Loading data...")
    ticker_dfs, fwd_ret_multi = _load_data(data_dir)
    print(f"  Universe: {len(ticker_dfs)} tickers | "
          f"Fwd return shape: {fwd_ret_multi.shape}")

    # cost đơn giản cho method cũ: lấy brokerage 2 chiều + thuế bán
    simple_cost = brokerage * 2 + tax_sell

    for alpha in library:
        alpha_id   = alpha.get("id", "?")
        expression = alpha.get("expression", "")
        lib_sharpe = alpha.get("sharpe_oos", float("nan"))

        print(f"\n  Đang tính alpha [{alpha_id}]: {expression[:80]}")

        signal_df = _exec_alpha(expression, ticker_dfs)
        if signal_df is None:
            print(f"  ✗ Không tính được signal cho {alpha_id}")
            continue

        ic_oos = _compute_ic_oos(signal_df, fwd_ret_multi)

        results = [
            method_baseline(signal_df, fwd_ret_multi),
            method_with_cost(signal_df, fwd_ret_multi, simple_cost),
            method_signal_weighted_full(signal_df, fwd_ret_multi,
                                        brokerage, tax_sell),
            method_signal_weighted_extreme(signal_df, fwd_ret_multi,
                                           threshold=1.0,
                                           brokerage=brokerage,
                                           tax_sell=tax_sell),
            method_threshold_rebalance(signal_df, fwd_ret_multi,
                                       min_trade=0.001,
                                       brokerage=brokerage,
                                       tax_sell=tax_sell),
            method_signal_weighted_threshold(signal_df, fwd_ret_multi,
                                             min_trade=0.001,
                                             brokerage=brokerage,
                                             tax_sell=tax_sell),
            method_concentrated(signal_df, fwd_ret_multi, top_n),
            method_clipped_signal(signal_df, fwd_ret_multi),
            method_binary(signal_df, fwd_ret_multi, threshold=0.5),
        ]

        _debug_turnover(signal_df, fwd_ret_multi)
        _print_results(alpha_id, ic_oos, results, lib_sharpe)

    print(f"\n{'='*100}")
    print("  NHẬN XÉT CHUNG:")
    print("  • Baseline                  — không cost, chỉ tham chiếu")
    print("  • With cost                 — rank-based + HOSE cost đầy đủ, không threshold")
    print("  • Signal-weighted full      — position ∝ signal, HOSE cost, không threshold")
    print("  • Signal-weighted extreme   — chỉ |s|>1, ít trade hơn → cost thấp hơn")
    print("  • Threshold rebalance       — rank-based, chỉ trade khi delta > 0.1% portfolio")
    print("  • Signal-weighted+threshold — kết hợp signal weighting + threshold rebalancing")
    print("  • [DEBUG Turnover]          — xem turnover thực tế để hiểu cost cao/thấp thế nào")
    print("  • IC_OOS là ground-truth, không bị ảnh hưởng bởi position sizing hay cost")
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description="So sánh các cách tính Sharpe trên alpha_library.json"
    )
    parser.add_argument(
        "--library", default="alpha_library.json",
        help="Đường dẫn tới alpha_library.json (mặc định: alpha_library.json)"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Thư mục chứa CSV data (mặc định: data)"
    )
    parser.add_argument(
        "--brokerage", type=float, default=0.15,
        help="Phí môi giới %% mỗi chiều (mặc định: 0.15%%)"
    )
    parser.add_argument(
        "--tax", type=float, default=0.1,
        help="Thuế bán %% (mặc định: 0.1%%)"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Số tickers cho concentrated method (mặc định: 20)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.library):
        print(f"Không tìm thấy {args.library}")
        sys.exit(1)

    run_comparison(
        library_path=args.library,
        data_dir=args.data_dir,
        top_n=args.top_n,
        brokerage=args.brokerage / 100.0,
        tax_sell=args.tax / 100.0,
    )


if __name__ == "__main__":
    main()