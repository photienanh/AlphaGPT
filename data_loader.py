"""
data_loader.py
Load toàn bộ CSV trong data/, tính technical indicators per ticker.
Output: (df_multi, fwd_ret_multi) — DataFrame có columns = tickers, index = dates.
"""
import os
import glob
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import logging

log = logging.getLogger(__name__)

REQUIRED_COLS = ["time", "ticker", "open", "high", "low", "close", "volume"]


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính tất cả technical indicators cho một ticker (single-ticker DataFrame).
    Input: DataFrame với index là dates, cột gồm open/high/low/close/volume.
    """
    df = df.copy().sort_index()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # vwap (xấp xỉ daily): (high + low + close) / 3
    df["vwap"] = (high + low + close) / 3

    # Moving averages
    df["sma_5"]  = close.rolling(5).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()

    # Momentum
    df["momentum_3"]  = close.diff(3)
    df["momentum_10"] = close.diff(10)

    # RSI
    df["rsi_14"] = _compute_rsi(close, 14)

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    std20 = close.rolling(20).std()
    df["bb_middle"] = df["sma_20"]
    df["bb_upper"]  = df["sma_20"] + 2 * std20
    df["bb_lower"]  = df["sma_20"] - 2 * std20

    # OBV
    direction = np.sign(close.diff())
    df["obv"] = (direction * volume).fillna(0).cumsum()

    # adv20 — average dollar volume 20 ngày
    df["adv20"] = (close * volume).rolling(20).mean()

    # returns
    df["returns"] = close.pct_change(1)

    return df


def _load_single_ticker(raw: pd.DataFrame) -> pd.DataFrame:
    """Load và tính indicators cho một ticker từ long-format slice."""
    raw = raw.copy()
    raw.columns = [c.lower() for c in raw.columns]

    raw["time"] = pd.to_datetime(raw["time"]).dt.normalize()
    raw = raw.sort_values("time").set_index("time")

    # Chỉ giữ cột OHLCV + industry
    keep_cols = ["open", "high", "low", "close", "volume"]
    if "industry" in raw.columns:
        keep_cols.append("industry")
    raw = raw[[c for c in keep_cols if c in raw.columns]]

    df = add_technical_indicators(raw)
    return df


def load_multi_stock(data_dir: str, min_history_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Đọc toàn bộ CSV trong data_dir (long format).
    Trả về:
      - df_panel: DataFrame MultiIndex (date, ticker) với tất cả indicators
      - signal_df: pivot table — index=dates, columns=tickers (dùng cho eval)
      - fwd_ret_multi: DataFrame index=dates, columns=tickers (forward returns)
    Ticker chỉ được đưa vào universe nếu có ít nhất min_history_days lịch sử.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        log.warning(f"[DataLoader] Không tìm thấy CSV trong {data_dir}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_dfs: Dict[str, pd.DataFrame] = {}
    for fpath in csv_files:
        try:
            raw = pd.read_csv(fpath)
            raw.columns = [c.lower() for c in raw.columns]

            # Long format: có cột ticker
            if "ticker" in raw.columns:
                for ticker, group in raw.groupby("ticker"):
                    try:
                        df_t = _load_single_ticker(group.drop(columns=["ticker"]))
                        if len(df_t) >= min_history_days:
                            all_dfs[str(ticker)] = df_t
                    except Exception as e:
                        log.warning(f"  Skip ticker {ticker}: {e}")
            else:
                # Mỗi file là một ticker — dùng tên file làm ticker
                ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
                try:
                    df_t = _load_single_ticker(raw)
                    if len(df_t) >= min_history_days:
                        all_dfs[ticker] = df_t
                except Exception as e:
                    log.warning(f"  Skip file {fpath}: {e}")
        except Exception as e:
            log.warning(f"[DataLoader] Lỗi đọc {fpath}: {e}")

    if not all_dfs:
        log.error("[DataLoader] Không load được ticker nào")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Build panel: concat theo tickers
    panels = []
    for ticker, df_t in all_dfs.items():
        df_t = df_t.copy()
        df_t["ticker"] = ticker
        panels.append(df_t)

    df_panel = pd.concat(panels)
    df_panel.index.name = "date"
    df_panel = df_panel.set_index("ticker", append=True)
    df_panel = df_panel.reorder_levels(["date", "ticker"])
    df_panel = df_panel.sort_index()

    # forward return per ticker
    fwd_parts = []
    for ticker, df_t in all_dfs.items():
        fwd = df_t["close"].pct_change(1).shift(-1).rename(ticker)
        fwd_parts.append(fwd)

    fwd_ret_multi = pd.concat(fwd_parts, axis=1)
    fwd_ret_multi.index.name = "date"

    return df_panel, all_dfs, fwd_ret_multi


def load_single_stock(path: str, min_history_days: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Backward compat: load một file CSV cho single ticker.
    Trả về (df, fwd_ret).
    """
    raw = pd.read_csv(path)
    raw.columns = [c.lower() for c in raw.columns]
    if "volumn" in raw.columns and "volume" not in raw.columns:
        raw = raw.rename(columns={"volumn": "volume"})

    if "ticker" in raw.columns:
        # Lấy ticker đầu tiên
        ticker = raw["ticker"].iloc[0]
        raw = raw[raw["ticker"] == ticker].drop(columns=["ticker"])

    df = _load_single_ticker(raw)
    fwd_ret = df["close"].pct_change(1).shift(-1).rename("fwd_1d")
    return df, fwd_ret


def make_sample_data_multi(n_days: int = 500, n_tickers: int = 10,
                           seed: int = 42) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Tạo synthetic multi-stock data để test."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    tickers = [f"SYM{i:02d}" for i in range(n_tickers)]

    ticker_dfs: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        returns = rng.normal(0.0005, 0.015, n_days)
        close   = 100.0 * np.exp(np.cumsum(returns))
        noise   = rng.uniform(0.005, 0.015, n_days)
        high    = close * (1 + noise)
        low     = close * (1 - noise)
        open_   = close * (1 + rng.normal(0, 0.005, n_days))
        volume  = rng.lognormal(15, 0.5, n_days)

        raw = pd.DataFrame({
            "open": open_, "high": high,
            "low": low, "close": close, "volume": volume,
        }, index=dates)
        raw.index.name = "time"
        df_t = add_technical_indicators(raw)
        df_t.index.name = "date"
        ticker_dfs[ticker] = df_t

    fwd_parts = [df_t["close"].pct_change(1).shift(-1).rename(t)
                 for t, df_t in ticker_dfs.items()]
    fwd_ret_multi = pd.concat(fwd_parts, axis=1)

    return ticker_dfs, fwd_ret_multi