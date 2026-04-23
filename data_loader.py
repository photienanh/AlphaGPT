"""
data_loader.py
Load và tính technical indicators từ OHLCV data.
Output: df với đầy đủ các fields cho alpha expressions.
"""
import numpy as np
import pandas as pd
from typing import Optional
import logging

log = logging.getLogger(__name__)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính technical indicators từ OHLCV.
    Giữ nguyên logic từ indicators.py của codebase cũ.
    """
    df = df.copy()

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        df = df.sort_values("time").reset_index(drop=True)

    close  = df["close"]
    volume = df["volume"]

    # SMA
    df["SMA_5"]  = close.rolling(5).mean()
    df["SMA_20"] = close.rolling(20).mean()

    # EMA
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()

    # Momentum
    df["Momentum_3"]  = close.diff(3)
    df["Momentum_10"] = close.diff(10)

    # RSI_14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = np.where(
        loss == 0, 100,
        np.where(gain == 0, 0, 100 - (100 / (1 + gain / loss)))
    )

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    rolling_std = close.rolling(20).std()
    df["BB_Middle"] = df["SMA_20"]
    df["BB_Upper"]  = df["SMA_20"] + 2 * rolling_std
    df["BB_Lower"]  = df["SMA_20"] - 2 * rolling_std

    # OBV
    direction = pd.Series(
        np.sign(close.diff().to_numpy()),
        index=close.index,
    )
    df["OBV"] = (direction * volume).fillna(0).cumsum()

    # Drop warm-up rows (EMA_26 butuhkan 26 baris)
    df = df.dropna(subset=["MACD"]).reset_index(drop=True)

    return df


def load_from_csv(path: str) -> pd.DataFrame:
    """Load từ CSV file, tính indicators, return df với time index."""
    raw = pd.read_csv(path)
    df  = add_technical_indicators(raw)
    if "time" in df.columns:
        df = df.set_index("time")
        df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()


def load_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nhận DataFrame OHLCV, tính indicators, return df sẵn dùng."""
    result = add_technical_indicators(df)
    if "time" in result.columns:
        result = result.set_index("time")
        result.index = pd.to_datetime(result.index).normalize()
    return result.sort_index()


def make_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Tính forward return để làm target cho IC computation."""
    return df["close"].pct_change(horizon).shift(-horizon).rename(f"fwd_{horizon}d")


def make_sample_data(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Tạo synthetic OHLCV data để test — không cần file thật.
    Giá theo GBM, volume có noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

    # Geometric Brownian Motion cho close price
    returns = rng.normal(0.0005, 0.015, n_days)
    close   = 100.0 * np.exp(np.cumsum(returns))

    # OHLV derived from close
    noise   = rng.uniform(0.005, 0.015, n_days)
    high    = close * (1 + noise)
    low     = close * (1 - noise)
    open_   = close * (1 + rng.normal(0, 0.005, n_days))
    volume  = rng.lognormal(15, 0.5, n_days)

    df = pd.DataFrame({
        "time":   dates,
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    })
    return add_technical_indicators(df)