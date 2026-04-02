import os
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.paths import PRICE_DIR, FEATURES_DIR


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        df = df.sort_values("time").reset_index(drop=True)

    close = df["close"]
    volume = df["volume"]

    # SMA: đường trung bình động đơn giản (5 kỳ và 20 kỳ)
    df["SMA_5"] = close.rolling(window=5).mean()
    df["SMA_20"] = close.rolling(window=20).mean()

    # EMA: đường trung bình động lũy thừa (10 kỳ)
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()

    # Momentum: chỉ báo động lượng (3 kỳ và 10 kỳ)
    df["Momentum_3"] = close.diff(periods=3)
    df["Momentum_10"] = close.diff(periods=10)

    # RSI_14: chỉ số sức mạnh tương đối (14 kỳ)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    df["RSI_14"] = np.where(
        loss == 0, 100,
        np.where(gain == 0, 0, 100 - (100 / (1 + gain / loss)))
    )

    # MACD + Signal
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20 kỳ, 2 độ lệch chuẩn)
    rolling_std_20 = close.rolling(window=20).std()
    df["BB_Middle"] = df["SMA_20"]
    df["BB_Upper"] = df["SMA_20"] + 2 * rolling_std_20
    df["BB_Lower"] = df["SMA_20"] - 2 * rolling_std_20

    # OBV: On-Balance Volume
    # Row đầu tiên: close.diff() = NaN → sign = NaN → fillna(0) → OBV bắt đầu từ 0 (convention chuẩn)
    direction = pd.Series(
        np.sign(close.diff().to_numpy()),
        index=close.index,
        name="direction",
    )
    df["OBV"] = (direction * volume).fillna(0).cumsum()

    # Drop warm-up rows (NaN từ rolling windows dài nhất = 26 kỳ cho EMA_26)
    df = df.dropna().reset_index(drop=True)

    return df


def run_pipeline() -> None:
    input_folder = PRICE_DIR
    output_folder = FEATURES_DIR

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        print(f"Không tìm thấy file nào trong '{input_folder}'.")
        return

    print(f"Bắt đầu tính toán technical indicators cho {len(csv_files)} file...\n")

    count = 0
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        symb = filename.split(".")[0]

        try:
            df = pd.read_csv(file_path)
            df_features = add_technical_indicators(df)

            out_path = os.path.join(output_folder, filename)
            df_features.to_csv(out_path, index=False)

            count += 1
            print(f"[OK] {symb}: {len(df_features)} rows, {len(df_features.columns)} cols → {out_path}")

        except Exception as e:
            print(f"[ERR] {symb}: {e}")

    print(f"\nHoàn tất! {count}/{len(csv_files)} cổ phiếu.")


if __name__ == "__main__":
    run_pipeline()