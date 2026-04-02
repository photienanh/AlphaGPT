import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from vnstock import Quote

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.universe import VN30_SYMBOLS
from core.paths import PRICE_DIR


def process_symbol(symbol: str, interval: str = "d") -> int:
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * 3)  # dùng timedelta thay vì replace(year=...) để tránh bug ngày 29/2

        quote = Quote(symbol=symbol, source="KBS")
        df = pd.DataFrame(
            quote.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
            )
        )
        if df.empty:
            raise Exception("No data returned")

        df = df.dropna()
        os.makedirs(PRICE_DIR, exist_ok=True)
        df.to_csv(os.path.join(PRICE_DIR, f"{symbol}.csv"), index=False)
        return 1

    except Exception as e:
        print(f"\nError processing {symbol}: {e}")
        return 0


def is_valid_csv(path: str) -> bool:
    """Kiểm tra file CSV tồn tại và có dữ liệu thực sự (không rỗng hoặc lỗi)."""
    try:
        df = pd.read_csv(path)
        return len(df) > 0
    except Exception:
        return False


def run_pipeline(symbols: list[str]) -> None:
    if not os.path.exists(PRICE_DIR):
        os.makedirs(PRICE_DIR)

    count_processed = 0

    for i, symbol in enumerate(symbols):
        path = os.path.join(PRICE_DIR, f"{symbol}.csv")

        # Skip chỉ khi file tồn tại VÀ có dữ liệu hợp lệ
        if os.path.exists(path) and is_valid_csv(path):
            count_processed += 1
            print(f"Skip {symbol} (đã có dữ liệu)", end="\r")
            continue

        count_processed += process_symbol(symbol=symbol)
        print(
            f"Processed {symbol} ({i+1}/{len(symbols)}), saved to {path}",
            end="\r",
        )

    print(f"\nProcessed {count_processed}/{len(symbols)} stocks")


if __name__ == "__main__":
    run_pipeline(VN30_SYMBOLS)