"""
run.py — CLI entry point cho Alpha-GPT pipeline.

Usage:
    # Chạy với tên mã — pipeline tự tìm file data và tự sinh trading idea
    python run.py VCB
    python run.py HPG --iterations 3
    python run.py FPT --data-dir ./my_data

    # Nhiều mã cùng lúc
    python run.py VCB HPG FPT

    # Nếu muốn chỉ định idea thủ công
    python run.py VCB --idea "RSI divergence momentum"
"""
import asyncio
import argparse
import logging
import os
import glob
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Thư mục data mặc định — có thể override bằng --data-dir
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def resolve_data_path(symbol: str, data_dir: str) -> str:
    """
    Tìm file CSV cho symbol trong data_dir.
    Thử các pattern: VCB.csv, vcb.csv, VCB_daily.csv, v.v.
    """
    symbol_upper = symbol.upper()
    symbol_lower = symbol.lower()
    patterns = [
        os.path.join(data_dir, f"{symbol_upper}.csv"),
        os.path.join(data_dir, f"{symbol_lower}.csv"),
        os.path.join(data_dir, f"{symbol_upper}_*.csv"),
        os.path.join(data_dir, f"{symbol_lower}_*.csv"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return ""


async def generate_trading_idea(symbol: str) -> str:
    """
    Dùng LLM để tự sinh trading idea cho mã cổ phiếu.
    Người dùng không cần biết gì về tài chính.
    """
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

    prompt = f"""Bạn là chuyên gia phân tích định lượng.
Hãy đề xuất MỘT trading idea ngắn gọn (1-2 câu) để tìm kiếm alpha factor
cho cổ phiếu {symbol} trên thị trường chứng khoán Việt Nam (HOSE).

Yêu cầu:
- Ý tưởng phải có thể triển khai bằng dữ liệu OHLCV + technical indicators
- Cụ thể, không chung chung
- Đề cập loại signal (momentum, mean-reversion, volume, volatility, v.v.)

Chỉ trả về 1-2 câu mô tả ý tưởng, không giải thích thêm."""

    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    idea = response.content.strip()
    log.info(f"[{symbol}] Auto-generated idea: {idea}")
    return idea


async def run_pipeline_for_symbol(
    symbol: str,
    data_dir: str,
    idea: str = "",
    max_iterations: int = 3,
    thread_id: str = "",
) -> dict:
    """Chạy pipeline cho một mã cổ phiếu."""
    from graph import graph
    from state import State

    # Resolve data path
    data_path = resolve_data_path(symbol, data_dir)
    if data_path:
        os.environ["ALPHAGPT_DATA_PATH"] = data_path
        log.info(f"[{symbol}] Data: {data_path}")
    else:
        log.warning(
            f"[{symbol}] Không tìm thấy file CSV trong {data_dir}, "
            f"dùng synthetic data"
        )
        os.environ.pop("ALPHAGPT_DATA_PATH", None)

    # Auto-generate idea nếu không được cung cấp
    trading_idea = idea
    if not trading_idea:
        trading_idea = await generate_trading_idea(symbol)

    # Thread ID mặc định theo symbol
    tid = thread_id or symbol.lower()

    initial_state = State(
        trading_idea=trading_idea,
        max_iterations=max_iterations,
    )
    config = {"configurable": {"thread_id": tid}}

    log.info(f"[{symbol}] Bắt đầu pipeline — {max_iterations} iterations")
    final_state = await graph.ainvoke(initial_state, config)

    # In kết quả
    _print_summary(symbol, trading_idea, final_state)
    return final_state


def _print_summary(symbol: str, idea: str, state: dict) -> None:
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ — {symbol}")
    print(f"{'='*60}")
    print(f"Trading idea : {idea}")
    print(f"Hypothesis   : {state.get('hypothesis', 'N/A')}")
    print(f"Iterations   : {state.get('iteration', 0)}")

    sota = state.get("sota_alphas", [])
    if sota:
        print(f"\nTop {len(sota)} alphas:")
        for a in sota:
            ret = a.get('return_oos')
            ret_str = f"{ret*100:+.1f}%/năm" if ret is not None else "N/A"
            print(
                f"  [{a.get('family','?')}] {a.get('id','?')}\n"
                f"    IC_OOS={a.get('ic_oos','N/A')}  "
                f"Sharpe={a.get('sharpe_oos','N/A')}  "
                f"Return={ret_str}"
                f"    {a.get('description','')}\n"
                f"    expr: {a.get('expression','')[:80]}"
            )
    else:
        print("\nKhông tìm được alpha nào đạt ngưỡng.")

    print(f"\nAnalyst: {state.get('analyst_summary', 'N/A')}")


async def run_all(symbols: list, data_dir: str, idea: str,
                  max_iterations: int) -> None:
    """Chạy tuần tự từng mã — không chạy song song để tránh rate limit."""
    results = {}
    for symbol in symbols:
        try:
            result = await run_pipeline_for_symbol(
                symbol=symbol.upper(),
                data_dir=data_dir,
                idea=idea,
                max_iterations=max_iterations,
            )
            results[symbol] = result
        except Exception as e:
            log.error(f"[{symbol}] Pipeline failed: {e}")
            results[symbol] = {"error": str(e)}

    # Bảng tóm tắt nếu chạy nhiều mã
    if len(symbols) > 1:
        print(f"\n{'='*60}")
        print("TÓM TẮT TOÀN BỘ")
        print(f"{'='*60}")
        for sym, res in results.items():
            if "error" in res:
                print(f"  {sym}: ERROR — {res['error']}")
                continue
            sota = res.get("sota_alphas", [])
            best = max(sota, key=lambda x: x.get("ic_oos") or 0)
            if best:
                ret = best.get('return_oos')
                ret_str = f"{ret*100:+.1f}%" if ret is not None else "N/A"
                print(
                    f"  {sym}: {len(sota)} alphas | "
                    f"best IC_OOS={best.get('ic_oos','N/A')} "
                    f"Return={ret_str}"
                )
            else:
                print(f"  {sym}: không có alpha đạt ngưỡng")


def main():
    parser = argparse.ArgumentParser(
        description="Alpha-GPT — sinh alpha factors cho cổ phiếu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python run.py VCB                        # tự sinh idea, tự tìm data
  python run.py VCB HPG FPT               # chạy 3 mã tuần tự
  python run.py VCB --iterations 5        # tăng số vòng lặp
  python run.py VCB --data-dir ./mydata   # chỉ định thư mục data
  python run.py VCB --idea "RSI momentum" # override idea thủ công
        """,
    )

    parser.add_argument(
        "symbols",
        nargs="+",
        type=str,
        help="Tên mã cổ phiếu (VD: VCB HPG FPT)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        metavar="N",
        help="Số vòng lặp tối đa (mặc định: 3)",
    )
    parser.add_argument(
        "--idea",
        type=str,
        default="",
        metavar="TEXT",
        help="Trading idea thủ công. Nếu bỏ trống, LLM tự sinh.",
    )

    args = parser.parse_args()

    # Tạo data dir nếu chưa có
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

    asyncio.run(run_all(
        symbols=args.symbols,
        data_dir=DEFAULT_DATA_DIR,
        idea=args.idea,
        max_iterations=args.iterations,
    ))


if __name__ == "__main__":
    main()