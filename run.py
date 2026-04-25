"""
run.py — CLI entry point cho Alpha-GPT pipeline.

Usage:
    python run.py --idea "Phân kỳ khối lượng và giá"
    python run.py
    python run.py --data-dir ./my_data
    python run.py --iterations 5
"""
import asyncio
import argparse
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

for _noisy_lib in [
    "httpx", "httpcore", "urllib3",
    "sentence_transformers", "transformers",
    "huggingface_hub", "filelock",
    "openai", "langchain", "langchain_core",
]:
    logging.getLogger(_noisy_lib).setLevel(logging.WARNING)
    
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


async def generate_trading_idea() -> str:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    prompt = f"""Bạn là chuyên gia phân tích định lượng chuyên về thị trường chứng khoán Việt Nam.

Đề xuất MỘT trading idea cực ngắn (5-10 từ) cho thị trường HOSE.
Idea chỉ cần đặt tên một hiện tượng thị trường quan sát được, không giải thích cơ chế.
Chỉ trả về tên hiện tượng, không thêm gì khác.
**Lưu ý**: Dữ liệu chỉ có giá (open, high, low, close, vwap), volume, adv20, obv, returns, sma_5, sma_20, ema_10, rsi_14, macd, macd_signal, bb_upper/middle/lower, momentum_3/10.
Không có dữ liệu khối ngoại, fundamental, earnings, P/E, market cap, news, sentiment.
Hãy đưa ra trading idea sao cho phù hợp với dữ liệu có sẵn, tránh ý tưởng liên quan đến khối ngoại, fundamental, news, sentiment.
"""
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    idea = response.content.strip()
    log.info(f"[Run] Auto-generated idea: {idea}")
    return idea


async def run_pipeline(
    data_dir: str,
    idea: str = "",
    max_iterations: int = 3,
    thread_id: str = "main",
) -> dict:
    from graph import graph
    from state import State

    os.environ["ALPHAGPT_DATA_DIR"] = data_dir

    trading_idea = idea or await generate_trading_idea()

    initial_state = State(
        trading_idea=trading_idea,
        max_iterations=max_iterations,
    )
    config = {"configurable": {"thread_id": thread_id}}

    log.info(f"[Run] Trading idea: {trading_idea}")
    log.info(f"[Run] Data dir: {data_dir}")
    log.info(f"[Run] Max iterations: {max_iterations}")

    final_state = await graph.ainvoke(initial_state, config)
    _print_summary(trading_idea, final_state)
    return final_state


def _print_summary(idea: str, state: dict) -> None:
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ ALPHA-GPT")
    print(f"{'='*60}")
    print(f"Trading idea : {idea}")
    print(f"Total Iterations : {state.get('iteration', 0)}\n")

    # Duyệt qua lịch sử để in báo cáo của từng vòng
    history = state.get("hypothesis_history", [])
    if history:
        print("--- LỊCH SỬ CÁC VÒNG LẶP ---")
        for h in history:
            print(f"\n[Vòng {h.get('iteration', '?')}]")
            print(f"  Hypothesis: {h.get('hypothesis', 'N/A')}")
            print(f"  Summary   : {h.get('alpha_summary', 'N/A')}")
            
            # Lấy nhận xét của Analyst nếu có
            analyst_data = h.get('analyst', {})
            if isinstance(analyst_data, dict):
                print(f"  Analyst   : {analyst_data.get('overall_assessment', 'N/A')}")
    else:
        print(f"Hypothesis   : {state.get('hypothesis', 'N/A')}")
        print(f"Analyst      : {state.get('analyst_summary', 'N/A')}")

    sota = state.get("sota_alphas", [])
    if sota:
        print(f"\nTop {len(sota)} alphas:")
        for a in sota:
            ret = a.get("return_oos")
            ret_str = f"{ret*100:+.1f}%/năm" if ret is not None else "N/A"
            print(
                f"  [{a.get('family','?')}] {a.get('id','?')}\n"
                f"    IC_OOS={a.get('ic_oos','N/A')}  "
                f"Sharpe={a.get('sharpe_oos','N/A')}  "
                f"Return={ret_str}\n"
                f"    {a.get('description','')}\n"
                f"    expr: {a.get('expression','')[:80]}"
            )
    else:
        print("\nKhông tìm được alpha nào đạt ngưỡng.")


def main():
    parser = argparse.ArgumentParser(
        description="Alpha-GPT — tìm alpha factors trên toàn bộ universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python run.py                                    # tự sinh idea
  python run.py --idea "Phân kỳ khối lượng và giá"
  python run.py --iterations 5
  python run.py --data-dir ./mydata
        """,
    )
    parser.add_argument("--idea", type=str, default="", metavar="TEXT",
                        help="Trading idea. Nếu bỏ trống, LLM tự sinh.")
    parser.add_argument("--iterations", type=int, default=3, metavar="N",
                        help="Số vòng lặp tối đa (mặc định: 3)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, metavar="DIR",
                        help="Thư mục chứa CSV files")
    parser.add_argument("--thread-id", type=str, default="main", metavar="ID",
                        help="Thread ID cho checkpoint (mặc định: main)")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    asyncio.run(run_pipeline(
        data_dir=args.data_dir,
        idea=args.idea,
        max_iterations=args.iterations,
        thread_id=args.thread_id,
    ))


if __name__ == "__main__":
    main()