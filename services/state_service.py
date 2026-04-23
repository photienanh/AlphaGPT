"""
services/state_service.py
Cung cấp hai hàm tiện ích để chạy pipeline và truy vấn lịch sử.

Thay đổi so với phiên bản cũ:
  - Bỏ import get_checkpoint_manager từ checkpointer_api (PostgreSQL)
  - Dùng AlphaGPTDB (SQLite) từ database.db
  - checkpoint_id không còn cần thiết (MemorySaver tự quản lý)
  - get_state_history trả về cấu trúc đầy đủ: hypotheses → alphas → backtest_results
"""
import asyncio
from typing import Dict, Any, Optional

from state import State
from database.db import get_db


def invoke_graph_with_state(
    initial_state: State,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Chạy pipeline đồng bộ từ initial_state.

    Args:
        initial_state: State khởi đầu, bắt buộc có trading_idea.
        thread_id: ID để checkpoint MemorySaver theo dõi. Nếu None thì dùng "default".

    Returns:
        Final state dict sau khi pipeline hoàn tất.
    """
    from graph import graph  # lazy import tránh circular

    config = {"configurable": {"thread_id": thread_id or "default"}}
    return graph.invoke(initial_state, config)


async def ainvoke_graph_with_state(
    initial_state: State,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Phiên bản async của invoke_graph_with_state.
    Dùng khi gọi từ môi trường async (FastAPI, Jupyter, v.v.)
    """
    from graph import graph  # lazy import tránh circular

    config = {"configurable": {"thread_id": thread_id or "default"}}
    return await graph.ainvoke(initial_state, config)


def get_state_history(thread_id: str) -> Dict[str, Any]:
    """
    Truy vấn toàn bộ lịch sử của một thread từ SQLite.

    Cấu trúc trả về:
        {
            "thread_id": str,
            "hypotheses": [
                {
                    "id": int,
                    "iteration": int,
                    "hypothesis": str,
                    ...các fields khác của hypotheses table...,
                    "alphas": [
                        {
                            "id": int,
                            "alpha_id": str,
                            "expression": str,
                            "ic_oos": float,
                            "score": float,
                            ...
                            "backtest_results": [...]
                        }
                    ]
                }
            ]
        }
    """
    db = get_db()
    hypotheses = db.get_hypothesis_history(thread_id)

    for hyp in hypotheses:
        alphas = db.get_alphas_for_hypothesis(hyp["id"])
        for alpha in alphas:
            alpha["backtest_results"] = db.get_backtest_results_for_alpha(alpha["id"])
        hyp["alphas"] = alphas

    return {"thread_id": thread_id, "hypotheses": hypotheses}