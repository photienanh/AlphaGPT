# agents/gp_agent.py
"""
GP Enhancement agent — paper Section 2.2, Alpha Compute Framework.
Fitness = cross-sectional IC trên toàn bộ universe.
Nhất quán với final backtest metric.
"""
import logging
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from state import State
from gp_search import enhance_alpha
from config import DEFAULT_CONFIG

log = logging.getLogger(__name__)


async def gp_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    from graph import DATA_STORE
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    data = DATA_STORE.get(thread_id)
    if data is None:
        log.warning("[GP] No data in DATA_STORE, returning seeds as candidates")
        return {"candidate_alphas": state.seed_alphas}

    _, df_by_ticker, forward_return = data

    if not df_by_ticker:
        return {"candidate_alphas": state.seed_alphas}

    total_days = len(forward_return.index)
    train_days = int(total_days * (1 - DEFAULT_CONFIG.test_ratio))

    log.info(
        f"[GP] Enhancing {len(state.seed_alphas)} seeds × "
        f"{DEFAULT_CONFIG.gp_iterations} iterations "
        f"trên {train_days}/{total_days} ngày dữ liệu"
    )

    candidates = enhance_alpha(
        state.seed_alphas,
        df_by_ticker,
        forward_return,
        n_iterations=DEFAULT_CONFIG.gp_iterations,
    )

    log.info(f"[GP] Done: produced {len(candidates)} candidates")

    return {"candidate_alphas": candidates}