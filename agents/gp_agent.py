# agents/gp_agent.py
"""
GP Enhancement agent — paper Section 2.2, Alpha Compute Framework.
Fitness = cross-sectional IC trên sampled universe (GP_SAMPLE_SIZE tickers).
Nhất quán với final backtest metric.
"""
import logging
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from state import State
from gp_search import enhance_population, GP_SAMPLE_SIZE
from config import DEFAULT_CONFIG

log = logging.getLogger(__name__)


async def gp_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    from graph import DATA_STORE
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    data = DATA_STORE.get(thread_id)
    if data is None:
        log.warning("[GP] No data in DATA_STORE, returning seeds as candidates")
        return {"candidate_alphas": state.seed_alphas}

    _, ticker_dfs, fwd_ret_multi = data

    if not ticker_dfs:
        return {"candidate_alphas": state.seed_alphas}

    n_tickers = len(ticker_dfs)
    sample_size = min(GP_SAMPLE_SIZE, n_tickers)
    log.info(
        f"[GP] Enhancing {len(state.seed_alphas)} seeds × "
        f"{DEFAULT_CONFIG.gp_iterations} iterations "
        f"| fitness: cross-sectional IC on {sample_size}/{n_tickers} tickers"
    )

    candidates = enhance_population(
        state.seed_alphas,
        ticker_dfs,
        fwd_ret_multi,
        n_iterations=DEFAULT_CONFIG.gp_iterations,
    )

    log.info(f"[GP] Done: produced {len(candidates)} candidates")

    for c in candidates:
        c.pop("series", None)

    return {"candidate_alphas": candidates}