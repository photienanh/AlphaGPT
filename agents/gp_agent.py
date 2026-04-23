# agents/gp_agent.py
"""
GP Enhancement agent — paper Section 2.2, Alpha Compute Framework.
Chạy GP trên seed alphas để tạo candidate population.
Fitness = IC_IS (fast) — không dùng OOS để tránh overfit vào test split.
"""
import logging
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from state import State
from gp_search import enhance_population
from evaluator import eval_alpha

log = logging.getLogger(__name__)
GP_ITERATIONS = 15


async def gp_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Nhận seed_alphas, chạy GP, trả về candidate_alphas.
    """
    from graph import DATA_STORE
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    data = DATA_STORE.get(thread_id)
    if data is None:
        log.warning("[GP] No data in DATA_STORE, returning seeds as candidates")
        return {"candidate_alphas": state.seed_alphas}
 
    df, fwd_ret = data

    # Eval seeds trước để GP có baseline
    seeds_evaled = []
    for seed in state.seed_alphas:
        result = eval_alpha(seed, df, fwd_ret, full=False)
        result["id"]          = seed.get("id", result.get("id", ""))
        result["description"] = seed.get("description", "")
        result["family"]      = seed.get("family", "")
        result["expression"]  = seed.get("expression", result.get("expression", ""))
        seeds_evaled.append(result)

    log.info(f"[GP] Enhancing {len(seeds_evaled)} seeds, {GP_ITERATIONS} iterations each")

    def _eval_fn(alpha_def, df, fwd_ret, full=False):
        return eval_alpha(alpha_def, df, fwd_ret, full=full)

    candidates = enhance_population(
        seeds_evaled, df, fwd_ret,
        eval_fn=_eval_fn,
        n_iterations=GP_ITERATIONS,
    )

    log.info(f"[GP] Done: produced {len(candidates)} candidates")

    # Strip series (không serialize được)
    for c in candidates:
        c.pop("series", None)

    return {"candidate_alphas": candidates}