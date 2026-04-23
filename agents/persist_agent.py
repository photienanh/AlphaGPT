# agents/persist_agent.py
"""
Persist agent — lưu kết quả vòng hiện tại vào SQLite.
Chạy sau Analyst, trước khi quyết định loop hay dừng.
"""
import logging
import os
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from state import State
from database.db import get_db

log = logging.getLogger(__name__)


async def persist_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Lưu hypothesis + evaluated alphas + backtest results vào SQLite."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    db = get_db()

    try:
        # Lưu hypothesis
        hyp_id = db.save_hypothesis(thread_id, {
            "trading_idea":          state.trading_idea,
            "hypothesis":            state.hypothesis,
            "reason":                state.reason,
            "concise_reason":        state.concise_reason,
            "concise_observation":   state.concise_observation,
            "concise_justification": state.concise_justification,
            "concise_knowledge":     state.concise_knowledge,
            "iteration":             state.iteration,
        })

        # Lưu từng alpha và backtest result
        sota_ids = {a.get("id") for a in (state.sota_alphas or [])}
        for alpha in state.evaluated_alphas:
            alpha_db_id = db.save_alpha(thread_id, hyp_id, alpha)
            db.save_backtest(
                thread_id, alpha_db_id, alpha,
                is_sota=(alpha.get("id") in sota_ids),
            )

        log.info(f"[Persist] Saved iteration {state.iteration} → DB (hyp_id={hyp_id})")
    except Exception as e:
        log.error(f"[Persist] DB save failed: {e}")

    return {}