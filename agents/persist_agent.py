# agents/persist_agent.py
"""
Persist agent — lưu kết quả vào SQLite và alpha_library.json.
alpha_library.json: file chung tích lũy tất cả alpha OK qua mọi run,
đánh số thứ tự toàn cục, dedup theo expression.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from state import State
from database.db import get_db

log = logging.getLogger(__name__)

LIBRARY_PATH = os.environ.get("ALPHA_LIBRARY_PATH", "data/alpha_library.json")


def _load_library(path: str) -> List[Dict]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_library(path: str, library: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)


def _normalize_expr(expr: str) -> str:
    """Normalize expression để dedup."""
    return " ".join(expr.split()).lower()

def _append_to_library(sota_alphas: List[Dict],
                       hypothesis: str) -> int:
    library = _load_library(LIBRARY_PATH)
    existing_exprs = {
        _normalize_expr(a.get("expression", ""))
        for a in library
    }

    next_no = len(library) + 1
    added   = 0

    for a in sota_alphas:
        expr = a.get("expression", "")
        if not expr:
            continue

        # Dedup theo expression
        if _normalize_expr(expr) in existing_exprs:
            log.debug(f"[Persist] Duplicate expression, skip: {a.get('id')}")
            continue

        ic  = a.get("ic_oos")
        ret = a.get("return_oos")

        entry = {
            "no":          next_no,
            "id":          a.get("id", f"alpha_{next_no}"),
            "description": a.get("description", ""),
            "expression":  expr,
            "ic_oos":      round(float(ic), 6)  if ic  is not None else None,
            "sharpe_oos":  round(float(a.get("sharpe_oos", 0) or 0), 4),
            "return_oos":  round(float(ret), 4) if ret is not None else None,
            "hypothesis":  hypothesis,
            "saved_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        library.append(entry)
        existing_exprs.add(_normalize_expr(expr))
        next_no += 1
        added   += 1

    if added > 0:
        _save_library(LIBRARY_PATH, library)

    return added


async def persist_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    db = get_db()

    try:
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

        sota_ids = {a.get("id") for a in (state.sota_alphas or [])}
        n_sota = 0
        for alpha in state.evaluated_alphas:
            alpha_db_id = db.save_alpha(thread_id, hyp_id, alpha)
            is_sota = alpha.get("id") in sota_ids
            db.save_backtest(thread_id, alpha_db_id, alpha, is_sota=is_sota)
            if is_sota:
                n_sota += 1

        log.info(
            f"[Persist] Saved iteration {state.iteration} → DB "
            f"(hyp_id={hyp_id}, {n_sota} sota alphas)"
        )

        # Log sota alphas iteration này
        for a in (state.sota_alphas or []):
            ic  = a.get("ic_oos")
            sharpe = a.get("sharpe_oos")
            ret = a.get("return_oos")
            if ic is not None and ret is not None:
                log.info(
                    f"  ✓ {a.get('id','?')} "
                    f"IC_OOS={ic:+.4f} Sharpe={sharpe:+.3f} Return={ret*100:+.1f}%/năm"
                )

    except Exception as e:
        log.error(f"[Persist] DB save failed: {e}")

    # Append alpha OK vào alpha_library.json
    if state.sota_alphas:
        try:
            added = _append_to_library(
                state.sota_alphas,
                hypothesis=state.hypothesis,
            )
            if added > 0:
                library = _load_library(LIBRARY_PATH)
                log.info(
                    f"[Persist] alpha_library.json: +{added} alpha mới "
                    f"(tổng {len(library)})"
                )
                # Invalidate FAISS cache — lần retrieve tiếp theo sẽ rebuild index
                try:
                    from knowledge.retriever import invalidate_cache
                    invalidate_cache()
                except Exception:
                    pass
            else:
                log.info("[Persist] alpha_library.json: không có alpha mới (đã có trong library)")
        except Exception as e:
            log.error(f"[Persist] Ghi alpha_library.json thất bại: {e}")

    return {}