# agents/alpha_generator_agent.py
"""
Quant Developer agent — paper Section 2.2 (Implementation).
RAG từ knowledge base, fallback từ KB thay vì hardcode.
"""
import json
import logging
import random
from typing import Any, Dict, List
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from state import State
from prompts.alpha_prompts import (
    ALPHA_SYSTEM_PROMPT,
    ALPHA_INITIAL_PROMPT,
    ALPHA_ITERATION_PROMPT,
    OPERATOR_SIGNATURES,
    DATA_FIELDS_BLOCK,
)
from knowledge.retriever import retrieve_similar_alphas, load_alpha_kb
from config import DEFAULT_CONFIG

log = logging.getLogger(__name__)
NUM_FACTORS_INITIAL    = 5
NUM_FACTORS_REFINEMENT = 3


def _get_fallback_alphas(n: int) -> List[Dict]:
    """Load fallback alphas từ knowledge base thay vì hardcode."""
    kb = load_alpha_kb()
    if not kb:
        return []
    sampled = random.sample(kb, min(n, len(kb)))
    # Đảm bảo format đúng cho pipeline
    result = []
    for a in sampled:
        result.append({
            "id": a.get("id", "fallback"),
            "family": a.get("family", "unknown"),
            "description": a.get("description", ""),
            "expression": a.get("expression", ""),
        })
    return result


def _format_rag_examples(alphas: list) -> str:
    if not alphas:
        return ""
    lines = ["\n## Example alphas from knowledge base (for reference, không copy trực tiếp)\n"]
    for a in alphas:
        lines.append(
            f"- {a['id']} [{a.get('family', '?')}]: {a.get('description', '')[:80]}\n"
            f"  expression: `{a.get('expression', '')[:100]}`"
        )
    return "\n".join(lines)


async def alpha_generator_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate seed alphas từ hypothesis — output là expression strings."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    is_first = not state.alpha_history

    # RAG: retrieve examples liên quan đến hypothesis
    query = state.hypothesis or state.trading_idea
    rag_alphas = retrieve_similar_alphas(query, top_k=DEFAULT_CONFIG.rag_top_k)
    rag_block = _format_rag_examples(rag_alphas)

    if is_first:
        prompt = ALPHA_INITIAL_PROMPT.format(
            hypothesis=state.hypothesis,
            num_factors=NUM_FACTORS_INITIAL,
            data_fields=DATA_FIELDS_BLOCK,
            operators=OPERATOR_SIGNATURES,
            rag_examples=rag_block,
        )
    else:
        weak_ids = set(state.analyst_weak_ids or [])
        if not weak_ids:
            weak_ids = {
                a["id"] for a in state.evaluated_alphas
                if a.get("status") in ("WEAK", "EVAL_ERROR")
            }

        weak_alphas = [a for a in state.evaluated_alphas if a.get("id") in weak_ids]
        good_alphas = [a for a in (state.sota_alphas or []) if a.get("id") not in weak_ids]

        weak_text = "\n".join(
            _format_alpha_for_prompt(a, show_reason=True) for a in weak_alphas
        ) or "Không có"
        good_text = "\n".join(
            _format_alpha_for_prompt(a, show_reason=False) for a in good_alphas
        ) or "Chưa có"

        num_to_generate = max(len(weak_ids), NUM_FACTORS_REFINEMENT)

        prompt = ALPHA_ITERATION_PROMPT.format(
            hypothesis=state.hypothesis,
            analyst_feedback=state.analyst_feedback or "",
            weak_alphas=weak_text,
            good_alphas=good_text,
            num_factors=num_to_generate,
            data_fields=DATA_FIELDS_BLOCK,
            operators=OPERATOR_SIGNATURES,
            rag_examples=rag_block,
        )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": ALPHA_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        content = response.content
        j_start = content.find("{")
        j_end   = content.rfind("}") + 1
        data    = json.loads(content[j_start:j_end] if j_start >= 0 else content)
        alphas  = data.get("alphas", [])
        log.info(f"[AlphaGenerator] Generated {len(alphas)} seed alphas")
    except Exception as e:
        log.warning(f"[AlphaGenerator] LLM failed: {e}, using fallbacks from KB")
        n = NUM_FACTORS_INITIAL if is_first else NUM_FACTORS_REFINEMENT
        alphas = _get_fallback_alphas(n)

    # Force unique ID theo iteration để tránh trùng ID giữa các round
    # state.iteration đã được hypothesis_agent cập nhật trước đó
    iter_prefix = f"r{state.iteration}"
    for i, a in enumerate(alphas):
        raw_id = a.get("id", f"alpha_{i+1}")
        if not raw_id.startswith(iter_prefix):
            a["id"] = f"{iter_prefix}_{raw_id}"

    updated_history = list(state.alpha_history) + [{
        "iteration": state.iteration,
        "count":     len(alphas),
    }]

    return {
        "seed_alphas":   alphas,
        "alpha_history": updated_history,
    }


def _format_alpha_for_prompt(a: dict, show_reason: bool) -> str:
    ic  = a.get("ic_oos")
    ret = a.get("return_oos")
    sh  = a.get("sharpe_oos")

    metrics = []
    if ic  is not None: metrics.append(f"IC_OOS={ic:+.4f}")
    if ret is not None: metrics.append(f"Return={ret*100:+.1f}%")
    if sh  is not None: metrics.append(f"Sharpe={sh:+.3f}")

    line = f"- {a.get('id','?')} [{a.get('family','?')}]: {a.get('description','')[:60]}"
    if metrics:
        line += f" | {' '.join(metrics)}"
    if show_reason and a.get("weak_reason"):
        line += f"\n  Lý do yếu: {a.get('weak_reason','')}"
    elif show_reason and a.get("error"):
        line += f"\n  Lỗi: {a.get('error','')[:60]}"
    return line