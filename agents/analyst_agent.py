# agents/analyst_agent.py
"""
Analyst agent — paper Section 2.3 (Review).
"""
import json
import logging
import numpy as np
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from state import State
from prompts.analyst_prompts import ANALYST_SYSTEM_PROMPT, ANALYST_PROMPT
from config import DEFAULT_CONFIG

log = logging.getLogger(__name__)


def _classify_eval_error(err: str) -> str:
    e = (err or "").lower()
    if not e:
        return "unknown"
    if "takes" in e and "positional argument" in e:
        return "operator_signature_error"
    if "keyerror" in e or "not in index" in e:
        return "missing_field_error"
    if "syntaxerror" in e or "invalid syntax" in e:
        return "syntax_error"
    if "nameerror" in e:
        return "unknown_symbol_error"
    if "did not produce pd.series" in e:
        return "invalid_output_error"
    if "validation:" in e:
        return "validation_error"
    return "runtime_error"


async def analyst_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    ok_alphas   = [a for a in state.evaluated_alphas if a.get("status") == "OK"]
    weak_alphas = [a for a in state.evaluated_alphas if a.get("status") == "WEAK"]
    err_alphas  = [a for a in state.evaluated_alphas if a.get("status") == "EVAL_ERROR"]

    avg_ic_oos = (np.mean([abs(a.get("ic_oos") or 0) for a in ok_alphas]) if ok_alphas else 0.0)
    avg_sharpe = (np.mean([a.get("sharpe_oos") or 0 for a in ok_alphas]) if ok_alphas else 0.0)
    avg_return = (np.mean([a.get("return_oos") or 0 for a in ok_alphas]) if ok_alphas else 0.0)

    results_text = []
    for a in ok_alphas:
        ret_str = (f"{a.get('return_oos', 0)*100:+.1f}%" if a.get("return_oos") is not None else "N/A")
        results_text.append(
            f"- {a['id']} [{a.get('family','?')}] status=OK\n"
            f"  IC_IS={a.get('ic_is',0):+.4f}  IC_OOS={a.get('ic_oos',0):+.4f}  "
            f"Sharpe={a.get('sharpe_oos',0):+.3f}  Return={ret_str}  "
            f"Turnover={a.get('turnover',0):.3f}\n"
            f"  {a.get('description','')[:80]}"
        )
    for a in weak_alphas:
        results_text.append(
            f"- {a['id']} [{a.get('family', '?')}] status=WEAK\n"
            f"  IC_OOS={a.get('ic_oos', 0):+.4f} — {a.get('weak_reason', '')}\n"
            f"  expression: {a.get('expression', '')[:80]}"
        )
    for a in err_alphas:
        results_text.append(f"- {a['id']} [EVAL_ERROR]: {a.get('error', '')[:60]}")

    error_groups = {}
    for a in err_alphas:
        category = _classify_eval_error(a.get("error", ""))
        error_groups.setdefault(category, []).append(a.get("id", "?"))
    if error_groups:
        diag_lines = [f"- {cat}: {', '.join(ids)}" for cat, ids in sorted(error_groups.items())]
        results_text.append(
            "## Error diagnostics\n" + "\n".join(diag_lines)
        )

    prompt = ANALYST_PROMPT.format(
        round_num=state.iteration,
        alpha_results="\n".join(results_text),
        n_ok=len(ok_alphas),
        n_weak=len(weak_alphas),
        n_err=len(err_alphas),
        n_total=len(state.evaluated_alphas),
        avg_ic_oos=avg_ic_oos,
        avg_sharpe=avg_sharpe,
        avg_return=avg_return * 100,
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        content = response.content
        j_start = content.find("{")
        j_end   = content.rfind("}") + 1
        data = json.loads(content[j_start:j_end] if j_start >= 0 else content)
    except Exception as e:
        log.warning(f"[Analyst] LLM failed: {e}")
        data = {
            "overall_assessment": "Analysis unavailable",
            "market_behavior": "unknown",
            "alpha_analyses": [],
            "weak_alpha_ids": [a["id"] for a in weak_alphas + err_alphas],
            "refinement_directions": [],
            "polisher_feedback": "",
            "round_summary": f"Round {state.iteration} completed",
        }

    summary = data.get("round_summary", "")
    log.info(f"[Analyst] Round {state.iteration}: {summary}")

    current_hyp = {
        "iteration":     state.iteration,
        "hypothesis":    state.hypothesis,
        "alpha_summary": (
            f"IC_OOS={avg_ic_oos:.4f} Sharpe={avg_sharpe:.3f} "
            f"Return={avg_return*100:+.1f}% OK={len(ok_alphas)}/{len(state.evaluated_alphas)}"
        ),
        "weak_alpha_ids": data.get("weak_alpha_ids", []),
        "analyst":        data,
    }
    updated_history = list(state.hypothesis_history) + [current_hyp]

    sota_count = len(state.sota_alphas or [])
    quality_ok = (
        sota_count >= DEFAULT_CONFIG.min_sota
        and avg_ic_oos >= DEFAULT_CONFIG.min_ic
        and avg_sharpe >= DEFAULT_CONFIG.min_sharpe
        and avg_return >= DEFAULT_CONFIG.min_return
    )
    weak_ids = data.get("weak_alpha_ids", [])
    should_continue = (
        state.iteration < state.max_iterations
        and not quality_ok
        and len(weak_ids) > 0
    )

    return {
        "analyst_summary":    data.get("overall_assessment", ""),
        "analyst_feedback":   data.get("polisher_feedback", ""),
        "analyst_weak_ids":   weak_ids,
        "hypothesis_history": updated_history,
        "should_continue":    should_continue,
    }