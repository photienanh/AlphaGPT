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

    ok_alphas   = [alpha for alpha in state.evaluated_alphas if alpha.get("status") == "OK"]
    weak_alphas = [alpha for alpha in state.evaluated_alphas if alpha.get("status") == "WEAK"]
    err_alphas  = [alpha for alpha in state.evaluated_alphas if alpha.get("status") == "EVAL_ERROR"]

    results_text = []
    for alpha in ok_alphas:
        return_str = (f"{alpha.get('return_oos', 0)*100:+.1f}%" if alpha.get("return_oos") is not None else "N/A")
        results_text.append(
            f"- {alpha['id']} status=OK\n"
            f"  IC_IS={alpha.get('ic_is',0):+.4f}  IC_OOS={alpha.get('ic_oos',0):+.4f}  "
            f"Sharpe={alpha.get('sharpe_oos',0):+.3f}  Return={return_str}  "
            f"Turnover={alpha.get('turnover',0):.3f}\n"
            f"  {alpha.get('description','')[:80]}"
        )
    for alpha in weak_alphas:
        return_str = (f"{alpha.get('return_oos', 0)*100:+.1f}%" if alpha.get("return_oos") is not None else "N/A")
        results_text.append(
            f"- {alpha['id']} status=WEAK\n"
            f"  IC_OOS={alpha.get('ic_oos', 0):+.4f}  "
            f"Sharpe={alpha.get('sharpe_oos',0):+.3f}  Return={return_str}  "
            f"Turnover={alpha.get('turnover',0):.3f}\n"
            f"  Weak Reason: {alpha.get('weak_reason', '')}\n"
            f"  expression: {alpha.get('expression', '')[:80]}"
        )
    for alpha in err_alphas:
        results_text.append(f"- {alpha['id']} [EVAL_ERROR]: {alpha.get('error', '')[:60]}")

    error_groups = {}
    for alpha in err_alphas:
        category = _classify_eval_error(alpha.get("error", ""))
        error_groups.setdefault(category, []).append(alpha.get("id", "?"))
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
            "alpha_analyses": [],
            "weak_alpha_ids": [a["id"] for a in weak_alphas + err_alphas],
            "refinement_directions": [],
            "polisher_feedback": "",
            "round_summary": f"Round {state.iteration} completed",
        }

    round_summary = data.get("round_summary", "")
    weak_ids = data.get("weak_alpha_ids", [])
    data.pop("weak_alpha_ids", None)
    log.info(f"[Analyst Summary] Round {state.iteration}: {round_summary}")

    alpha_summary = ""
    for alpha in data.get("alpha_analyses", []):
        alpha_summary += f"Alpha id: {alpha.get('alpha_id', '?')}: {alpha.get('status', '?')} - Phân tích: {alpha.get('explanation', '')}\n"

    refinement_directions = ""
    for direction in data.get("refinement_directions", []):
        refinement_directions += f"   - {direction}\n"

    current_hyp = {
        "iteration":     state.iteration,
        "hypothesis":    state.hypothesis,
        "alpha_summary": alpha_summary,
        "round_summary": round_summary,
    }
    updated_history = list(state.hypothesis_history) + [current_hyp]

    def _passes_quality(alpha: Dict[str, Any]) -> bool:
        ic_oos = alpha.get("ic_oos")
        sharpe = alpha.get("sharpe_oos")
        ret = alpha.get("return_oos")
        if ic_oos is None or sharpe is None or ret is None:
            return False
        return (
            ic_oos >= DEFAULT_CONFIG.ic_signal_threshold
            and sharpe > DEFAULT_CONFIG.sharpe_min_threshold
            and ret > DEFAULT_CONFIG.return_min_threshold
        )

    sota_alphas = state.sota_alphas or []
    sota_count = len(sota_alphas)
    quality_ok = (
        sota_count >= DEFAULT_CONFIG.min_sota
        and all(_passes_quality(alpha) for alpha in sota_alphas)
    )
    
    should_continue = (
        state.iteration < state.max_iterations
        and not quality_ok
    )

    return {
        "analyst_summary": data.get("overall_assessment", ""),
        "refinement_directions": refinement_directions,
        "analyst_weak_ids": weak_ids,
        "hypothesis_history": updated_history,
        "should_continue": should_continue,
    }