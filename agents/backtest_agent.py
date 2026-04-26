# agents/backtest_agent.py
"""
Trading Backtest Engine — paper Section 2.3.
Full cross-sectional evaluation trên toàn universe.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from langchain_core.runnables import RunnableConfig
from state import State
from evaluator import eval_alpha
from validators import normalize_expression
from config import DEFAULT_CONFIG

log = logging.getLogger(__name__)


def _select_sota(evaluated: List[Dict]) -> List[Dict]:
    def _is_stable(a: Dict) -> bool:
        ic_is  = a.get("ic_is")
        ic_oos = a.get("ic_oos")
        if ic_is is None or ic_oos is None:
            return True
        return ic_is * ic_oos >= 0

    def _passes_quality(a: Dict) -> bool:
        ic_oos = a.get("ic_oos")
        sharpe = a.get("sharpe_oos")
        ret    = a.get("return_oos")
        if ic_oos is None or sharpe is None or ret is None:
            return False
        return (ic_oos >= DEFAULT_CONFIG.ic_signal_threshold
                and sharpe > DEFAULT_CONFIG.sharpe_min_threshold
                and ret   > DEFAULT_CONFIG.return_min_threshold)

    ok = [a for a in evaluated
          if a.get("status") == "OK" and _is_stable(a) and _passes_quality(a)]
    ok.sort(key=lambda x: (x.get("ic_oos") or 0, x.get("sharpe_oos") or 0), reverse=True)

    seen_exprs = set()
    deduped = []
    for a in ok:
        norm = normalize_expression(a.get("expression", ""))
        if norm not in seen_exprs:
            seen_exprs.add(norm)
            deduped.append(a)

    selected = []
    for cand in deduped:
        if len(selected) >= DEFAULT_CONFIG.max_sota:
            break

        s_cand  = cand.get("_series")
        corr_ok = True

        for sel in selected:
            s_sel = sel.get("_series")
            if s_cand is None or s_sel is None:
                continue
            try:
                merged = pd.concat(
                    [s_cand.stack(), s_sel.stack()], axis=1
                ).dropna()
                if len(merged) >= 50:
                    cv = abs(merged.iloc[:, 0].corr(
                        merged.iloc[:, 1], method="spearman"
                    ))
                    if cv >= DEFAULT_CONFIG.corr_threshold:
                        corr_ok = False
                        break
            except Exception:
                pass

        if corr_ok:
            selected.append(cand)

    for a in selected:
        a.pop("_series",    None)
        a.pop("_ic_series", None)

    return selected


async def backtest_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Full cross-sectional evaluation trên candidate pool."""
    from graph import DATA_STORE

    thread_id = config.get("configurable", {}).get("thread_id", "default")
    data = DATA_STORE.get(thread_id)
    if data is None:
        log.warning("[Backtest] No data in DATA_STORE, skipping")
        return {"evaluated_alphas": state.candidate_alphas, "sota_alphas": []}

    _, ticker_dfs, fwd_ret_multi = data

    evaluated = []
    for cand in state.candidate_alphas:
        result = eval_alpha(cand, ticker_dfs, fwd_ret_multi, full=True)
        result["id"]          = cand.get("id", "")
        result["description"] = cand.get("description", "")
        expression            = result.get("expression", "")
        evaluated.append(result)

        status = result.get("status", "ERR")
        ic_oos = result.get("ic_oos") or 0.0
        sharpe_oos = result.get("sharpe_oos") or 0.0
        return_oos    = result.get("return_oos") or 0.0
        if status == "OK":
            log.info(
                f"  [OK] {result['id']}: {expression}\n"
                f"\t\t\tIC_OOS={ic_oos:+.4f} - Sharpe_OOS={sharpe_oos:+.4f} - Return_OOS={return_oos:+.2%}"
            )
        elif status == "WEAK":
            log.info(
                f"  [WEAK] {result['id']}: {expression}\n"
                f"\t\t\tIC_OOS={ic_oos:+.4f} - Sharpe_OOS={sharpe_oos:+.4f} - Return_OOS={return_oos:+.2%}\n"
                f"\t\t\tWeak reason: {result.get('weak_reason','')}"
            )
        else:
            log.info(f"  [ERR] {result['id']} — {result.get('error','')[:60]}")

    sota = _select_sota(evaluated)

    for a in evaluated:
        a.pop("_series", None)

    n_ok   = sum(1 for a in evaluated if a.get("status") == "OK")
    n_weak = sum(1 for a in evaluated if a.get("status") == "WEAK")
    n_err  = sum(1 for a in evaluated if a.get("status") == "EVAL_ERROR")
    log.info(
        f"[Backtest] OK={n_ok} WEAK={n_weak} ERR={n_err} "
        f"| {len(sota)} sota selected"
    )

    # Cộng dồn sota từ các vòng trước với sota mới, dedup theo id, giữ top max_sota
    existing_sota = {a["id"]: a for a in (state.sota_alphas or [])}
    for a in sota:
        existing_sota[a["id"]] = a

    all_sota = sorted(existing_sota.values(), key=lambda x: x.get("ic_oos") or 0, reverse=True)
    accumulated_sota = all_sota[:DEFAULT_CONFIG.max_sota]

    return {
        "evaluated_alphas": evaluated,
        "sota_alphas":      accumulated_sota,
    }