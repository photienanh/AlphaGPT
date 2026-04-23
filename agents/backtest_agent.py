# agents/backtest_agent.py
"""
Trading Backtest Engine — paper Section 2.3.
Chạy full evaluation (IC_IS, IC_OOS, Sharpe_OOS, turnover) trên candidate pool.
Đây là full metrics để báo cáo cho Analyst — KHÁC với GP fitness (chỉ IC_IS).
"""
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Any, Dict, List
from langchain_core.runnables import RunnableConfig
from state import State
from evaluator import eval_alpha

log = logging.getLogger(__name__)
CORR_THRESHOLD      = 0.55
MAX_SOTA            = 5


def _select_sota(evaluated: List[Dict]) -> List[Dict]:
    """
    Greedy selection: chỉ chọn alpha có status="OK" (IC_OOS > 0).
    MARGINAL, WEAK và EVAL_ERROR không được chọn — Analyst sẽ dùng thông tin
    đó để feedback cho generator sửa lại.
    """
    # Loại alpha có IC_IS và IC_OOS quá chênh lệch (dấu hiệu thiếu ổn định)
    def _is_stable(a: Dict[str, Any]) -> bool:
        ic_is = a.get("ic_is")
        ic_oos = a.get("ic_oos")
        if ic_is is None or ic_oos is None:
            return True
        if ic_is * ic_oos < 0 and abs(ic_is) > 0.05:
            return False
        return True

    ok = [a for a in evaluated
          if a.get("status") == "OK" and _is_stable(a)]
    ok.sort(key=lambda x: (
        x.get("ic_oos") or 0,
        x.get("sharpe_oos") or 0,
    ), reverse=True)

 
    selected = []
    for cand in ok:
        if len(selected) >= MAX_SOTA:
            break
        s_cand = cand.get("_series")
        if s_cand is None:
            selected.append(cand)
            continue
        corr_ok = True
        for sel in selected:
            s_sel = sel.get("_series")
            if s_sel is None:
                continue
            merged = pd.concat([s_cand, s_sel], axis=1).dropna()
            if len(merged) >= 20:
                cv = abs(merged.iloc[:, 0].corr(
                    merged.iloc[:, 1], method="spearman"))
                if cv >= CORR_THRESHOLD:
                    corr_ok = False
                    break
        if corr_ok:
            selected.append(cand)

    # Clean _series trước khi trả về
    for a in selected:
        a.pop("_series", None)
    return selected


async def backtest_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Full evaluation trên candidate pool."""
    from graph import DATA_STORE
 
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    data = DATA_STORE.get(thread_id)
    if data is None:
        log.warning("[Backtest] No data in DATA_STORE, skipping")
        return {"evaluated_alphas": state.candidate_alphas, "sota_alphas": []}
 
    df, fwd_ret = data

    evaluated = []
    for cand in state.candidate_alphas:
        result = eval_alpha(cand, df, fwd_ret, full=True)
        result["id"]          = cand.get("id", "")
        result["description"] = cand.get("description", "")
        result["family"]      = cand.get("family", "")
        # Tạm giữ series để select_sota dùng, sẽ xóa sau
        result["_series"]     = result.pop("series", None)
        evaluated.append(result)
        status = result.get("status", "ERR")
        ic_oos = result.get("ic_oos") or 0.0
        if status == "OK":
            log.info(
                f"  [OK  ] {result['id']} [{result.get('family','?')}] "
                f"IC_OOS={ic_oos:+.4f}"
            )
        elif status == "MARGINAL":
            log.info(
                f"  [MARG] {result['id']} [{result.get('family','?')}] "
                f"IC_OOS={ic_oos:+.4f} — {result.get('weak_reason','')}"
            )
        elif status == "WEAK":
            log.info(
                f"  [WEAK] {result['id']} [{result.get('family','?')}] "
                f"IC_OOS={ic_oos:+.4f} — {result.get('weak_reason','')}"
            )
        else:
            log.info(
                f"  [ERR ] {result['id']} — {result.get('error','')[:60]}"
            )

    sota = _select_sota(evaluated)

    # Xóa _series trước khi lưu vào state
    for a in evaluated:
        a.pop("_series", None)

    n_ok   = sum(1 for a in evaluated if a.get("status") == "OK")
    n_marg = sum(1 for a in evaluated if a.get("status") == "MARGINAL")
    n_weak = sum(1 for a in evaluated if a.get("status") == "WEAK")
    n_err  = sum(1 for a in evaluated if a.get("status") == "EVAL_ERROR")
    log.info(
        f"[Backtest] OK={n_ok} MARGINAL={n_marg} WEAK={n_weak} ERR={n_err} "
        f"| {len(sota)} sota selected"
    )
 
    return {
        "evaluated_alphas": evaluated,
        "sota_alphas":      sota,
    }