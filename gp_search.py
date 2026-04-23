"""
gp_search.py
Lightweight GP enhancement — bước 2.2 trong paper.
Fitness function dùng IC_IS (nhanh), không dùng OOS để tránh overfit.
"""
import re
import random
import logging
from copy import deepcopy
from typing import Callable, Dict, Any, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WINDOW_VALUES = [3, 5, 7, 10, 12, 15, 20, 25, 30]

OPERATOR_FAMILIES = {
    "smoothing":     ["ts_mean", "ts_ema", "ts_decayed_linear"],
    "normalization": ["ts_zscore_scale", "ts_maxmin_scale", "ts_rank"],
    "momentum":      ["ts_delta", "ts_delta_ratio"],
    "volatility":    ["ts_std", "ts_ir"],
    "extreme":       ["ts_max", "ts_min"],
    "correlation":   ["ts_corr", "ts_cov"],
}


# ── Mutation functions ────────────────────────────────────────────────

def mutate_window(expr: str) -> str:
    """Thay đổi một window size."""
    matches = [(m.start(), m.end(), int(m.group()))
               for m in re.finditer(r"\b(\d{1,2})\b", expr)
               if 3 <= int(m.group()) <= 60]
    if not matches:
        return expr
    start, end, val = random.choice(matches)
    idx = min(range(len(WINDOW_VALUES)),
              key=lambda i: abs(WINDOW_VALUES[i] - val))
    new_idx = max(0, min(len(WINDOW_VALUES) - 1,
                         idx + random.choice([-1, 1])))
    return expr[:start] + str(WINDOW_VALUES[new_idx]) + expr[end:]


def mutate_operator(expr: str) -> str:
    """Swap operator trong cùng family."""
    for fam, ops in OPERATOR_FAMILIES.items():
        for op_name in ops:
            if op_name + "(" in expr:
                peers = [p for p in ops if p != op_name]
                if peers:
                    return expr.replace(op_name + "(", random.choice(peers) + "(", 1)
    return expr


def mutate_wrap_normalize(expr: str) -> str:
    """Wrap toàn bộ alpha bằng normalization."""
    if "alpha = " not in expr:
        return expr
    rhs = expr.split("alpha = ", 1)[1].strip()
    for norm in ["ts_zscore_scale", "ts_maxmin_scale", "tanh"]:
        if rhs.startswith(norm + "("):
            return expr
    template = random.choice([
        "ts_zscore_scale({}, 20)",
        "tanh(ts_zscore_scale({}, 15))",
    ])
    return f"alpha = {template.format(rhs)}"


def crossover(expr_a: str, expr_b: str) -> str:
    """Lấy cấu trúc A nhưng thay một argument bằng từ B."""
    func_pat = re.compile(r"(ts_[a-z_]+|grouped_[a-z_]+)\(")
    ops_a = {m.group(1) for m in func_pat.finditer(expr_a)}
    ops_b = {m.group(1) for m in func_pat.finditer(expr_b)}
    common = ops_a & ops_b
    if not common:
        return mutate_window(expr_a)
    op = random.choice(list(common))
    m_b = re.search(rf"{op}\(([^)]+)\)", expr_b)
    m_a = re.search(rf"{op}\(([^)]+)\)", expr_a)
    if not m_b or not m_a:
        return mutate_window(expr_a)
    return expr_a[:m_a.start(1)] + m_b.group(1) + expr_a[m_a.end(1):]


# ── Main GP loop ──────────────────────────────────────────────────────

def enhance_alpha(
    seed: Dict[str, Any],
    df: pd.DataFrame,
    fwd_ret: pd.Series,
    eval_fn: Callable,
    n_iterations: int = 15,
    population_size: int = 6,
) -> Dict[str, Any]:
    """
    GP enhancement cho một seed alpha.
    eval_fn nhận (alpha_def, df, fwd_ret, full=False) → result dict.
    Fitness = IC_IS (fast path).
    """
    best = eval_fn(seed, df, fwd_ret, full=False)
    if best.get("status") != "OK":
        return best

    best_ic = best.get("ic_is") or 0.0
    mutation_fns = [mutate_window, mutate_operator,
                    mutate_wrap_normalize,
                    lambda e: crossover(e, best.get("expression", e))]
    probs = [0.50, 0.25, 0.15, 0.10]

    for _ in range(n_iterations):
        mutants = []
        for _ in range(population_size):
            r = random.random()
            cumul = 0.0
            chosen_fn = mutation_fns[0]
            for prob, fn in zip(probs, mutation_fns):
                cumul += prob
                if r < cumul:
                    chosen_fn = fn
                    break
            new_expr = chosen_fn(best.get("expression", seed["expression"]))
            if new_expr != best.get("expression"):
                mutants.append(new_expr)

        for expr in mutants:
            candidate = deepcopy(seed)
            candidate["expression"] = expr
            result = eval_fn(candidate, df, fwd_ret, full=False)
            if result.get("status") == "OK" and (result.get("ic_is") or 0) > best_ic:
                best_ic = result["ic_is"]
                best = deepcopy(result)
                best["expression"] = expr

    return best


def enhance_population(
    seeds: List[Dict[str, Any]],
    df: pd.DataFrame,
    fwd_ret: pd.Series,
    eval_fn: Callable,
    n_iterations: int = 15,
) -> List[Dict[str, Any]]:
    """GP enhancement cho toàn bộ population."""
    results = []
    for seed in seeds:
        if seed.get("status") == "EVAL_ERROR":
            results.append(seed)
            continue
        try:
            enhanced = enhance_alpha(seed, df, fwd_ret, eval_fn, n_iterations)
            enhanced["id"]          = seed.get("id", enhanced.get("id"))
            enhanced["description"] = seed.get("description", "")
            enhanced["family"]      = seed.get("family", "")
            results.append(enhanced)
        except Exception as e:
            log.warning(f"GP failed for alpha {seed.get('id')}: {e}")
            results.append(seed)
    return results