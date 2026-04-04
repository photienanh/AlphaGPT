"""
core/genetic_search.py
═══════════════════════════════════════════════════════════════════════
Implements Algorithmic Alpha Mining layer from Alpha-GPT paper.

Paper description (Section 4.2):
  "The Alpha Search Enhancement module uses techniques like genetic
   programming to generate a diverse set of alpha candidates."

Pipeline:
  LLM seed alphas ──► GP mutation/crossover ──► Backtest filter ──► Best alphas

This is a lightweight GP implementation focused on:
  - Window size mutation (most impactful, least risky change)
  - Operator swap (within same family)
  - Sub-expression wrapping (add normalization layer)
  - Crossover between two parent expressions

NOT a full GP tree search (too slow for 30 tickers × 5 alphas).
Goal: quick improvement of LLM-generated seeds in ~10-20 iterations.
"""

import re
import logging
import random
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Mutation tables ───────────────────────────────────────────────────

# Window values to try during mutation
WINDOW_VALUES = [3, 5, 7, 10, 12, 15, 20, 25, 30]

# Operator families — can swap within family
OPERATOR_FAMILIES = {
    "smoothing":     ["ts_mean", "ts_ema", "ts_decayed_linear"],
    "normalization": ["ts_zscore_scale", "ts_maxmin_scale", "ts_rank"],
    "momentum":      ["ts_delta", "ts_delta_ratio"],
    "volatility":    ["ts_std", "ts_ir"],
    "extreme":       ["ts_max", "ts_min"],
    "correlation":   ["ts_corr", "ts_cov"],
    "distribution":  ["ts_skew", "ts_kurt"],
}

# Build reverse map: operator → family
OP_TO_FAMILY = {}
for fam, ops in OPERATOR_FAMILIES.items():
    for op in ops:
        OP_TO_FAMILY[op] = fam


# ── Mutation functions ────────────────────────────────────────────────

def mutate_window(expression: str) -> str:
    """
    Randomly change one window size argument to an adjacent value.
    Example: ts_mean(s, 20) → ts_mean(s, 15)
    """
    # Find all integer arguments that look like window sizes (5-60)
    pattern = re.compile(r"(\b)(\d{1,2})(\b)")
    matches = [(m.start(), m.end(), int(m.group())) for m in pattern.finditer(expression)
               if 3 <= int(m.group()) <= 60]

    if not matches:
        return expression

    # Pick one random window to mutate
    start, end, val = random.choice(matches)
    # Move to adjacent window value
    idx = WINDOW_VALUES.index(min(WINDOW_VALUES, key=lambda x: abs(x - val)))
    delta = random.choice([-1, 1])
    new_idx = max(0, min(len(WINDOW_VALUES) - 1, idx + delta))
    new_val = WINDOW_VALUES[new_idx]

    return expression[:start] + str(new_val) + expression[end:]


def mutate_operator(expression: str) -> str:
    """
    Swap one operator with another in the same family.
    Example: ts_mean(s, 10) → ts_ema(s, 10)
    """
    for family, ops in OPERATOR_FAMILIES.items():
        for op in ops:
            if op + "(" in expression:
                peers = [p for p in ops if p != op]
                if peers:
                    new_op = random.choice(peers)
                    return expression.replace(op + "(", new_op + "(", 1)
    return expression


def mutate_wrap_normalize(expression: str) -> str:
    """
    Wrap the entire alpha expression with a normalization layer.
    Example: alpha = X → alpha = ts_zscore_scale(X, 20)
    """
    # Extract RHS of alpha = ...
    if "alpha = " not in expression:
        return expression
    rhs = expression.split("alpha = ", 1)[1].strip()

    # Don't double-wrap
    normalizers = ["ts_zscore_scale", "ts_maxmin_scale", "ts_rank", "tanh", "winsorize_scale"]
    for n in normalizers:
        if rhs.startswith(n + "("):
            return expression  # already normalized

    # Choose normalizer
    norm = random.choice(["ts_zscore_scale({}, 20)", "tanh(ts_zscore_scale({}, 15))"])
    new_rhs = norm.format(rhs)
    return f"alpha = {new_rhs}"


def crossover(expr_a: str, expr_b: str) -> str:
    """
    Simple crossover: take the outer structure of expr_a,
    but replace one inner sub-expression with one from expr_b.

    Strategy: find common operator in both, swap one argument.
    """
    # Find operators in both
    func_pattern = re.compile(r"(ts_[a-z_]+|grouped_[a-z_]+)\(")
    ops_a = set(m.group(1) for m in func_pattern.finditer(expr_a))
    ops_b = set(m.group(1) for m in func_pattern.finditer(expr_b))
    common = ops_a & ops_b

    if not common:
        # No common operator → mutate A instead
        return mutate_window(expr_a)

    op = random.choice(list(common))

    # Find the argument of this operator in expr_b
    match_b = re.search(rf"{op}\(([^)]+)\)", expr_b)
    if not match_b:
        return mutate_window(expr_a)

    arg_b = match_b.group(1)

    # Replace first occurrence in expr_a
    match_a = re.search(rf"{op}\(([^)]+)\)", expr_a)
    if not match_a:
        return mutate_window(expr_a)

    new_expr = expr_a[:match_a.start(1)] + arg_b + expr_a[match_a.end(1):]
    return new_expr


# ── GP enhancement loop ───────────────────────────────────────────────

def enhance_alpha(
    seed_expression: str,
    seed_idea: str,
    df: pd.DataFrame,
    fwd_ret: pd.Series,
    fwd_ret_5d: Optional[pd.Series],
    eval_fn: Callable,
    n_iterations: int = 20,
    population_size: int = 8,
    mutation_probs: tuple = (0.5, 0.25, 0.15, 0.10),  # window, operator, wrap, crossover
) -> dict:
    """
    Run GP enhancement on a single seed alpha.

    Args:
        seed_expression: LLM-generated alpha expression
        eval_fn: function(expr, idea, id, df, fwd_ret, fwd_ret_5d) → result dict
        n_iterations: number of GP generations
        population_size: number of mutations per generation

    Returns:
        Best result dict found (may be same as seed if no improvement)
    """
    # Evaluate seed
    seed_result = eval_fn(
        {"id": 99, "idea": seed_idea, "expression": seed_expression},
        df, fwd_ret, fwd_ret_5d
    )

    if seed_result["status"] != "OK":
        log.debug(f"[GP] Seed invalid, skipping enhancement")
        return seed_result

    best = deepcopy(seed_result)
    best_score = best.get("score", 0.0)

    log.debug(f"[GP] Seed score={best_score:.4f}, running {n_iterations} iterations")

    for iteration in range(n_iterations):
        # Generate population of mutants
        mutants = []

        for _ in range(population_size):
            r = random.random()
            cumulative = 0.0
            for prob, mutate_fn in zip(
                mutation_probs,
                [mutate_window, mutate_operator, mutate_wrap_normalize,
                 lambda e: crossover(e, best.get("expression", e))]
            ):
                cumulative += prob
                if r < cumulative:
                    new_expr = mutate_fn(best.get("expression", seed_expression))
                    break
            else:
                new_expr = mutate_window(best.get("expression", seed_expression))

            if new_expr != best.get("expression"):
                mutants.append(new_expr)

        # Evaluate mutants and keep best
        for expr in mutants:
            try:
                result = eval_fn(
                    {"id": 99, "idea": seed_idea, "expression": expr},
                    df, fwd_ret, fwd_ret_5d
                )
                if result["status"] == "OK" and result.get("score", 0) > best_score:
                    best_score = result["score"]
                    best = deepcopy(result)
                    best["expression"] = expr  # keep mutated expression
                    log.debug(f"[GP] iter={iteration} improved score={best_score:.4f}")
            except Exception:
                continue

    improved = best.get("expression", seed_expression) != seed_expression
    if improved:
        seed_score = seed_result.get("score")
        seed_score_str = f"{seed_score:.4f}" if seed_score is not None else "?"
        log.info(f"[GP] Enhanced: {seed_score_str} -> {best_score:.4f}")

    return best


def enhance_alpha_population(
    seed_results: list[dict],
    df: pd.DataFrame,
    fwd_ret: pd.Series,
    fwd_ret_5d: Optional[pd.Series],
    eval_fn: Callable,
    n_iterations: int = 15,
) -> list[dict]:
    """
    Apply GP enhancement to a population of seed alphas.
    Returns enhanced results (replaces seed if improved).
    """
    enhanced = []
    for seed in seed_results:
        if seed.get("status") != "OK":
            enhanced.append(seed)
            continue
        try:
            result = enhance_alpha(
                seed_expression=seed["expression"],
                seed_idea=seed.get("idea", ""),
                df=df,
                fwd_ret=fwd_ret,
                fwd_ret_5d=fwd_ret_5d,
                eval_fn=eval_fn,
                n_iterations=n_iterations,
            )
            # Preserve original metadata (idea, id) if expression changed
            if result.get("expression") != seed.get("expression"):
                result["idea"]         = seed.get("idea", result.get("idea", ""))
                result["id"]           = seed.get("id", result.get("id"))
                result["gp_enhanced"]  = True
            enhanced.append(result)
        except Exception as e:
            log.warning(f"[GP] Enhancement failed for alpha {seed.get('id')}: {e}")
            enhanced.append(seed)

    return enhanced