"""
gp_search.py
Lightweight GP enhancement — paper Section 2.2, Alpha Compute Framework.
Fitness = cross-sectional IC trên toàn bộ universe.
"""
import re
import random
import logging
from copy import deepcopy
from typing import Callable, Dict, Any, List, Set, Optional, Tuple

import numpy as np
import pandas as pd

from validators import validate_expression, normalize_expression
from backtester import compute_ic
from config import DEFAULT_CONFIG

log = logging.getLogger(__name__)

WINDOW_VALUES = [3, 5, 7, 10, 12, 15, 20, 25, 30]

OPERATOR_FAMILIES = {
    "smoothing":     ["ts_mean", "ts_ema", "ts_decayed_linear", "decay_linear"],
    "normalization": ["ts_zscore_scale", "ts_maxmin_scale", "ts_rank"],
    "momentum":      ["ts_delta", "ts_delta_ratio", "delta"],
    "volatility":    ["ts_std", "stddev", "ts_ir"],
    "extreme":       ["ts_max", "ts_min"],
    "correlation":   ["ts_corr", "ts_cov", "correlation", "covariance"],
}

# ── Mutation functions ────────────────────────────────────────────────

def mutate_window(expr: str) -> str:
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
    for _, ops in OPERATOR_FAMILIES.items():
        for op_name in ops:
            if op_name + "(" in expr:
                peers = [p for p in ops if p != op_name]
                if peers:
                    return expr.replace(op_name + "(", random.choice(peers) + "(", 1)
    return expr


def mutate_wrap_normalize(expr: str) -> str:
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


def _extract_subtrees(expr: str) -> list:
    """
    Trích xuất tất cả function call subtrees từ expression.
    Trả về list of (start_idx, end_idx, subtree_string).
    Xử lý đúng ngoặc lồng nhau.
    """
    subtrees = []
    func_pat = re.compile(
        r'\b(ts_[a-z_]+|grouped_[a-z_]+|rank|neg|div|add|minus|cwise_mul'
        r'|cwise_max|cwise_min|abso|sign|log|tanh|relu|greater|less'
        r'|zscore_scale|normed_rank|scale|shift|delay|delta|stddev'
        r'|correlation|covariance|product|sum_op|decay_linear)\s*\('
    )
    for m in func_pat.finditer(expr):
        func_end = m.end() - 1
        depth = 0
        j = func_end
        while j < len(expr):
            if expr[j] == '(':
                depth += 1
            elif expr[j] == ')':
                depth -= 1
                if depth == 0:
                    subtrees.append((m.start(), j + 1, expr[m.start():j + 1]))
                    break
            j += 1
    return subtrees


def crossover(expr_a: str, expr_b: str) -> str:
    """
    Swap một subtree từ expr_b vào vị trí subtree cùng operator trong expr_a.
    Dùng _extract_subtrees để xử lý đúng ngoặc lồng nhau.
    Fallback về mutate_window nếu không tìm được operator chung.
    """
    if expr_a == expr_b:
        return mutate_window(expr_a)

    trees_a = _extract_subtrees(expr_a)
    trees_b = _extract_subtrees(expr_b)

    if not trees_a or not trees_b:
        return mutate_window(expr_a)

    def _op_name(subtree_str: str) -> str:
        m = re.match(r'(\w+)\s*\(', subtree_str)
        return m.group(1) if m else ""

    groups_a = {}
    for start, end, s in trees_a:
        op = _op_name(s)
        if op:
            groups_a.setdefault(op, []).append((start, end, s))

    groups_b = {}
    for start, end, s in trees_b:
        op = _op_name(s)
        if op:
            groups_b.setdefault(op, []).append((start, end, s))

    common_ops = list(set(groups_a.keys()) & set(groups_b.keys()))
    if not common_ops:
        return mutate_window(expr_a)

    chosen_op = random.choice(common_ops)
    _, _, src_subtree = random.choice(groups_b[chosen_op])
    dst_start, dst_end, _ = random.choice(groups_a[chosen_op])

    new_expr = expr_a[:dst_start] + src_subtree + expr_a[dst_end:]
    return new_expr


# ── Cross-sectional fitness ───────────────────────────────────────────

def _compute_cs_fitness(
    expression: str,
    df_by_ticker: Dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
) -> float:
    """
    Tính cross-sectional IC trên sampled universe.
    Trả về mean_ic (float), NaN nếu không tính được.
    """
    import alpha_operators as op_module

    def _exec_ticker(expr, df_t):
        import numpy as np
        ns = {name: getattr(op_module, name)
              for name in dir(op_module) if not name.startswith("_")}
        ns.update({"df": df_t, "np": np})
        for col in df_t.columns:
            ns[col] = df_t[col]
        exec(expr, ns)
        series = ns.get("alpha")
        if not isinstance(series, pd.Series):
            return None
        series = series.replace([float("inf"), float("-inf")], float("nan"))
        mu  = series.expanding(min_periods=20).mean()
        std = series.expanding(min_periods=20).std()
        return ((series - mu) / (std + 1e-9)).clip(-5, 5)

    signal_parts = {}
    for ticker, df_t in df_by_ticker.items():
        try:
            norm = _exec_ticker(expression, df_t)
            if norm is not None and norm.dropna().std() > 1e-9:
                signal_parts[ticker] = norm
        except Exception:
            pass

    if len(signal_parts) < 5:
        return float("nan")

    signal_df = pd.DataFrame(signal_parts)
    signal_df.index = pd.to_datetime(signal_df.index)

    fwd = forward_return.copy()
    fwd.index = pd.to_datetime(fwd.index)

    # Cross-sectional normalize mỗi ngày
    signal_norm = signal_df.apply(
        lambda row: (row - row.mean()) / (row.std() + 1e-9),
        axis=1,
    )

    common_dates = sorted(signal_norm.index.intersection(fwd.index))
    if not common_dates:
        return float("nan")

    # GP fitness = IC trên 70% ngày đầu (IC_IS).
    split_idx = int(len(common_dates) * (1 - DEFAULT_CONFIG.test_ratio))
    train_dates = common_dates[:split_idx] if split_idx > 0 else common_dates

    mean_ic = compute_ic(
        signal_norm.loc[train_dates],
        fwd.loc[train_dates],
    )
    return mean_ic if mean_ic is not None and not (mean_ic != mean_ic) else float("nan")


# ── Main GP loop ──────────────────────────────────────────────────────

def enhance_alpha(
    seeds: List[Dict[str, Any]],
    df_by_ticker: Dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    n_iterations: int = None,
) -> List[Dict[str, Any]]:
    if n_iterations is None:
        n_iterations = DEFAULT_CONFIG.gp_iterations

    seen_expressions: Set[str] = set()
    seed_ic_map: Dict[str, float] = {}  # origin_id → ic_is của seed gốc

    population = []
    for seed in seeds:
        expr = seed.get("expression", "")
        if not expr:
            continue
        ic_is = _compute_cs_fitness(expr, df_by_ticker, forward_return)
        ic_val = round(float(ic_is), 6) if (ic_is == ic_is) else 0.0
        entry = deepcopy(seed)
        entry["ic_is"] = ic_val
        entry["_origin_id"] = seed.get("id", "")
        entry["_origin_desc"] = seed.get("description", "")
        seed_ic_map[seed.get("id", "")] = ic_val
        seen_expressions.add(normalize_expression(expr))
        population.append(entry)

    if not population:
        return seeds

    mutation_fns = [mutate_window, mutate_operator, mutate_wrap_normalize]
    mutation_probs = [0.50, 0.25, 0.15]

    for _ in range(n_iterations):
        candidates = []
        pop_size = max(DEFAULT_CONFIG.population_size, len(population))

        for _ in range(pop_size):
            tournament = random.sample(population, min(3, len(population)))
            parent = max(tournament, key=lambda x: x.get("ic_is", 0.0))

            r = random.random()
            cumul = 0.0
            new_expr = None

            for prob, fn in zip(mutation_probs, mutation_fns):
                cumul += prob
                if r < cumul:
                    new_expr = fn(parent["expression"])
                    break

            if new_expr is None:
                others = [p for p in population if p is not parent]
                if others:
                    partner = max(
                        random.sample(others, min(3, len(others))),
                        key=lambda x: x.get("ic_is", 0.0)
                    )
                    new_expr = crossover(parent["expression"], partner["expression"])
                else:
                    new_expr = mutate_window(parent["expression"])

            if not new_expr:
                continue

            is_valid, _ = validate_expression(new_expr)
            if not is_valid:
                continue

            norm = normalize_expression(new_expr)
            if norm in seen_expressions:
                continue
            seen_expressions.add(norm)

            ic_is = _compute_cs_fitness(new_expr, df_by_ticker, forward_return)

            new_indiv = deepcopy(parent)
            new_indiv["expression"] = new_expr
            new_indiv["ic_is"] = round(float(ic_is), 6)
            candidates.append(new_indiv)

        if candidates:
            all_indivs = population + candidates
            all_indivs.sort(key=lambda x: x.get("ic_is", 0.0), reverse=True)

            # Đảm bảo mỗi origin_id luôn có ít nhất 1 đại diện trong population
            kept_origins: Set[str] = set()
            new_population = []
            for indiv in all_indivs:
                origin_id = indiv.get("_origin_id", "")
                if origin_id not in kept_origins:
                    new_population.append(indiv)
                    kept_origins.add(origin_id)
            for indiv in all_indivs:
                if len(new_population) >= len(seeds):
                    break
                if indiv not in new_population:
                    new_population.append(indiv)
            population = new_population

    # Map best result về từng seed theo _origin_id
    best_by_origin: Dict[str, Any] = {}
    for indiv in population:
        origin_id = indiv.get("_origin_id", "")
        if not origin_id:
            continue
        current_best = best_by_origin.get(origin_id)
        if current_best is None or indiv.get("ic_is", 0.0) > current_best.get("ic_is", 0.0):
            best_by_origin[origin_id] = indiv

    results = []
    for seed in seeds:
        seed_id = seed.get("id", "")
        best = best_by_origin.get(seed_id)

        if best is None:
            # Seed không có expression hợp lệ để chạy GP
            results.append(deepcopy(seed))
            continue

        origin_ic = seed_ic_map.get(seed_id, 0.0)
        best_ic = best.get("ic_is", 0.0)

        is_gp_improved = (
            normalize_expression(best.get("expression", "")) != normalize_expression(seed.get("expression", ""))
            and best_ic > origin_ic
        )

        if is_gp_improved:
            result = deepcopy(best)
            result["id"] = seed_id
            result["description"] = f"GP from {seed_id}: {seed.get('description', '')}"
            result.pop("_origin_id", None)
            result.pop("_origin_desc", None)
        else:
            result = deepcopy(seed)
            result["ic_is"] = origin_ic

        result["status"] = "OK" if (result.get("ic_is") or 0) > 0 else "WEAK"
        results.append(result)

    return results