"""
gp_search.py
Lightweight GP enhancement — paper Section 2.2, Alpha Compute Framework.
Fitness = cross-sectional IC trên sampled universe (~20 tickers).
Nhất quán với final backtest metric, nhanh hơn full 403 tickers.
"""
import re
import random
import logging
from copy import deepcopy
from typing import Callable, Dict, Any, List, Set, Optional, Tuple

import numpy as np
import pandas as pd

from validators import validate_expression, normalize_expression
from backtester import compute_ic_cross_sectional
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

# Số tickers sample cho GP fitness — đủ để cross-sectional có ý nghĩa,
# ít hơn full universe để nhanh
GP_SAMPLE_SIZE = 30


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

    mean_ic = compute_ic_cross_sectional(signal_norm, fwd)
    return mean_ic if mean_ic is not None and not (mean_ic != mean_ic) else float("nan")


def _sample_universe(
    df_by_ticker: Dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    sample_size: int = GP_SAMPLE_SIZE,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Sample ngẫu nhiên sample_size tickers từ universe.
    Ưu tiên tickers có nhiều dữ liệu hơn.
    """
    all_tickers = list(df_by_ticker.keys())
    if len(all_tickers) <= sample_size:
        return df_by_ticker, forward_return

    # Sort theo số rows giảm dần, lấy top 2*sample_size rồi random sample
    sorted_tickers = sorted(all_tickers, key=lambda t: len(df_by_ticker[t]), reverse=True)
    pool = sorted_tickers[:min(len(sorted_tickers), 2 * sample_size)]
    sampled = random.sample(pool, sample_size)

    sampled_df_by_ticker = {t: df_by_ticker[t] for t in sampled}
    sampled_forward_return = forward_return[[t for t in sampled if t in forward_return.columns]]
    return sampled_df_by_ticker, sampled_forward_return


# ── Main GP loop ──────────────────────────────────────────────────────

def enhance_alpha(
    seeds: List[Dict[str, Any]],
    df_by_ticker: Dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    n_iterations: int = None,
) -> List[Dict[str, Any]]:
    """
    GP enhancement với shared population và elitist selection.
    Tất cả seeds tham gia cùng một population → có thể trao đổi
    genetic material qua crossover.
    """
    if n_iterations is None:
        n_iterations = DEFAULT_CONFIG.gp_iterations

    sampled_df_by_ticker, sampled_forward_return = _sample_universe(df_by_ticker, forward_return)

    seen_expressions: Set[str] = set()

    population = []
    for seed in seeds:
        expr = seed.get("expression", "")
        if not expr:
            continue
        ic = _compute_cs_fitness(expr, sampled_df_by_ticker, sampled_forward_return)
        entry = deepcopy(seed)
        entry["_ic"] = ic if (ic == ic) else 0.0
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
            parent = max(tournament, key=lambda x: x["_ic"])

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
                        key=lambda x: x["_ic"]
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

            ic = _compute_cs_fitness(new_expr, sampled_df_by_ticker, sampled_forward_return)
            ic_val = ic if (ic == ic) else 0.0

            new_indiv = deepcopy(parent)
            new_indiv["expression"] = new_expr
            new_indiv["_ic"] = ic_val
            candidates.append(new_indiv)

        if candidates:
            all_indivs = population + candidates
            all_indivs.sort(key=lambda x: x["_ic"], reverse=True)
            population = all_indivs[:len(seeds)]

    results = []
    for i, indiv in enumerate(population):
        ic_val = indiv.pop("_ic", 0) or 0
        if i < len(seeds):
            indiv["id"]          = seeds[i].get("id", indiv.get("id", ""))
            indiv["description"] = seeds[i].get("description", "")
        indiv["status"] = "OK" if ic_val > 0 else "WEAK"
        results.append(indiv)

    return results