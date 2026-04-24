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
    for fam, ops in OPERATOR_FAMILIES.items():
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


def crossover(expr_a: str, expr_b: str) -> str:
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


# ── Cross-sectional fitness ───────────────────────────────────────────

def _compute_cs_fitness(
    expression: str,
    ticker_dfs: Dict[str, pd.DataFrame],
    fwd_ret_multi: pd.DataFrame,
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
    for ticker, df_t in ticker_dfs.items():
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

    fwd = fwd_ret_multi.copy()
    fwd.index = pd.to_datetime(fwd.index)

    # Cross-sectional normalize mỗi ngày
    signal_norm = signal_df.apply(
        lambda row: (row - row.mean()) / (row.std() + 1e-9),
        axis=1,
    )

    mean_ic, _, _ = compute_ic_cross_sectional(signal_norm, fwd)
    return mean_ic if mean_ic is not None and not (mean_ic != mean_ic) else float("nan")


def _sample_universe(
    ticker_dfs: Dict[str, pd.DataFrame],
    fwd_ret_multi: pd.DataFrame,
    sample_size: int = GP_SAMPLE_SIZE,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Sample ngẫu nhiên sample_size tickers từ universe.
    Ưu tiên tickers có nhiều dữ liệu hơn.
    """
    all_tickers = list(ticker_dfs.keys())
    if len(all_tickers) <= sample_size:
        return ticker_dfs, fwd_ret_multi

    # Sort theo số rows giảm dần, lấy top 2*sample_size rồi random sample
    sorted_tickers = sorted(all_tickers, key=lambda t: len(ticker_dfs[t]), reverse=True)
    pool = sorted_tickers[:min(len(sorted_tickers), 2 * sample_size)]
    sampled = random.sample(pool, sample_size)

    sampled_dfs = {t: ticker_dfs[t] for t in sampled}
    sampled_fwd = fwd_ret_multi[[t for t in sampled if t in fwd_ret_multi.columns]]
    return sampled_dfs, sampled_fwd


# ── Main GP loop ──────────────────────────────────────────────────────

def enhance_alpha(
    seed: Dict[str, Any],
    ticker_dfs: Dict[str, pd.DataFrame],
    fwd_ret_multi: pd.DataFrame,
    n_iterations: int = None,
    population_size: int = None,
    seen_expressions: Set[str] = None,
) -> Dict[str, Any]:
    """
    GP enhancement cho một seed alpha.
    Fitness = cross-sectional IC trên sampled universe.
    """
    if n_iterations is None:
        n_iterations = DEFAULT_CONFIG.gp_iterations
    if population_size is None:
        population_size = DEFAULT_CONFIG.population_size
    if seen_expressions is None:
        seen_expressions = set()

    expr = seed.get("expression", "")
    if not expr:
        return seed

    # Sample universe cho toàn bộ GP run của seed này
    sampled_dfs, sampled_fwd = _sample_universe(ticker_dfs, fwd_ret_multi)

    best_ic   = _compute_cs_fitness(expr, sampled_dfs, sampled_fwd)
    best_expr = expr
    seen_expressions.add(normalize_expression(expr))

    # Seed không có signal hợp lệ trên sampled universe
    if best_ic != best_ic:  # isnan
        best_ic = 0.0

    log.debug(
        f"[GP] Seed {seed.get('id','?')} "
        f"baseline IC={best_ic:+.4f} "
        f"({len(sampled_dfs)} tickers sampled)"
    )

    best_result = deepcopy(seed)
    best_result["ic_is"] = round(float(best_ic), 6) if best_ic == best_ic else None

    mutation_fns = [
        mutate_window,
        mutate_operator,
        mutate_wrap_normalize,
        lambda e: crossover(e, best_expr),
    ]
    probs = [0.50, 0.25, 0.15, 0.10]

    for iteration in range(n_iterations):
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
            # crossover closure cần capture best_expr tại thời điểm hiện tại
            if chosen_fn is mutation_fns[3]:
                new_expr = crossover(best_expr, best_expr)
            else:
                new_expr = chosen_fn(best_expr)
            if not new_expr:
                continue
            is_valid, _ = validate_expression(new_expr)
            if not is_valid:
                continue
            norm_expr = normalize_expression(new_expr)
            if norm_expr in seen_expressions:
                continue
            seen_expressions.add(norm_expr)
            mutants.append(new_expr)

        improved = False
        for mut_expr in mutants:
            ic = _compute_cs_fitness(mut_expr, sampled_dfs, sampled_fwd)
            if ic == ic and ic > best_ic:  # ic không phải NaN và tốt hơn
                best_ic   = ic
                best_expr = mut_expr
                improved  = True

        if improved:
            log.debug(f"[GP] iter {iteration+1}: IC improved to {best_ic:+.4f}")

    best_result["expression"] = best_expr
    best_result["ic_is"]      = round(float(best_ic), 6) if best_ic == best_ic else None
    best_result["status"]     = "OK" if (best_ic == best_ic and best_ic > 0) else "WEAK"
    return best_result


def enhance_population(
    seeds: List[Dict[str, Any]],
    ticker_dfs: Dict[str, pd.DataFrame],
    fwd_ret_multi: pd.DataFrame,
    n_iterations: int = None,
) -> List[Dict[str, Any]]:
    """GP enhancement cho toàn bộ population."""
    if n_iterations is None:
        n_iterations = DEFAULT_CONFIG.gp_iterations

    seen_expressions: Set[str] = set()

    results = []
    for seed in seeds:
        try:
            enhanced = enhance_alpha(
                seed, ticker_dfs, fwd_ret_multi,
                n_iterations=n_iterations,
                seen_expressions=seen_expressions,
            )
            enhanced["id"]          = seed.get("id", enhanced.get("id", ""))
            enhanced["description"] = seed.get("description", "")
            enhanced["family"]      = seed.get("family", "")
            results.append(enhanced)
        except Exception as e:
            log.warning(f"[GP] Failed for {seed.get('id','?')}: {e}")
            results.append(seed)
    return results