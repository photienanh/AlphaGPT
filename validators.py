"""
validators.py
Kiểm tra và normalize alpha expressions trước khi exec.
"""
import re
from typing import Tuple, Set

VALID_FIELDS: Set[str] = {
    "open", "high", "low", "close", "volume",
    "vwap", "adv20", "returns",
    "sma_5", "sma_20", "ema_10",
    "momentum_3", "momentum_10",
    "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_middle", "bb_lower",
    "obv",
}

VALID_OPERATORS: Set[str] = {
    # time-series
    "shift", "delay", "ts_corr", "correlation", "ts_cov", "covariance",
    "ts_mean", "ts_std", "stddev", "ts_sum", "sum_op",
    "ts_product", "product", "ts_min", "ts_max",
    "ts_argmax", "ts_argmin", "ts_argmaxmin_diff", "ts_max_diff", "ts_min_diff",
    "ts_median", "ts_rank", "ts_zscore_scale", "ts_maxmin_scale",
    "ts_skew", "ts_kurt", "ts_delta", "delta", "ts_delta_ratio",
    "ts_ir", "ts_decayed_linear", "decay_linear",
    "ts_ema", "ts_percentile", "ts_linear_reg",
    # cross-sectional
    "rank", "cs_rank", "rank_ts","zscore_scale", "winsorize_scale",
    "normed_rank", "cwise_max", "cwise_min",
    "scale", "signed_power", "indneutralize",
    # group-wise
    "grouped_mean", "grouped_std", "grouped_max", "grouped_min",
    "grouped_sum", "grouped_demean", "grouped_zscore_scale",
    "grouped_winsorize_scale",
    # element-wise
    "relu", "neg", "abso", "abs_op", "log", "log1p", "sign",
    "pow_op", "pow_sign", "round_op",
    "add", "minus", "div", "greater", "less",
    "cwise_mul", "normed_rank_diff", "tanh", "clip",
}

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(if|else|elif|for|while|lambda|import|exec|eval|__)\b"
)


def validate_expression(expr: str) -> Tuple[bool, str]:
    """
    Kiểm tra expression trước khi exec.
    Returns: (is_valid, error_message)
    """
    if not expr or not expr.strip():
        return False, "expression rỗng"

    if "alpha" not in expr:
        return False, "thiếu assignment 'alpha = ...'"

    # Phải có dạng "alpha = ..."
    if not re.search(r"alpha\s*=", expr):
        return False, "thiếu 'alpha = ' trong expression"

    # Không cho phép các cấu trúc nguy hiểm
    m = _FORBIDDEN_KEYWORDS.search(expr)
    if m:
        return False, f"cấu trúc không được phép: '{m.group()}'"

    # Kiểm tra df['field'] — chỉ cho phép VALID_FIELDS
    field_refs = re.findall(r"df\[[\'\"](\w+)[\'\"]\]", expr)
    for field in field_refs:
        if field.lower() not in VALID_FIELDS:
            return False, f"field không hợp lệ: df['{field}']"

    # Kiểm tra tên hàm được gọi
    func_calls = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", expr)
    for fn in func_calls:
        if fn in ("alpha", "df", "np", "pd", "float", "int", "abs", "min", "max"):
            continue
        if fn not in VALID_OPERATORS:
            return False, f"operator không hợp lệ: '{fn}'"

    return True, ""


def normalize_expression(expr: str) -> str:
    """
    Normalize expression để deduplication.
    - Strip whitespace thừa
    - Lowercase
    - Sort args của commutative operators (add, cwise_mul)
    """
    if not expr:
        return expr

    normalized = " ".join(expr.split()).lower()

    # Normalize add(a, b) và add(b, a) → cùng dạng (sort args)
    def _sort_commutative(m):
        op = m.group(1)
        args_str = m.group(2)
        # Tách args đơn giản (không lồng nhau)
        if "(" not in args_str:
            args = [a.strip() for a in args_str.split(",")]
            if len(args) == 2:
                args.sort()
                return f"{op}({', '.join(args)})"
        return m.group(0)

    for comm_op in ("add", "cwise_mul"):
        normalized = re.sub(
            rf"\b({comm_op})\(([^()]+)\)",
            _sort_commutative,
            normalized,
        )

    return normalized