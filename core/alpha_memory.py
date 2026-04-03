"""
core/alpha_memory.py
═══════════════════════════════════════════════════════════════════════
Implements the Knowledge Library described in Alpha-GPT paper (Figure 5).

Paper description:
  "The AlphaBot layer leverages RAG over a vector database of financial
   literature and historical alphas to ground the model's outputs."

This module provides:
  1. AlphaMemory   — stores successful alphas with metadata & decomposition
  2. retrieve()    — finds top-K similar alphas by idea/expression similarity
  3. decompose()   — breaks alpha expression into sub-expressions with explanations
  4. compile()     — builds few-shot examples block for LLM prompt (Knowledge Compiler)

Storage: JSON files per ticker in data/alpha_memory/
         + a global cross-ticker pool (data/alpha_memory/_global.json)

Design matches paper Figure 5:
  External Memory ──► Knowledge Compiler ──► Prompt Template ──► LLM
"""

import os
import json
import re
import logging
from copy import deepcopy
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Simple TF-IDF-style similarity (no external dependencies)
# ─────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Tokenize idea/expression into lowercase words + operator tokens."""
    text = text.lower()
    # Keep operator names as whole tokens
    tokens = re.findall(r"ts_[a-z_]+|grouped_[a-z_]+|[a-z]{3,}", text)
    return tokens


def _tfidf_sim(a: str, b: str) -> float:
    """Simple bag-of-words cosine similarity."""
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    return len(intersection) / (len(ta | tb) + 1e-9)


def _alpha_similarity(mem_entry: dict, query_idea: str, query_expr: str = "") -> float:
    """
    Score similarity between a memory entry and a new query.
    Weights: idea similarity 60%, expression operator overlap 40%.
    """
    idea_sim = _tfidf_sim(mem_entry.get("idea", ""), query_idea)
    expr_sim  = _tfidf_sim(mem_entry.get("expression", ""), query_expr)
    return 0.6 * idea_sim + 0.4 * expr_sim


# ─────────────────────────────────────────────────────────────────────
# Alpha decomposition (Thought De-compiler direction)
# ─────────────────────────────────────────────────────────────────────

def decompose_expression(expression: str) -> list[dict]:
    """
    Break expression into sub-components and label each.
    Paper: "decompose alpha into sub-expressions and explain them"

    Returns list of {"sub_expr": str, "role": str} dicts.
    """
    components = []

    # Find top-level function calls
    # Pattern: function_name(...)
    func_pattern = re.compile(
        r"(ts_[a-z_]+|grouped_[a-z_]+|zscore_scale|winsorize_scale|normed_rank|"
        r"cwise_mul|cwise_max|cwise_min|div|minus|add|tanh|relu|neg|sign|"
        r"ts_corr|ts_cov|ts_rank|ts_ir|ts_linear_reg)\s*\("
    )

    operators_found = []
    for m in func_pattern.finditer(expression):
        operators_found.append(m.group(1))

    # Label components by operator family
    role_map = {
        "ts_zscore_scale": "normalization",
        "ts_maxmin_scale": "normalization",
        "zscore_scale":    "normalization",
        "ts_rank":         "rank_transform",
        "ts_corr":         "correlation_signal",
        "ts_cov":          "covariance_signal",
        "ts_delta":        "momentum",
        "ts_delta_ratio":  "momentum_ratio",
        "ts_ir":           "information_ratio",
        "ts_linear_reg":   "trend_slope",
        "ts_ema":          "smoothing",
        "ts_mean":         "smoothing",
        "ts_std":          "volatility",
        "ts_skew":         "distribution_shape",
        "ts_kurt":         "distribution_shape",
        "ts_argmaxmin_diff": "cycle_position",
        "grouped_demean":  "demeaning",
        "tanh":            "soft_clipping",
        "cwise_mul":       "interaction",
        "div":             "ratio",
        "minus":           "divergence",
        "add":             "combination",
        "relu":            "threshold",
        "sign":            "direction",
    }

    for op in set(operators_found):
        role = role_map.get(op, "transform")
        components.append({"operator": op, "role": role})

    # Detect data sources used
    data_fields = re.findall(r"df\['([^']+)'\]", expression)
    sent_fields = [f for f in data_fields if f.endswith("_S")]
    tech_fields = [f for f in data_fields if not f.endswith("_S")]

    return {
        "operators": components,
        "data_sources": {
            "technical": list(set(tech_fields)),
            "sentiment": list(set(sent_fields)),
        },
        "n_components": len(components),
        "uses_sentiment": len(sent_fields) > 0,
    }


# ─────────────────────────────────────────────────────────────────────
# Main AlphaMemory class
# ─────────────────────────────────────────────────────────────────────

class AlphaMemory:
    """
    Persistent memory store for successful alphas.
    Implements the External Memory + Knowledge Library from Alpha-GPT paper.

    Per-ticker memory: top alphas for that specific ticker.
    Global memory:     best alphas across all tickers (cross-asset patterns).
    """

    GLOBAL_KEY = "_global"

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self._cache: dict[str, list[dict]] = {}

    # ── I/O ──────────────────────────────────────────────────────────

    def _path(self, key: str) -> str:
        return os.path.join(self.memory_dir, f"{key}.json")

    def _load(self, key: str) -> list[dict]:
        if key in self._cache:
            return self._cache[key]
        path = self._path(key)
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else data.get("entries", [])
            self._cache[key] = entries
            return entries
        except Exception as e:
            log.warning(f"[Memory] load {key} failed: {e}")
            return []

    def _save(self, key: str, entries: list[dict]) -> None:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        self._cache[key] = entries

    # ── Store ─────────────────────────────────────────────────────────

    def store(self, ticker: str, alpha: dict, max_per_ticker: int = 20) -> None:
        """
        Save a successful alpha to memory.
        Stores in both ticker-specific and global pool.
        Deduplicates by expression similarity.
        """
        if alpha.get("status") != "OK":
            return
        if not alpha.get("ic_oos") and not alpha.get("ic"):
            return

        entry = {
            "ticker":     ticker,
            "idea":       alpha.get("idea", ""),
            "expression": alpha.get("expression", ""),
            "ic_oos":     alpha.get("ic_oos"),
            "ic_is":      alpha.get("ic"),
            "sharpe_oos": alpha.get("sharpe_oos"),
            "sharpe":     alpha.get("sharpe"),
            "score":      alpha.get("score", 0.0),
            "flipped":    alpha.get("flipped", False),
            "decomposition": decompose_expression(alpha.get("expression", "")),
        }

        # Store in ticker pool
        self._upsert(ticker, entry, max_per_ticker)
        # Store in global pool (keep top 100 globally)
        self._upsert(self.GLOBAL_KEY, entry, 100)

    def _upsert(self, key: str, entry: dict, max_entries: int) -> None:
        entries = self._load(key)

        # Dedup: if very similar expression already exists, keep the better one
        for i, existing in enumerate(entries):
            if _tfidf_sim(existing.get("expression", ""), entry.get("expression", "")) > 0.85:
                if entry.get("score", 0) > existing.get("score", 0):
                    entries[i] = entry
                self._save(key, entries)
                return

        entries.append(entry)
        # Keep top entries by score
        entries.sort(key=lambda x: x.get("score", 0), reverse=True)
        entries = entries[:max_entries]
        self._save(key, entries)

    # ── Retrieve ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query_idea: str,
        query_expr: str = "",
        ticker: Optional[str] = None,
        top_k: int = 3,
        min_ic_oos: float = 0.01,
    ) -> list[dict]:
        """
        Find top-K most relevant alphas from memory.
        Paper: "Demonstration Retrieval" step in Knowledge Compiler.

        Searches ticker-specific pool first, then global pool.
        Deduplicates results.
        """
        candidates = []

        # Ticker-specific pool
        if ticker:
            for e in self._load(ticker):
                ic = e.get("ic_oos") or e.get("ic_is") or 0.0
                if abs(ic) >= min_ic_oos:
                    candidates.append(e)

        # Global pool
        for e in self._load(self.GLOBAL_KEY):
            if e.get("ticker") != ticker:  # avoid double-counting same ticker
                ic = e.get("ic_oos") or e.get("ic_is") or 0.0
                if abs(ic) >= min_ic_oos:
                    candidates.append(e)

        if not candidates:
            return []

        # Score by similarity
        scored = []
        for e in candidates:
            sim = _alpha_similarity(e, query_idea, query_expr)
            scored.append((sim, e))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by expression similarity
        selected = []
        for _, e in scored:
            if len(selected) >= top_k:
                break
            is_dup = any(
                _tfidf_sim(e.get("expression", ""), s.get("expression", "")) > 0.8
                for s in selected
            )
            if not is_dup:
                selected.append(e)

        return selected

    def retrieve_diverse(
        self,
        ticker: Optional[str] = None,
        top_k: int = 5,
        min_ic_oos: float = 0.01,
    ) -> list[dict]:
        """
        Retrieve diverse high-performing alphas covering different alpha families.
        Used for seed prompt when no specific query direction yet.
        """
        all_entries = []
        if ticker:
            all_entries.extend(self._load(ticker))
        all_entries.extend(self._load(self.GLOBAL_KEY))

        # Filter by IC
        good = [e for e in all_entries
                if abs(e.get("ic_oos") or e.get("ic_is") or 0) >= min_ic_oos]

        if not good:
            return []

        # Group by operator family and pick best from each group
        family_buckets: dict[str, list] = {}
        for e in good:
            decomp = e.get("decomposition", {})
            ops = [c.get("role", "other") for c in decomp.get("operators", [])]
            # Primary family = first operator role
            family = ops[0] if ops else "other"
            if family not in family_buckets:
                family_buckets[family] = []
            family_buckets[family].append(e)

        # Best from each family
        selected = []
        for family, entries in sorted(family_buckets.items(),
                                       key=lambda x: max(e.get("score", 0) for e in x[1]),
                                       reverse=True):
            if len(selected) >= top_k:
                break
            best = max(entries, key=lambda x: x.get("score", 0))
            selected.append(best)

        return selected[:top_k]

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self, ticker: Optional[str] = None) -> dict:
        ticker_entries = self._load(ticker) if ticker else []
        global_entries = self._load(self.GLOBAL_KEY)
        return {
            "ticker_count": len(ticker_entries),
            "global_count": len(global_entries),
            "ticker_avg_ic": (
                float(np.mean([abs(e.get("ic_oos") or e.get("ic_is") or 0)
                               for e in ticker_entries]))
                if ticker_entries else 0.0
            ),
        }


# ─────────────────────────────────────────────────────────────────────
# Prompt compilation helper (Knowledge Compiler in paper Figure 5)
# ─────────────────────────────────────────────────────────────────────

def compile_memory_block(
    memories: list[dict],
    label: str = "retrieved_memory",
    max_examples: int = 3,
) -> str:
    """
    Format retrieved memories into few-shot examples block for LLM prompt.
    Paper: combines retrieved examples into prompt template.
    """
    if not memories:
        return ""

    lines = [f"\n## {label} — Ví dụ alpha tốt từ lịch sử (học từ đây, KHÔNG copy):\n"]
    for i, m in enumerate(memories[:max_examples], 1):
        ic_val  = m.get("ic_oos") or m.get("ic_is") or 0
        sh_val  = m.get("sharpe_oos") or m.get("sharpe") or 0
        ticker  = m.get("ticker", "?")
        decomp  = m.get("decomposition", {})
        ops     = [c.get("role") for c in decomp.get("operators", [])][:3]
        ops_str = ", ".join(ops) if ops else "N/A"

        lines.append(
            f"### Example {i} [{ticker}] IC_oos={ic_val:+.4f} Sharpe={sh_val:.3f}\n"
            f"**Ý tưởng**: {m.get('idea', '')}\n"
            f"**Cấu trúc**: {ops_str}\n"
            f"**Expression**: `{m.get('expression', '')}`\n"
            f"→ Học cấu trúc và ý tưởng, SÁNG TẠO variation MỚI cho mã hiện tại.\n"
        )

    lines.append(
        "\n*Lưu ý: Dùng examples trên để học PATTERN, không copy expression.*\n"
    )
    return "\n".join(lines)