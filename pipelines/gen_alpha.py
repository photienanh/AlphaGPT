"""
pipelines/gen_alpha.py  (v3 — Alpha-GPT paper implementation)
═══════════════════════════════════════════════════════════════════════
Implements the full Alpha-GPT agentic workflow (Figure 5 in paper):

  IDEATION:
    Trading Idea Polisher → structured prompt
    RAG retrieval from alpha memory (Knowledge Compiler)

  IMPLEMENTATION:
    Quant Developer (LLM) → seed alpha expressions
    Alpha Compute Framework (GP) → search enhancement

  REVIEW:
    Analyst agent → walk-forward backtest (IC_OOS, Sharpe_OOS)
    Thought De-compiler → natural language summary
    Human feedback loop → next round refinement

Key improvements vs v2:
  ① Walk-forward IC  — IC evaluated on OOS split (no data leakage)
  ② RAG memory       — few-shot examples from successful alphas
  ③ GP enhancement   — algorithmic search after LLM seed generation
  ④ Better prompts   — hypothesis-driven, operator decomposition guidance
  ⑤ Overfit penalty  — composite score penalises IS >> OOS alphas

Usage:
    python pipelines/gen_alpha.py --ticker HPG
    python pipelines/gen_alpha.py --all --force
    python pipelines/gen_alpha.py --ticker HPG --refine-only
"""

import os, json, argparse, logging, sys, random
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.universe import VN30_SYMBOLS, TICKER_INDUSTRY
from core.paths import (FEATURES_DIR, SENTIMENT_OUTPUT_DIR,
                        ALPHA_VALUES_DIR, ALPHA_FORMULA_DIR, ALPHA_MEMORY_DIR)
from core import alpha_operators as op
from core.backtester import (compute_ic, compute_ic_oos, compute_sharpe,
                              compute_sharpe_oos, compute_turnover, composite_score)
from core.alpha_memory import AlphaMemory, compile_memory_block, decompose_expression
from core.genetic_search import enhance_alpha_population

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Directories ───────────────────────────────────────────────────────
SENTIMENT_DIR = SENTIMENT_OUTPUT_DIR
for d in [ALPHA_VALUES_DIR, ALPHA_FORMULA_DIR, ALPHA_MEMORY_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────
MAX_ROUNDS             = 5
SEED_OVERSAMPLE        = 10     # generate N candidates, select best 5
GP_ITERATIONS          = 15     # GP enhancement iterations per alpha
GP_ENABLED             = True   # set False to skip GP (faster but worse)
IC_THRESHOLD           = 0.015  # minimum |IC_OOS| to pass
SHARPE_THRESHOLD       = 0.20   # minimum Sharpe_OOS to pass
CORR_THRESHOLD         = 0.55   # max pairwise correlation (lowered from 0.65)
OOS_TEST_RATIO         = 0.30   # 30% held-out for OOS evaluation
LLM_MODEL              = "gpt-4o-mini"
LLM_TEMPERATURE        = 0.7
LLM_MAX_TOKENS         = 4500
REFINE_ATTEMPTS        = 3
MEMORY_RETRIEVE_K      = 3      # RAG: how many examples to retrieve

# Minimum nonzero ratio for sentiment columns to be included in prompt
# Relaxed to 0.05 (5%) because news is mostly neutral — but we still
# need at least some signal days to be useful
SENT_MIN_NONZERO_RATIO = 0.05

# ── Memory store ──────────────────────────────────────────────────────
memory = AlphaMemory(ALPHA_MEMORY_DIR)


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data(ticker: str) -> pd.DataFrame:
    feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
    sent_path = os.path.join(SENTIMENT_DIR, f"{ticker}_Full_Sentiment.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features not found: {feat_path}")
    if not os.path.exists(sent_path):
        raise FileNotFoundError(f"Sentiment not found: {sent_path}")
    df_feat = pd.read_csv(feat_path)
    df_feat["time"] = pd.to_datetime(df_feat["time"]).dt.normalize()
    df_feat = df_feat.set_index("time").sort_index()
    df_sent = pd.read_csv(sent_path, index_col="time", parse_dates=True)
    df_sent.index = pd.to_datetime(df_sent.index).normalize()
    df = df_feat.join(df_sent.sort_index(), how="inner")
    df = df.dropna(subset=["close"]).fillna(0.0)
    log.info(f"[{ticker}] Loaded {len(df)} rows × {len(df.columns)} cols")
    return df


def make_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    return df["close"].pct_change(horizon).shift(-horizon).rename(f"fwd_{horizon}d")


def get_data_stats(df: pd.DataFrame) -> str:
    cols = ["close", "volume", "RSI_14", "MACD", "BB_Upper", "BB_Lower",
            "OBV", "Momentum_3", "Momentum_10"]
    lines = []
    for c in cols:
        if c in df.columns:
            s = df[c].dropna()
            lines.append(
                f"  {c}: μ={s.mean():.2f} σ={s.std():.2f} "
                f"[{s.min():.2f}, {s.max():.2f}]"
            )
    return "\n".join(lines)


def get_sentiment_quality(sent_cols: list[str], df: pd.DataFrame) -> dict:
    """
    Analyse sentiment column quality.
    Sentiment is mostly neutral (0) — threshold relaxed to 5%.
    """
    result = {}
    for c in sent_cols:
        if c not in df.columns:
            result[c] = {"ok": False, "nonzero": 0.0, "reason": "missing"}
            continue
        s = df[c].dropna()
        if len(s) == 0:
            result[c] = {"ok": False, "nonzero": 0.0, "reason": "empty"}
            continue
        nonzero = float((s != 0).mean())
        pos_ratio = float((s > 0).mean())
        neg_ratio = float((s < 0).mean())
        ok = nonzero >= SENT_MIN_NONZERO_RATIO
        result[c] = {
            "ok": ok,
            "nonzero": nonzero,
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio,
            "reason": "OK" if ok else f"too sparse (nonzero={nonzero:.1%})",
        }
    return result


# ═══════════════════════════════════════════════════════════════════════
# 2. EVALUATION (with walk-forward OOS)
# ═══════════════════════════════════════════════════════════════════════

def eval_alpha_expression(expr: str, df: pd.DataFrame):
    """Execute alpha expression string in operator namespace."""
    namespace = {name: getattr(op, name) for name in dir(op) if not name.startswith("_")}
    namespace.update({"df": df, "np": np, "pd": pd, "op": op})
    exec(expr, namespace)
    result = namespace.get("alpha")
    if not isinstance(result, pd.Series):
        return None
    return result.replace([np.inf, -np.inf], np.nan)


def is_valid_series(series) -> tuple[bool, str]:
    if series is None:
        return False, "None series"
    n = len(series)
    if n == 0:
        return False, "empty"
    n_valid = series.dropna().shape[0]
    if n_valid / n < 0.5:
        return False, f"too many NaN ({n-n_valid}/{n})"
    s = series.dropna()
    if s.std() < 1e-9:
        return False, "constant (std≈0)"
    if (s == 0).all():
        return False, "all zeros"
    # Sparsity: > 65% zeros → signal is too sparse for IC to be reliable
    zero_ratio = (s == 0).mean()
    if zero_ratio > 0.65:
        return False, (
            f"signal too sparse ({zero_ratio:.0%} zeros). "
            "Use sentiment as continuous modifier, not binary multiplier."
        )
    return True, "OK"


def eval_one(alpha_def: dict, df: pd.DataFrame,
             fwd_ret: pd.Series, fwd_ret_5d: pd.Series | None = None) -> dict:
    """
    Evaluate one alpha definition with walk-forward OOS metrics.
    Auto-flips if IC_IS is negative (paper mentions directional correction).
    """
    result = deepcopy(alpha_def)
    result.update({
        "ic": None, "ic_oos": None, "ic_5d": None,
        "sharpe": None, "sharpe_oos": None,
        "turnover": None, "score": 0.0,
        "status": "EVAL_ERROR", "series": None,
        "flipped": False, "gp_enhanced": False,
    })

    try:
        series = eval_alpha_expression(alpha_def["expression"], df)
    except Exception as e:
        result["error_reason"] = str(e)[:120]
        return result

    valid, reason = is_valid_series(series)
    if not valid:
        result["error_reason"] = reason
        return result

    # Normalize to zero-mean unit-variance
    norm = (series - series.mean()) / (series.std() + 1e-9)

    # Walk-forward IC
    ic_is, ic_oos = compute_ic_oos(norm, fwd_ret, test_ratio=OOS_TEST_RATIO)

    # Auto-flip: if IS IC is negative, reverse signal
    flipped = False
    if not np.isnan(ic_is) and ic_is < 0:
        norm   = -norm
        ic_is  = -ic_is
        ic_oos = -ic_oos if not np.isnan(ic_oos) else ic_oos
        flipped = True

    # Other metrics (computed on full series for display; OOS used for scoring)
    sharpe     = compute_sharpe(norm, fwd_ret)
    sharpe_oos = compute_sharpe_oos(norm, fwd_ret, test_ratio=OOS_TEST_RATIO)
    turnover   = compute_turnover(norm)

    # 5d IC (full period — indicative only)
    ic_5d = None
    if fwd_ret_5d is not None:
        raw_5d = compute_ic(norm, fwd_ret_5d)
        if not np.isnan(raw_5d):
            ic_5d = round(abs(raw_5d), 6)

    score = composite_score(ic_oos, sharpe_oos, ic_is)

    result.update({
        "ic":         round(ic_is, 6)      if not np.isnan(ic_is)      else None,
        "ic_oos":     round(ic_oos, 6)     if not np.isnan(ic_oos)     else None,
        "ic_5d":      ic_5d,
        "sharpe":     round(sharpe, 4)     if not np.isnan(sharpe)     else None,
        "sharpe_oos": round(sharpe_oos, 4) if not np.isnan(sharpe_oos) else None,
        "turnover":   round(turnover, 4)   if not np.isnan(turnover)   else None,
        "score":      score,
        "status":     "OK",
        "series":     norm,
        "flipped":    flipped,
    })
    return result


# ═══════════════════════════════════════════════════════════════════════
# 3. PROMPT ENGINEERING (improved with RAG + hypothesis-driven)
# ═══════════════════════════════════════════════════════════════════════

OPERATOR_SIGNATURES = """
## Operators — syntax STRICT (wrong arg count = runtime error)

### Time-series (need: series, window)
shift(s, period) | ts_delta(s, period) | ts_delta_ratio(s, period)
ts_mean(s,w) | ts_std(s,w) | ts_sum(s,w) | ts_min(s,w) | ts_max(s,w)
ts_rank(s,w) | ts_median(s,w) | ts_skew(s,w) | ts_kurt(s,w)
ts_zscore_scale(s, w)   ← NEEDS 2 ARGS always
ts_maxmin_scale(s, w)   ← NEEDS 2 ARGS always
ts_corr(s1, s2, w)  ts_cov(s1, s2, w)   ← 3 ARGS
ts_ir(s,w) | ts_linear_reg(s,w) | ts_ema(s, span)
ts_decayed_linear(s,w) | ts_percentile(s,w) | ts_argmaxmin_diff(s,w)
ts_max_diff(s,w) | ts_min_diff(s,w)

### Group-wise (rolling on single series)
grouped_mean(s,w) | grouped_std(s,w) | grouped_demean(s,w)
grouped_zscore_scale(s,w)

### Element-wise
add(s1,s2) | minus(s1,s2) | cwise_mul(s1,s2) | div(s1,s2)
relu(s) | neg(s) | abso(s) | sign(s) | tanh(s) | log(s) | log1p(s)
pow_op(s, exp) | pow_sign(s, exp) | clip(s, lower, upper)
greater(s1, s2) | less(s1, s2) | cwise_max(s1,s2) | cwise_min(s1,s2)
normed_rank_diff(s1, s2)

### Normalize (no window needed)
zscore_scale(s) | winsorize_scale(s) | normed_rank(s)
"""

DATA_FIELDS_TEMPLATE = """
## Data fields available (df.index = trading date)

OHLCV:   df['open'], df['high'], df['low'], df['close'], df['volume']
Tech:    df['SMA_5'], df['SMA_20'], df['EMA_10']
         df['Momentum_3'], df['Momentum_10']
         df['RSI_14'], df['MACD'], df['MACD_Signal']
         df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['OBV']
Sentiment: {sent_list}
  (encoding: -1=negative, 0=neutral, 1=positive. Mostly 0 on most days.)

## Live data statistics (use to calibrate formula magnitude)
{data_stats}
"""

SYSTEM_PROMPT = """Bạn là Senior Quant Researcher tại HOSE (Việt Nam).
Nhiệm vụ: thiết kế formulaic alpha signals dự đoán excess return ngắn hạn.

══════════════════════════════════════════════════
## TRIẾT LÝ ALPHA CHẤT LƯỢNG CAO
══════════════════════════════════════════════════

### 1. Hypothesis-Driven Design (QUAN TRỌNG NHẤT)
Mỗi alpha phải có MARKET HYPOTHESIS rõ ràng:
  "Khi [observable condition], thị trường [inefficiency/behavior],
   dẫn đến [price impact] vì [mechanism]. Formula bắt được bằng [logic]."

Ví dụ TỐT:
  "Khi volume tăng đột biến (z-score > 2) nhưng return ngắn hạn âm, đây là
   forced selling bởi margin call, tạo temporary underpricing → mean reversion.
   Alpha = inverse of (volume spike × negative momentum)."

Ví dụ XẤU:
  "Kết hợp MACD với volume để tạo tín hiệu mạnh hơn." (không có mechanism)

### 2. Signal Architecture
Alpha cần CẢ HAI thành phần:
  [A] CONDITION: khi nào thị trường ở trạng thái khai thác được?
  [B] DIRECTION: lên hay xuống từ đây?

Ví dụ: alpha = [condition_score] × [direction_score]
  condition = ts_std(close, 15)     ← volatility regime
  direction = -ts_delta_ratio(close, 3)  ← mean-reversion direction

### 3. Cấu trúc output chuẩn
Output là CONTINUOUS signal (-∞, +∞):
  - Dương = kỳ vọng giá tăng
  - Âm = kỳ vọng giá giảm
  - Magnitude = độ mạnh của tín hiệu
  
Normalize cuối cùng bằng: ts_zscore_scale(s, w) hoặc tanh(...)

### 4. Phân loại Alpha Families (mỗi alpha thuộc 1 family rõ ràng)
  momentum:       lag-N return, moving average crossover, price acceleration
  mean_reversion: Bollinger bands, RSI extremes, statistical deviation from mean  
  volume_flow:    OBV trend, volume-price divergence, accumulation/distribution
  volatility:     GARCH-proxy, high-low range, return variance regime change
  sentiment_catalyst: news sentiment × technical signal, sentiment divergence
  correlation:    price-volume correlation change, cross-asset lead-lag
  pattern:        intraday range patterns, candle patterns (high-open vs close-low)

### 5. Cách dùng Sentiment ĐÚNG
Sentiment data thường SPARSE: 80-95% ngày = 0 (neutral).
→ Dùng sentiment làm AMPLITUDE MODIFIER, không phải GATE/FILTER.

✗ SAI: cwise_mul(condition, df['VCB_S'])  
   → signal = 0 khi VCB_S = 0 (hầu hết ngày)

✓ ĐÚNG:
   Option A (modifier):
     base_signal = [technical alpha]
     sent_boost  = add(1.0, cwise_mul(0.2, ts_ema(df['SSI_S'], 5)))
     alpha = cwise_mul(base_signal, sent_boost)
   
   Option B (divergence):
     sent_diff = minus(df['VCB_S'], ts_mean(df['VCB_S'], 20))
     alpha = cwise_mul(technical_signal, tanh(ts_zscore_scale(sent_diff, 10)))
   
   Option C (regime):
     sent_regime = ts_mean(df['HPG_S'], 10)
     alpha = cwise_mul(sign(sent_regime), abso(technical_signal))

### 6. Diversity Requirements
Khi sinh nhiều alphas, đảm bảo mỗi cái thuộc FAMILY KHÁC:
  alpha_1: momentum family
  alpha_2: mean_reversion family
  alpha_3: volume_flow family
  alpha_4: volatility family
  alpha_5: sentiment_catalyst hoặc correlation family

### 7. Syntax Rules
- Assign: alpha = <expression>
- Use operators from list only (no if/else, no loops, no lambda)
- add/minus/cwise_mul accept float scalars: add(series, 0.5) is OK
- ts_zscore_scale(s, w) LUÔN cần 2 args
- div(a, b) for safe division (avoids /0)
"""


def build_seed_prompt(ticker: str, sent_cols: list[str], df: pd.DataFrame,
                      sent_quality: dict, memory_block: str = "") -> str:
    """
    Build seed generation prompt.
    Incorporates: data fields, stats, sentiment quality, RAG memory examples.
    """
    industry = TICKER_INDUSTRY.get(ticker, "Khác")
    good_sent = [c for c, q in sent_quality.items() if q["ok"]]
    bad_sent  = [c for c, q in sent_quality.items() if not q["ok"]]

    sent_list = ", ".join(f"df['{c}']" for c in good_sent[:8]) if good_sent else "(none suitable)"
    data_stats = get_data_stats(df)
    fields_block = DATA_FIELDS_TEMPLATE.format(
        sent_list=sent_list, data_stats=data_stats
    )

    # Sentiment guidance section
    if good_sent:
        sent_usable = ", ".join(f"df['{c}']" for c in good_sent[:5])
        same_ind = [c for c in good_sent if TICKER_INDUSTRY.get(c.replace("_S","")) == industry]
        same_ind_str = ", ".join(same_ind[:3]) if same_ind else "none"
        sent_section = f"""
## Sentiment Columns (use as modifier, not gate)
Usable ({len(good_sent)}/{len(sent_quality)}): {sent_usable}
Same industry as {ticker} ({industry}): {same_ind_str}
Skipped ({len(bad_sent)}): {', '.join(bad_sent[:3]) if bad_sent else 'none'}

Reminder: sentiment modulates signal amplitude by ±20-30%.
Technical signal = primary driver. Sentiment = optional amplifier.
"""
    else:
        sent_section = "\n## Sentiment: All columns too sparse → use pure technical alphas.\n"

    return f"""
{fields_block}
{OPERATOR_SIGNATURES}
{memory_block}
{sent_section}

## Task: Generate {SEED_OVERSAMPLE} Alpha Candidates for {ticker} ({industry})

Requirements per alpha:
1. **family**: one of [momentum, mean_reversion, volume_flow, volatility,
                       sentiment_catalyst, correlation, pattern]
2. **hypothesis**: specific market inefficiency + mechanism + why formula captures it
3. **expression**: syntactically correct formula consistent with hypothesis
4. **diversity**: each alpha must be from a DIFFERENT family

Constraints:
- Combine ≥ 2 data sources (e.g., price + volume, technical + sentiment)
- Output must be continuous (>35% non-zero values)
- Different window lengths (try 5, 10, 15, 20, 30)
- No duplicate operator+field combinations across alphas

Output STRICT JSON:
{{
  "alphas": [
    {{
      "id": 1,
      "family": "<family_name>",
      "hypothesis": "<2-3 sentences: condition + mechanism + prediction>",
      "idea": "<concise 1-sentence summary>",
      "expression": "alpha = <formula>"
    }},
    ... × {SEED_OVERSAMPLE}
  ]
}}
"""


def build_refine_prompt(ticker: str, sent_cols: list[str], df: pd.DataFrame,
                        current_results: list[dict], weak: list[tuple],
                        corr_matrix: pd.DataFrame,
                        sent_quality: dict, memory_block: str = "") -> str:
    """Build refinement prompt showing current results and requesting replacements."""
    industry = TICKER_INDUSTRY.get(ticker, "Khác")

    # Summarise current alphas
    alpha_summary = []
    for r in current_results:
        if r["status"] == "OK":
            gp_tag = " [GP✓]" if r.get("gp_enhanced") else ""
            flip_tag = " [flipped]" if r.get("flipped") else ""
            alpha_summary.append(
                f"  α{r['id']} [{r.get('family','?')}]{gp_tag}{flip_tag}\n"
                f"     IC_IS={r.get('ic',0):+.4f}  IC_OOS={r.get('ic_oos',0):+.4f}  "
                f"Sharpe_OOS={r.get('sharpe_oos',0):+.3f}  Score={r.get('score',0):.4f}\n"
                f"     Idea: {r.get('idea','')[:80]}"
            )
        else:
            alpha_summary.append(
                f"  α{r['id']} [ERROR]: {r.get('error_reason','?')[:80]}"
            )

    weak_summary = []
    for aid, reason in weak:
        r = next((x for x in current_results if x["id"] == aid), {})
        weak_summary.append(
            f"  REPLACE α{aid} (family={r.get('family','?')}): {reason}"
        )

    keep_ids = [r["id"] for r in current_results if r["id"] not in [w[0] for w in weak]]
    keep_families = [r.get("family", "?") for r in current_results
                     if r["id"] in keep_ids and r["status"] == "OK"]
    needed_families = ["momentum", "mean_reversion", "volume_flow",
                       "volatility", "sentiment_catalyst", "correlation", "pattern"]
    available_families = [f for f in needed_families if f not in keep_families]

    corr_text = corr_matrix.to_string() if not corr_matrix.empty else "N/A"
    good_sent = [c for c, q in sent_quality.items() if q["ok"]]
    sent_list = ", ".join(f"df['{c}']" for c in good_sent[:5]) or "(none)"

    data_stats = get_data_stats(df)
    fields_block = DATA_FIELDS_TEMPLATE.format(sent_list=sent_list, data_stats=data_stats)

    return f"""
## Refinement Round — {ticker} ({industry})

### Current Portfolio
{chr(10).join(alpha_summary)}

### Correlation Matrix
{corr_text}

### Alphas to Replace (with reasons)
{chr(10).join(weak_summary)}

### Families already covered (keep): {', '.join(keep_families)}
### Families available for replacement: {', '.join(available_families[:len(weak)])}

---
{fields_block}
{OPERATOR_SIGNATURES}
{memory_block}

## Task: Generate {len(weak)} Replacement Alphas

IDs to replace: {[w[0] for w in weak]}
Preferred families: {available_families[:len(weak)]}

Requirements:
- Each replacement must be from a DIFFERENT family than existing alphas
- Include specific hypothesis (condition + mechanism + prediction)
- IC_OOS > 0.02 is the target (we use walk-forward evaluation)
- Expression must be consistent with hypothesis

Output JSON:
{{
  "alphas": [
    {{"id": <id_to_replace>, "family": "...", "hypothesis": "...",
      "idea": "...", "expression": "alpha = ..."}}
  ]
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# 4. SELECTION & WEAK ALPHA IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════

def compute_corr_matrix(results: list[dict]) -> pd.DataFrame:
    ok = [r for r in results if r["status"] == "OK" and r.get("series") is not None]
    if not ok:
        return pd.DataFrame()
    df_s = pd.DataFrame({f"alpha_{r['id']}": r["series"] for r in ok})
    return df_s.corr(method="spearman").round(3)


def select_best_alphas(results: list[dict], n: int = 5) -> list[dict]:
    """
    Greedy selection: highest OOS score + low pairwise correlation.
    Prioritises diversity across alpha families.
    """
    ok = [r for r in results if r["status"] == "OK"]
    ok.sort(key=lambda r: r.get("score", 0), reverse=True)

    selected = []
    selected_families = set()

    # First pass: pick best from each family
    for cand in ok:
        if len(selected) >= n:
            break
        family = cand.get("family", "other")
        if family in selected_families:
            continue
        c_s = cand.get("series")
        if c_s is None:
            continue
        corr_ok = all(
            abs(pd.concat([c_s, e.get("series")], axis=1).dropna().corr(method="spearman").iloc[0, 1]) < CORR_THRESHOLD
            for e in selected if e.get("series") is not None
        )
        if corr_ok:
            selected.append(cand)
            selected_families.add(family)

    # Second pass: fill remaining slots with best remaining (any family)
    for cand in ok:
        if len(selected) >= n:
            break
        if cand in selected:
            continue
        c_s = cand.get("series")
        if c_s is None:
            continue
        corr_ok = all(
            abs(pd.concat([c_s, e.get("series")], axis=1).dropna().corr(method="spearman").iloc[0, 1]) < CORR_THRESHOLD
            for e in selected if e.get("series") is not None
        )
        if corr_ok:
            selected.append(cand)

    # Re-assign IDs 1-5
    for i, r in enumerate(selected):
        r["id"] = i + 1

    return selected[:n]


def identify_weak_alphas(results: list[dict], corr_matrix: pd.DataFrame,
                         fwd_ret: pd.Series | None = None) -> list[tuple]:
    """
    Identify alphas that need replacement.
    Uses OOS metrics for decisions (not IS).
    """
    from core.backtester import detect_decay

    weak = []
    weak_ids = set()

    # 1. EVAL errors
    for r in results:
        if r["status"] == "EVAL_ERROR":
            weak.append((r["id"], f"EVAL_ERROR: {r.get('error_reason','?')[:60]}"))
            weak_ids.add(r["id"])

    # 2. Weak OOS IC
    for r in results:
        if r["id"] in weak_ids or r["status"] != "OK":
            continue
        ic_oos = abs(r.get("ic_oos") or 0.0)
        sh_oos = r.get("sharpe_oos") or 0.0
        if ic_oos < IC_THRESHOLD and sh_oos < SHARPE_THRESHOLD:
            weak.append((r["id"],
                f"OOS too weak: IC_OOS={ic_oos:.4f} < {IC_THRESHOLD}, "
                f"Sharpe_OOS={sh_oos:.3f} < {SHARPE_THRESHOLD}. "
                f"Try different operator family or longer window."))
            weak_ids.add(r["id"])

    # 3. Severely negative Sharpe_OOS
    for r in results:
        if r["id"] in weak_ids or r["status"] != "OK":
            continue
        sh_oos = r.get("sharpe_oos") or 0.0
        if sh_oos < -0.5:
            weak.append((r["id"],
                f"Sharpe_OOS={sh_oos:.3f} severely negative → opposite direction"))
            weak_ids.add(r["id"])

    # 4. High correlation (keep higher-score, replace lower)
    if not corr_matrix.empty:
        ok_results = {r["id"]: r for r in results if r["status"] == "OK"}
        ids = list(ok_results.keys())
        for i, id_i in enumerate(ids):
            for id_j in ids[i+1:]:
                col_i, col_j = f"alpha_{id_i}", f"alpha_{id_j}"
                if col_i not in corr_matrix.columns or col_j not in corr_matrix.columns:
                    continue
                corr_val = abs(corr_matrix.loc[col_i, col_j])
                if corr_val < CORR_THRESHOLD:
                    continue
                sc_i = ok_results[id_i].get("score", 0)
                sc_j = ok_results[id_j].get("score", 0)
                loser = id_i if sc_i <= sc_j else id_j
                winner = id_j if loser == id_i else id_i
                if loser not in weak_ids:
                    weak.append((loser,
                        f"|corr|={corr_val:.3f} with α{winner} ≥ {CORR_THRESHOLD}. "
                        "Need a completely different data source."))
                    weak_ids.add(loser)

    # 5. IC decay
    if fwd_ret is not None:
        for r in results:
            if r["id"] in weak_ids or r["status"] != "OK":
                continue
            s = r.get("series")
            if s is None:
                continue
            try:
                d = detect_decay(s, fwd_ret)
                if d.get("decaying"):
                    weak.append((r["id"],
                        f"Decaying IC: {d.get('hist_ic',0):+.4f} → {d.get('recent_ic',0):+.4f} "
                        f"(−{d.get('drop_pct',0):.0f}%)"))
                    weak_ids.add(r["id"])
            except Exception:
                pass

    return weak


def should_replace(old_r: dict, new_r: dict, fwd_ret: pd.Series | None = None) -> tuple[bool, str]:
    """Replace only if new OOS score clearly better."""
    if new_r.get("status") != "OK":
        return False, "new is ERROR"
    old_score = old_r.get("score", 0.0) if old_r.get("status") == "OK" else 0.0
    new_score = new_r.get("score", 0.0)
    if new_score > old_score:
        return True, f"score {old_score:.4f} → {new_score:.4f}"
    # Decay exception: allow small score drop if old alpha is decaying
    if fwd_ret is not None and old_r.get("status") == "OK":
        from core.backtester import detect_decay
        old_s = old_r.get("series")
        new_s = new_r.get("series")
        if old_s is not None and new_s is not None:
            try:
                old_d = detect_decay(old_s, fwd_ret)
                new_d = detect_decay(new_s, fwd_ret)
                if old_d.get("decaying") and not new_d.get("decaying") and new_score >= old_score * 0.85:
                    return True, f"decay fixed, score {old_score:.4f}→{new_score:.4f}"
            except Exception:
                pass
    return False, f"no improvement ({new_score:.4f} ≤ {old_score:.4f})"


# ═══════════════════════════════════════════════════════════════════════
# 5. LLM CALLS
# ═══════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, expected: int, temperature: float | None = None) -> list[dict]:
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temp,
                max_tokens=LLM_MAX_TOKENS,
            )
            data = json.loads(resp.choices[0].message.content)
            alphas = data.get("alphas", [])
            if len(alphas) >= expected:
                return alphas[:expected]
            log.warning(f"  LLM returned {len(alphas)}/{expected}, retrying...")
        except Exception as e:
            log.warning(f"  LLM attempt {attempt+1} failed: {e}")
    raise RuntimeError(f"LLM failed after 3 attempts (need {expected} alphas)")


def rescue_errors(results: list[dict], ticker: str, sent_cols: list[str],
                  df: pd.DataFrame, fwd_ret: pd.Series,
                  fwd_ret_5d: pd.Series | None, sent_quality: dict) -> list[dict]:
    """Retry EVAL_ERROR alphas with a simpler, error-aware prompt."""
    error_results = [r for r in results if r["status"] == "EVAL_ERROR"]
    if not error_results:
        return results

    error_details = "\n".join(
        f"  α{r['id']}: ERROR={r.get('error_reason','?')[:80]}\n"
        f"    Previous: {r.get('expression','')[:80]}"
        for r in error_results
    )

    good_sent = [c for c, q in sent_quality.items() if q["ok"]]
    sent_list = ", ".join(f"df['{c}']" for c in good_sent[:4]) or "(none)"
    data_stats = get_data_stats(df)

    rescue_prompt = f"""
## Rescue: Fix {len(error_results)} failed alphas for {ticker}

### Failed expressions
{error_details}

### Common errors to fix:
- ts_zscore_scale(s) with 1 arg → must be ts_zscore_scale(s, 20)
- division by series that can be zero → use div(a, b) not a/b
- series all NaN → use operators that produce non-NaN output
- expression references undefined variable → use df['col'] format

### Available fields
OHLCV: df['close'], df['volume'], df['high'], df['low']
Tech: df['RSI_14'], df['MACD'], df['MACD_Signal'], df['BB_Upper'], df['BB_Lower'], df['OBV']
      df['SMA_20'], df['EMA_10'], df['Momentum_3'], df['Momentum_10']
Sentiment: {sent_list}

{OPERATOR_SIGNATURES}

## Task: Write {len(error_results)} SIMPLE reliable alphas (easier expressions, 2 components max)
IDs: {[r['id'] for r in error_results]}

Output JSON:
{{"alphas": [{{"id": <id>, "family": "...", "hypothesis": "...", "idea": "...", "expression": "alpha = ..."}}]}}
"""

    for attempt in range(3):
        error_ids = [r["id"] for r in results if r["status"] == "EVAL_ERROR"]
        if not error_ids:
            break
        try:
            new_defs = call_llm(rescue_prompt, expected=len(error_ids), temperature=0.4)
            for nd in new_defs:
                nr = eval_one(nd, df, fwd_ret, fwd_ret_5d)
                if nr["status"] == "OK":
                    idx = next((i for i, r in enumerate(results) if r["id"] == nr["id"]), None)
                    if idx is not None:
                        results[idx] = nr
                        log.info(f"  [RESCUED] α{nr['id']} IC_OOS={nr.get('ic_oos',0):+.4f}")
        except Exception as e:
            log.warning(f"  Rescue attempt {attempt+1} failed: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# 6. FALLBACK ALPHAS (reliable technical baselines)
# ═══════════════════════════════════════════════════════════════════════

FALLBACK_ALPHAS = [
    {
        "id": 99, "family": "momentum",
        "hypothesis": "Price acceleration over multiple timeframes captures trend strength.",
        "idea": "Multi-timeframe momentum divergence",
        "expression": "alpha = minus(ts_zscore_scale(df['Momentum_3'], 10), ts_zscore_scale(df['Momentum_10'], 20))"
    },
    {
        "id": 99, "family": "mean_reversion",
        "hypothesis": "RSI extremes signal mean reversion in short-term price movements.",
        "idea": "RSI mean-reversion signal",
        "expression": "alpha = tanh(ts_zscore_scale(minus(50.0, df['RSI_14']), 15))"
    },
    {
        "id": 99, "family": "volume_flow",
        "hypothesis": "Volume-price divergence reveals distribution/accumulation phases.",
        "idea": "Volume-price rank divergence",
        "expression": "alpha = minus(ts_rank(df['volume'], 20), ts_rank(ts_delta(df['close'], 5), 20))"
    },
    {
        "id": 99, "family": "volatility",
        "hypothesis": "Volatility regime shifts predict subsequent price direction via risk-on/off.",
        "idea": "Volatility-adjusted price position",
        "expression": "alpha = div(minus(df['close'], df['BB_Middle']), add(minus(df['BB_Upper'], df['BB_Lower']), 0.01))"
    },
    {
        "id": 99, "family": "correlation",
        "hypothesis": "MACD histogram acceleration captures trend momentum changes.",
        "idea": "MACD histogram acceleration",
        "expression": "alpha = ts_zscore_scale(ts_delta(minus(df['MACD'], df['MACD_Signal']), 3), 10)"
    },
]


# ═══════════════════════════════════════════════════════════════════════
# 7. HELPERS
# ═══════════════════════════════════════════════════════════════════════

def strip_series(results: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "series"} for r in results]


def log_round(ticker: str, round_num: int, results: list[dict],
              corr: pd.DataFrame) -> None:
    log.info(f"\n{'='*65}")
    log.info(f"[{ticker}] ROUND {round_num}")
    log.info(f"{'='*65}")
    for r in results:
        if r["status"] == "OK":
            tags = ""
            if r.get("flipped"):     tags += " [↕flip]"
            if r.get("gp_enhanced"): tags += " [GP✓]"
            log.info(
                f"  α{r['id']} [{r.get('family','?')}]{tags}\n"
                f"    IC_IS={r.get('ic',0):+.4f}  IC_OOS={r.get('ic_oos',0):+.4f}  "
                f"Sh_OOS={r.get('sharpe_oos',0):+.3f}  Score={r.get('score',0):.4f}\n"
                f"    {r.get('idea','')[:80]}"
            )
        else:
            log.info(f"  α{r['id']} [ERROR] {r.get('error_reason','')[:60]}")
    if not corr.empty:
        log.info(f"\n  Correlation:\n{corr.to_string()}")


# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def _run_pipeline(ticker: str, max_rounds: int = MAX_ROUNDS,
                  refine_only: bool = False) -> dict:

    out_json = os.path.join(ALPHA_FORMULA_DIR, f"{ticker}_alphas.json")
    out_csv  = os.path.join(ALPHA_VALUES_DIR,  f"{ticker}_alpha_values.csv")

    df         = load_data(ticker)
    fwd_ret    = make_forward_return(df, horizon=1)
    fwd_ret_5d = make_forward_return(df, horizon=5)
    sent_cols  = [c for c in df.columns if c.endswith("_S")]
    sent_quality = get_sentiment_quality(sent_cols, df)

    good_sent = [c for c, q in sent_quality.items() if q["ok"]]
    log.info(f"[{ticker}] Sentiment: {len(good_sent)}/{len(sent_cols)} usable cols")

    # ── RAG: retrieve relevant examples from memory ──────────────────
    # Paper: "Demonstration Retrieval" — find similar successful alphas
    mem_examples = memory.retrieve_diverse(ticker=ticker, top_k=MEMORY_RETRIEVE_K)
    memory_block = compile_memory_block(mem_examples, label="Historical Alpha Examples")
    log.info(f"[{ticker}] Memory: retrieved {len(mem_examples)} examples")

    # ── Phase 1: Seed generation ──────────────────────────────────────
    if refine_only and os.path.exists(out_json):
        log.info(f"[{ticker}] Loading existing alphas for refinement-only mode")
        with open(out_json) as f:
            prev = json.load(f)
        existing_defs = [
            {"id": a["id"], "family": a.get("family","?"),
             "hypothesis": a.get("hypothesis",""), "idea": a["idea"],
             "expression": a["expression"]}
            for a in prev.get("alphas", [])
        ]
        results = [eval_one(a, df, fwd_ret, fwd_ret_5d) for a in existing_defs]
    else:
        log.info(f"\n[{ticker}] === ROUND 1: Seed ({SEED_OVERSAMPLE} candidates) ===")
        seed_prompt = build_seed_prompt(ticker, sent_cols, df, sent_quality, memory_block)
        seed_defs = call_llm(seed_prompt, expected=SEED_OVERSAMPLE)

        # Evaluate seeds
        all_results = [eval_one(a, df, fwd_ret, fwd_ret_5d) for a in seed_defs]

        for r in all_results:
            tag = "OK" if r["status"] == "OK" else "ERR"
            extra = (f"IC_OOS={r.get('ic_oos',0):+.4f} Sc={r.get('score',0):.4f}"
                     if r["status"] == "OK" else r.get("error_reason","?")[:50])
            log.info(f"  [{tag}] α{r.get('id','?')} [{r.get('family','?')}] {extra}")

        # ── Phase 1b: GP Enhancement ─────────────────────────────────
        # Paper: "Alpha Compute Framework → genetic programming"
        if GP_ENABLED:
            log.info(f"[{ticker}] Running GP enhancement on {len(all_results)} seeds...")
            all_results = enhance_alpha_population(
                all_results, df, fwd_ret, fwd_ret_5d,
                eval_fn=eval_one, n_iterations=GP_ITERATIONS
            )

        # Select best 5 (diversity-aware)
        results = select_best_alphas(all_results, n=5)

    # Supplement if < 5
    attempts = 0
    while len(results) < 5 and attempts < 4:
        attempts += 1
        n_need = 5 - len(results)
        existing_ideas = [r.get("idea","") for r in results if r["status"] == "OK"]
        existing_families = [r.get("family","?") for r in results if r["status"] == "OK"]
        error_reasons = list(set(
            r.get("error_reason","?") for r in (all_results if not refine_only else results)
            if r["status"] == "EVAL_ERROR"
        ))[:3]

        supp_prompt = build_seed_prompt(ticker, sent_cols, df, sent_quality, memory_block)
        supp_prompt += f"\n\nAlready have: {existing_families}\n"
        supp_prompt += f"Previous errors to avoid: {error_reasons}\n"
        supp_prompt += f"Generate EXACTLY {n_need} more alphas from UNUSED families.\n"

        try:
            supp_defs = call_llm(supp_prompt, expected=n_need, temperature=0.9)
            for sd in supp_defs:
                sd["id"] = len(results) + 1
                sr = eval_one(sd, df, fwd_ret, fwd_ret_5d)
                if sr["status"] == "OK":
                    results.append(sr)
                    log.info(f"  [SUPP] α{sr['id']} IC_OOS={sr.get('ic_oos',0):+.4f}")
                if len(results) >= 5:
                    break
        except Exception as e:
            log.warning(f"  Supplement attempt {attempts} failed: {e}")

    # Fallback: use reliable technical alphas if still < 5
    if len(results) < 5:
        for fb in FALLBACK_ALPHAS:
            if len(results) >= 5:
                break
            existing_families = {r.get("family","?") for r in results}
            if fb["family"] in existing_families:
                continue
            fb_copy = deepcopy(fb)
            fb_copy["id"] = len(results) + 1
            fb_r = eval_one(fb_copy, df, fwd_ret, fwd_ret_5d)
            if fb_r["status"] == "OK":
                results.append(fb_r)
                log.info(f"  [FALLBACK] α{fb_r['id']} family={fb['family']} IC_OOS={fb_r.get('ic_oos',0):+.4f}")

    corr = compute_corr_matrix(results)
    log_round(ticker, 1, results, corr)
    history = [{"round": 1, "alphas": strip_series(results)}]

    # ── Refinement rounds ─────────────────────────────────────────────
    for rnd in range(2, max_rounds + 1):
        weak = identify_weak_alphas(results, corr, fwd_ret=fwd_ret)
        if not weak:
            log.info(f"[{ticker}] All alphas OK — stopping at round {rnd - 1}")
            break

        log.info(f"\n[{ticker}] === ROUND {rnd}: Refinement ({len(weak)} weak) ===")
        for aid, reason in weak:
            log.info(f"  → Replace α{aid}: {reason[:70]}")

        # RAG: retrieve memory relevant to what we're trying to improve
        query_idea = " ".join(r.get("idea","") for r in results
                              if r.get("id") in [w[0] for w in weak])
        refine_memories = memory.retrieve(query_idea, ticker=ticker, top_k=2)
        refine_memory_block = compile_memory_block(refine_memories, "Refinement Examples")

        replaced = 0
        keep_ids = {r["id"] for r in results if r["id"] not in [w[0] for w in weak]}
        pending = list(weak)

        for attempt in range(1, REFINE_ATTEMPTS + 1):
            if not pending:
                break
            try:
                prompt = build_refine_prompt(
                    ticker, sent_cols, df, results, pending, corr,
                    sent_quality, refine_memory_block
                )
                new_defs = call_llm(prompt, expected=len(pending),
                                    temperature=min(1.0, LLM_TEMPERATURE + 0.1 * (attempt - 1)))
            except RuntimeError as e:
                log.warning(f"  Refine attempt {attempt} LLM failed: {e}")
                continue

            # GP enhance refinement candidates too
            new_evals = [eval_one(nd, df, fwd_ret, fwd_ret_5d) for nd in new_defs]
            if GP_ENABLED:
                new_evals = enhance_alpha_population(
                    new_evals, df, fwd_ret, fwd_ret_5d,
                    eval_fn=eval_one, n_iterations=10  # fewer iterations in refinement
                )

            unresolved = {w[0] for w in pending}
            for nr in new_evals:
                tid = nr["id"]
                old_r = next((r for r in results if r["id"] == tid), None)
                if old_r is None:
                    continue
                can, reason = should_replace(old_r, nr, fwd_ret)
                if not can:
                    continue
                # Corr check against kept alphas
                nr_s = nr.get("series")
                corr_ok = True
                if nr_s is not None:
                    for kid in keep_ids:
                        kept = next((r for r in results if r["id"] == kid), None)
                        if kept is None or kept.get("series") is None:
                            continue
                        m = pd.concat([nr_s, kept["series"]], axis=1).dropna()
                        if len(m) >= 10:
                            c_val = abs(m.iloc[:,0].corr(m.iloc[:,1], method="spearman"))
                            if c_val >= CORR_THRESHOLD:
                                corr_ok = False
                                break
                if corr_ok:
                    idx = next(i for i, r in enumerate(results) if r["id"] == tid)
                    results[idx] = nr
                    keep_ids.add(tid)
                    replaced += 1
                    unresolved.discard(tid)
                    log.info(f"  [REPLACED] α{tid}: {reason}")

            pending = [w for w in pending if w[0] in unresolved]

        corr = compute_corr_matrix(results)
        log_round(ticker, rnd, results, corr)
        history.append({"round": rnd, "replaced": replaced, "alphas": strip_series(results)})

    # ── Rescue EVAL_ERRORs ────────────────────────────────────────────
    results = rescue_errors(results, ticker, sent_cols, df,
                            fwd_ret, fwd_ret_5d, sent_quality)

    # ── Save to memory (Knowledge Library update) ────────────────────
    # Paper: successful alphas fed back into memory for future retrieval
    ok_count = 0
    for r in results:
        if r["status"] == "OK":
            memory.store(ticker, r)
            ok_count += 1
    log.info(f"[{ticker}] Stored {ok_count} alphas to memory")

    # ── Save alpha values CSV ─────────────────────────────────────────
    series_dict = {}
    for r in results:
        col = f"alpha_{r['id']}"
        series_dict[col] = (
            r["series"] if r["status"] == "OK" and r.get("series") is not None
            else pd.Series(0.0, index=df.index)
        )
    df_vals = pd.DataFrame(series_dict, index=df.index)
    df_vals.index.name = "time"
    df_vals.to_csv(out_csv)

    # ── Save formula JSON ─────────────────────────────────────────────
    final = strip_series(results)
    output = {
        "ticker":   ticker,
        "n_rows":   len(df),
        "n_rounds": len(history),
        "alphas":   final,
        "history":  history,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Summary log ──────────────────────────────────────────────────
    ok = [r for r in final if r["status"] == "OK"]
    log.info(f"\n{'='*65}")
    log.info(f"[{ticker}] FINAL: {len(ok)}/5 OK after {len(history)} rounds")
    log.info(f"{'='*65}")
    for r in ok:
        gp_tag = " [GP]" if r.get("gp_enhanced") else ""
        log.info(
            f"  α{r['id']} [{r.get('family','?')}]{gp_tag}  "
            f"IC_IS={r.get('ic',0):+.4f}  IC_OOS={r.get('ic_oos',0):+.4f}  "
            f"Sh_OOS={r.get('sharpe_oos',0):+.3f}  Score={r.get('score',0):.4f}\n"
            f"    {r.get('idea','')}"
        )
    return output


def run_single(ticker: str, max_rounds: int = MAX_ROUNDS, force: bool = False) -> dict:
    out_json = os.path.join(ALPHA_FORMULA_DIR, f"{ticker}_alphas.json")
    if os.path.exists(out_json) and not force:
        log.info(f"[{ticker}] Skipping (exists). Use --force to regenerate.")
        return json.load(open(out_json))
    return _run_pipeline(ticker, max_rounds=max_rounds, refine_only=False)


def run_single_refine(ticker: str, max_rounds: int = MAX_ROUNDS) -> dict:
    out_json = os.path.join(ALPHA_FORMULA_DIR, f"{ticker}_alphas.json")
    if not os.path.exists(out_json):
        log.warning(f"[{ticker}] No existing alphas — falling back to full generation.")
        return _run_pipeline(ticker, max_rounds=max_rounds, refine_only=False)
    return _run_pipeline(ticker, max_rounds=max_rounds, refine_only=True)


def run_all(max_rounds: int = MAX_ROUNDS, force: bool = False) -> None:
    results = []
    for ticker in VN30_SYMBOLS:
        try:
            out = run_single(ticker, max_rounds=max_rounds, force=force)
            ok = [a for a in out["alphas"] if a["status"] == "OK"]
            avg_ic = (np.mean([abs(a.get("ic_oos") or a.get("ic") or 0) for a in ok])
                      if ok else 0.0)
            results.append({"ticker": ticker, "ok": len(ok), "avg_ic_oos": round(avg_ic, 4)})
        except Exception as e:
            log.error(f"[{ticker}] FAILED: {e}")
            results.append({"ticker": ticker, "ok": 0, "avg_ic_oos": 0.0})

    log.info("\n" + "="*65 + "\nVN30 SUMMARY\n" + "="*65)
    for r in sorted(results, key=lambda x: -x["avg_ic_oos"]):
        log.info(f"  {r['ticker']:5s}  {r['ok']}/5  avg_IC_OOS={r['avg_ic_oos']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha-GPT v3")
    parser.add_argument("--ticker",      type=str)
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--rounds",      type=int, default=MAX_ROUNDS)
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--refine-only", action="store_true")
    parser.add_argument("--no-gp",       action="store_true",
                        help="Disable GP enhancement (faster)")
    args = parser.parse_args()

    if args.no_gp:
        GP_ENABLED = False

    if args.ticker:
        t = args.ticker.upper()
        if args.refine_only:
            run_single_refine(t, max_rounds=args.rounds)
        else:
            run_single(t, max_rounds=args.rounds, force=args.force)
    elif args.all:
        run_all(max_rounds=args.rounds, force=args.force)
    else:
        parser.print_help()