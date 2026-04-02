"""
Bước 2.2 — Gen Alpha với Iterative Refinement (v2)
====================================================
Cải tiến so với v1:
  - Sinh 8 alpha candidates → chọn 5 tốt nhất (oversampling)
  - Auto-flip alpha có IC âm (neg()) → đảm bảo directional đúng
  - Đưa data statistics vào prompt (ranges, std) → LLM hiểu data thực
  - Operator signatures đầy đủ trong prompt → tránh lỗi wrong arg count
  - Multi-horizon IC (1d + 5d) → alpha robust hơn
  - Composite score (IC × Sharpe) cho selection/replacement
  - 5 refinement rounds, temperature 0.7 cho diversity
  - Rescue EVAL_ERROR bằng LLM retry với error-aware prompt

Output:
    alphas/<TICKER>_alphas.json       — definitions + metrics + history
    alphas/<TICKER>_alpha_values.csv  — 5 cột alpha values (features cho transformer)

Chạy:
    python pipelines/gen_alpha.py --ticker ACB
    python pipelines/gen_alpha.py --ticker ACB --rounds 5
    python pipelines/gen_alpha.py --all
    python pipelines/gen_alpha.py --all --force
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.universe import VN30_SYMBOLS
from core.paths import FEATURES_DIR, SENTIMENT_OUTPUT_DIR, ALPHA_VALUES_DIR, ALPHA_FORMULA_DIR
from core import alpha_operators as op

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===================================================================
# CONFIG
# ===================================================================
SENTIMENT_DIR = SENTIMENT_OUTPUT_DIR
ALPHA_DIR = ALPHA_VALUES_DIR
ALPHA_FORMULA = ALPHA_FORMULA_DIR
os.makedirs(ALPHA_DIR, exist_ok=True)
os.makedirs(ALPHA_FORMULA, exist_ok=True)

MAX_ROUNDS       = 5      # seed + 4 refinement rounds (tăng từ 3)
SEED_OVERSAMPLE  = 10     # sinh 10 candidates → chọn 5 tốt nhất + diverse nhất
IC_THRESHOLD     = 0.015  # |IC| < ngưỡng này → alpha yếu (hạ từ 0.02)
CORR_THRESHOLD   = 0.65   # |corr| >= ngưỡng này → trùng lặp (hạ từ 0.70)
LLM_MODEL        = "gpt-4o-mini"
LLM_TEMPERATURE  = 0.7    # tăng từ 0.3 cho diversity
LLM_MAX_TOKENS   = 4000   # tăng từ 2500
REFINE_ATTEMPTS_PER_ROUND = 3  # mỗi vòng refinement thử nhiều lần nếu chưa thay được

# ===================================================================
# BẢN ĐỒ NGÀNH VN30 (INDUSTRY MAPPING)
# ===================================================================
TICKER_INDUSTRY = {
    "ACB": "Ngân hàng", "BID": "Ngân hàng", "CTG": "Ngân hàng", "HDB": "Ngân hàng",
    "LPB": "Ngân hàng", "MBB": "Ngân hàng", "SHB": "Ngân hàng", "SSB": "Ngân hàng",
    "STB": "Ngân hàng", "TCB": "Ngân hàng", "TPB": "Ngân hàng", "VCB": "Ngân hàng",
    "VIB": "Ngân hàng", "VPB": "Ngân hàng",
    "VHM": "Bất động sản", "VIC": "Bất động sản", "VRE": "Bất động sản", "VPL": "Bất động sản",
    "MWG": "Bán lẻ", "FPT": "Công nghệ", 
    "MSN": "Tiêu dùng/Thực phẩm", "SAB": "Tiêu dùng/Thực phẩm", "VNM": "Tiêu dùng/Thực phẩm",
    "HPG": "Thép/Vật liệu", "GAS": "Dầu khí", "PLX": "Dầu khí",
    "DGC": "Hóa chất", "GVR": "Hóa chất/Cao su",
    "VJC": "Hàng không", "SSI": "Chứng khoán"
}

# ===================================================================
# 1. DATA LOADING
# ===================================================================

def load_data(ticker: str) -> pd.DataFrame:
    feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
    sent_path = os.path.join(SENTIMENT_DIR, f"{ticker}_Full_Sentiment.csv")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Không tìm thấy features: {feat_path}")
    if not os.path.exists(sent_path):
        raise FileNotFoundError(f"Không tìm thấy sentiment: {sent_path}")

    df_feat = pd.read_csv(feat_path)
    df_feat["time"] = pd.to_datetime(df_feat["time"]).dt.normalize()
    df_feat = df_feat.set_index("time").sort_index()

    df_sent = pd.read_csv(sent_path, index_col="time", parse_dates=True)
    df_sent.index = pd.to_datetime(df_sent.index).normalize()
    df_sent = df_sent.sort_index()

    df = df_feat.join(df_sent, how="inner")
    df = df.dropna(subset=["close"]).fillna(0.0)

    log.info(f"[{ticker}] Loaded: {len(df)} rows, {len(df.columns)} cols")
    return df


def make_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    return df["close"].pct_change(horizon).shift(-horizon).rename(f"fwd_ret_{horizon}d")


def compute_data_stats(df: pd.DataFrame) -> str:
    """Tóm tắt statistics để LLM hiểu ranges thực."""
    cols = ["close", "volume", "RSI_14", "MACD", "BB_Upper", "BB_Lower", "OBV",
            "Momentum_3", "Momentum_10"]
    lines = []
    for c in cols:
        if c in df.columns:
            s = df[c].dropna()
            lines.append(f"  {c}: mean={s.mean():.2f}, std={s.std():.2f}, "
                         f"range=[{s.min():.2f}, {s.max():.2f}]")
    return "\n".join(lines)


# ===================================================================
# 2. EVAL
# ===================================================================

def eval_alpha_expression(expr: str, df: pd.DataFrame) -> pd.Series | None:
    namespace = {name: getattr(op, name) for name in dir(op) if not name.startswith("_")}
    namespace["op"] = op
    namespace["df"] = df
    namespace["np"] = np
    namespace["pd"] = pd
    try:
        exec(expr, namespace)
        result = namespace.get("alpha")
        if not isinstance(result, pd.Series):
            log.warning(f"  Expression trả về {type(result)}, bỏ qua.")
            return None
        result = result.replace([np.inf, -np.inf], np.nan)
        return result
    except Exception as e:
        log.warning(f"  Eval lỗi: {e}")
        log.debug(f"  Expression: {expr}")
        return None


def compute_ic(alpha: pd.Series, fwd_ret: pd.Series) -> float:
    c = pd.concat([alpha, fwd_ret], axis=1).dropna()
    if len(c) < 30:
        return np.nan
    return float(c.iloc[:, 0].corr(c.iloc[:, 1], method="spearman"))


def compute_sharpe(alpha: pd.Series, fwd_ret: pd.Series) -> float:
    c = pd.concat([alpha, fwd_ret], axis=1).dropna()
    c.columns = ["a", "r"]
    if len(c) < 30:
        return np.nan
    pos = np.where(c["a"] > c["a"].median(), 1.0, -1.0)
    pnl = pos * c["r"]
    if pnl.std() < 1e-9:
        return np.nan
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def compute_turnover(alpha: pd.Series) -> float:
    scale = alpha.abs().mean()
    if scale < 1e-9:
        return np.nan
    return float(alpha.diff().abs().mean() / scale)


def composite_score(ic: float | None, sharpe: float | None) -> float:
    """IC (0.6) + Sharpe chuẩn hoá (0.4) cho ranking."""
    _ic = abs(ic) if ic is not None else 0.0
    _sh = sharpe if sharpe is not None else 0.0
    return 0.6 * _ic + 0.4 * max(_sh / 5.0, 0.0)


def is_valid_series(series, min_valid_ratio: float = 0.5):
    if series is None:
        return False, "series là None"
    n_total = len(series)
    if n_total == 0:
        return False, "series rỗng"
    n_valid = series.dropna().shape[0]
    if n_valid / n_total < min_valid_ratio:
        return False, f"quá nhiều NaN ({n_total - n_valid}/{n_total} rows)"
    s_clean = series.dropna()
    if s_clean.std() < 1e-9:
        return False, "series constant (std≈0)"
    if (s_clean == 0).all():
        return False, "series toàn 0"
    if np.isinf(s_clean).any():
        return False, "series chứa inf"
    # Sparsity check: signal quá thưa → Sharpe âm, IC không robust
    zero_ratio = (s_clean == 0).mean()
    if zero_ratio > 0.6:
        return False, (f"signal quá thưa ({zero_ratio:.0%} = 0). "
                       f"Nguyên nhân thường gặp: nhân binary × sparse sentiment. "
                       f"Dùng sentiment như modifier liên tục thay vì multiplier.")
    return True, "OK"


def _check_idea_expression_consistency(idea: str, expression: str) -> tuple[bool, str]:
    """
    Validate: nếu idea mention sentiment thì expression phải chứa _S column.
    Returns (is_consistent, reason).
    """
    idea_lower = idea.lower()
    # Check if idea mentions sentiment-related keywords
    sent_keywords = ['sentiment', '_s', 'cảm xúc', 'tin tức', 'tâm lý thị trường']
    # Check for specific ticker sentiment mentions like "LPB", "VCB" in sentiment context
    idea_mentions_sentiment = any(kw in idea_lower for kw in sent_keywords)

    # Also check for ticker_S patterns like "LPB" near "sentiment" or "tín hiệu"
    import re
    ticker_mention = re.findall(r'\b([A-Z]{2,4})\b', idea)
    for t in ticker_mention:
        if f"{t}_S" not in expression and any(
            kw in idea_lower for kw in [t.lower(), f'sentiment từ {t.lower()}',
                                         f'sentiment {t.lower()}']
        ):
            # Idea mentions this ticker in sentiment context but expression doesn't use it
            if f"df['{t}_S']" not in expression and f'_{t.lower()}' not in expression.lower():
                idea_mentions_sentiment = True

    expr_has_sentiment = '_S' in expression and "df['" in expression

    if idea_mentions_sentiment and not expr_has_sentiment:
        return False, "idea mentions sentiment but expression has no _S column"

    return True, "OK"


def eval_one(alpha_def: dict, df: pd.DataFrame, fwd_ret: pd.Series,
             fwd_ret_5d: pd.Series | None = None) -> dict:
    """Eval 1 alpha, tự động flip nếu IC âm. Validate idea-expression consistency."""
    result = deepcopy(alpha_def)

    # Validate idea-expression consistency TRƯỚC KHI eval
    idea = alpha_def.get("idea", "")
    expression = alpha_def.get("expression", "")
    consistent, reason = _check_idea_expression_consistency(idea, expression)
    if not consistent:
        log.warning(f"  Alpha {alpha_def['id']} REJECTED: {reason}")
        log.warning(f"    Idea: {idea[:80]}...")
        log.warning(f"    Expr: {expression[:80]}...")
        result.update({"ic": None, "ic_5d": None, "sharpe": None, "turnover": None,
                        "status": "EVAL_ERROR", "series": None,
                        "error_reason": reason, "score": 0.0, "flipped": False})
        return result

    series = eval_alpha_expression(alpha_def["expression"], df)

    valid, reason = is_valid_series(series)
    if not valid:
        log.warning(f"  Alpha {alpha_def['id']} invalid: {reason}")
        result.update({"ic": None, "ic_5d": None, "sharpe": None, "turnover": None,
                        "status": "EVAL_ERROR", "series": None,
                        "error_reason": reason, "score": 0.0, "flipped": False})
        return result

    norm = (series - series.mean()) / (series.std() + 1e-9)
    ic = compute_ic(norm, fwd_ret)

    # Auto-flip: IC âm → đảo signal
    flipped = False
    if not np.isnan(ic) and ic < 0:
        norm = -norm
        ic = -ic
        flipped = True

    sharpe   = compute_sharpe(norm, fwd_ret)
    turnover = compute_turnover(norm)

    ic_5d = None
    if fwd_ret_5d is not None:
        ic_5d_raw = compute_ic(norm, fwd_ret_5d)
        if not np.isnan(ic_5d_raw):
            ic_5d = round(abs(ic_5d_raw), 6)

    score = composite_score(ic, sharpe)

    result.update({
        "ic":       round(ic, 6)       if not np.isnan(ic)       else None,
        "ic_5d":    ic_5d,
        "sharpe":   round(sharpe, 4)   if not np.isnan(sharpe)   else None,
        "turnover": round(turnover, 4) if not np.isnan(turnover) else None,
        "status":   "OK",
        "series":   norm,
        "score":    round(score, 6),
        "flipped":  flipped,
    })
    return result


def compute_corr_matrix(results: list[dict]) -> pd.DataFrame:
    ok = [r for r in results if r["status"] == "OK" and r.get("series") is not None]
    if not ok:
        return pd.DataFrame()
    df_s = pd.DataFrame({f"alpha_{r['id']}": r["series"] for r in ok})
    return df_s.corr(method="spearman").round(3)


def identify_weak_alphas(results, corr_matrix, fwd_ret: pd.Series | None = None):
    weak = []
    weak_ids = set()

    for r in results:
        if r["status"] == "EVAL_ERROR":
            weak.append((r["id"], "expression lỗi eval"))
            weak_ids.add(r["id"])

    for r in results:
        if r["status"] != "OK" or r["id"] in weak_ids:
            continue
        ic     = abs(r.get("ic") or 0.0)
        sharpe = r.get("sharpe") or 0.0
        if ic < IC_THRESHOLD and sharpe <= 0.3:
            weak.append((r["id"],
                f"|IC|={ic:.4f} < {IC_THRESHOLD} và Sharpe={sharpe:.3f} ≤ 0.3. "
                f"Thử: kết hợp 2-3 indicators, dùng ts_linear_reg, ts_ir, "
                f"hoặc price-volume divergence"))
            weak_ids.add(r["id"])

    for r in results:
        if r["status"] != "OK" or r["id"] in weak_ids:
            continue
        sharpe = r.get("sharpe") or 0.0
        if sharpe < -0.5:
            weak.append((r["id"],
                f"Sharpe={sharpe:.3f} quá âm. Thử đổi logic hoàn toàn"))
            weak_ids.add(r["id"])

    if not corr_matrix.empty:
        ok_results = {r["id"]: r for r in results if r["status"] == "OK"}
        ids = list(ok_results.keys())
        for i, id_i in enumerate(ids):
            for id_j in ids[i + 1:]:
                col_i, col_j = f"alpha_{id_i}", f"alpha_{id_j}"
                if col_i not in corr_matrix.columns or col_j not in corr_matrix.columns:
                    continue
                corr_val = abs(corr_matrix.loc[col_i, col_j])
                if corr_val < CORR_THRESHOLD:
                    continue
                sc_i = ok_results[id_i].get("score", 0)
                sc_j = ok_results[id_j].get("score", 0)
                loser = id_i if sc_i <= sc_j else id_j
                if loser not in weak_ids:
                    winner = id_j if loser == id_i else id_i
                    weak.append((loser,
                        f"|corr|={corr_val:.3f} với alpha_{winner} ≥ {CORR_THRESHOLD}. "
                        f"Phải dùng data source hoàn toàn khác"))
                    weak_ids.add(loser)

    # Align with dashboard decay logic: even strong-looking alpha can be decaying fast
    # versus its own historical IC and should be considered for refinement.
    if fwd_ret is not None:
        from core.backtester import detect_decay
        for r in results:
            if r["status"] != "OK" or r["id"] in weak_ids:
                continue
            s = r.get("series")
            if s is None:
                continue
            try:
                d = detect_decay(s, fwd_ret)
                if d.get("decaying"):
                    weak.append((
                        r["id"],
                        f"IC decay {d.get('drop_pct', 0):.1f}% "
                        f"({d.get('hist_ic', 0):+.4f} → {d.get('recent_ic', 0):+.4f})"
                    ))
                    weak_ids.add(r["id"])
            except Exception:
                # Decay check failure should not break refinement loop.
                pass

    return weak


def should_replace_alpha(old_r: dict, new_r: dict, fwd_ret: pd.Series | None = None) -> tuple[bool, str]:
    """
    Replacement policy for refinement.
    Priority order:
      1) Strict score improvement.
      2) If old alpha is decaying, allow replacement when decay is materially improved
         even if score is slightly lower.
    """
    if new_r.get("status") != "OK":
        return False, "new ERROR"

    old_score = old_r.get("score", 0.0) if old_r.get("status") == "OK" else 0.0
    new_score = new_r.get("score", 0.0)

    if new_score > old_score:
        return True, f"score {old_score:.4f} -> {new_score:.4f}"

    if fwd_ret is None:
        return False, f"score not improved ({new_score:.4f} <= {old_score:.4f})"

    old_s = old_r.get("series")
    new_s = new_r.get("series")
    if old_s is None or new_s is None:
        return False, f"score not improved ({new_score:.4f} <= {old_score:.4f})"

    try:
        from core.backtester import detect_decay
        old_decay = detect_decay(old_s, fwd_ret)
        new_decay = detect_decay(new_s, fwd_ret)

        old_is_dec = bool(old_decay.get("decaying"))
        new_is_dec = bool(new_decay.get("decaying"))
        old_drop = float(old_decay.get("drop_pct", 0.0) or 0.0)
        new_drop = float(new_decay.get("drop_pct", 0.0) or 0.0)

        # Case A: old decaying -> new no longer decaying.
        if old_is_dec and not new_is_dec and new_score >= old_score * 0.80:
            return True, (
                "decay fixed "
                f"({old_drop:.1f}% -> {new_drop:.1f}%) with acceptable score drift"
            )

        # Case B: both still decaying, but decay improves materially.
        if old_is_dec and new_is_dec:
            if (new_drop <= old_drop - 15.0) and (new_score >= old_score * 0.90):
                return True, (
                    "decay improved "
                    f"({old_drop:.1f}% -> {new_drop:.1f}%) with controlled score drift"
                )
    except Exception:
        pass

    return False, f"score not improved ({new_score:.4f} <= {old_score:.4f})"


# ===================================================================
# 3. PROMPT
# ===================================================================

OPERATOR_SIGNATURES = """
## Toán tử — CHÚ Ý SỐ LƯỢNG THAM SỐ (thiếu sẽ lỗi!)

### Time-series (luôn cần window/period)
shift(s, period)  ts_delta(s, period)  ts_delta_ratio(s, period)
ts_mean(s, w)  ts_std(s, w)  ts_sum(s, w)  ts_min(s, w)  ts_max(s, w)
ts_rank(s, w)  ts_median(s, w)  ts_skew(s, w)  ts_kurt(s, w)
ts_zscore_scale(s, w)   ← PHẢI CÓ 2 ARGS: series VÀ window
ts_maxmin_scale(s, w)   ← PHẢI CÓ 2 ARGS
ts_corr(s1, s2, w)  ts_cov(s1, s2, w)   ← 3 ARGS
ts_ir(s, w)  ts_linear_reg(s, w)  ts_ema(s, span)
ts_decayed_linear(s, w)  ts_percentile(s, w, pct=0.5)
ts_argmax(s, w)  ts_argmin(s, w)  ts_argmaxmin_diff(s, w)
ts_max_diff(s, w)  ts_min_diff(s, w)  ts_product(s, w)

### Group-wise (rolling theo 1 series)
grouped_mean(s, w)  grouped_std(s, w)  grouped_max(s, w)
grouped_min(s, w)  grouped_sum(s, w)  grouped_demean(s, w)
grouped_zscore_scale(s, w)  grouped_winsorize_scale(s, w)

### Element-wise
add(s1, s2)  minus(s1, s2)  cwise_mul(s1, s2)  div(s1, s2)
relu(s)  neg(s)  abso(s)  sign(s)  tanh(s)  log(s)  log1p(s)
pow_op(s, exp)  pow_sign(s, exp)  round_op(s, decimals=2)
clip(s, lower, upper)
greater(s1, s2)  less(s1, s2)  cwise_max(s1, s2)  cwise_min(s1, s2)
normed_rank_diff(s1, s2)

### Normalize (KHÔNG cần window)
zscore_scale(s)  winsorize_scale(s)  normed_rank(s)
"""

FIELD_TEMPLATE = """
## Data fields (df index = trading date)

Stock:   df['open'], df['high'], df['low'], df['close'], df['volume']
TI:      df['SMA_5'], df['SMA_20'], df['EMA_10'], df['Momentum_3'], df['Momentum_10']
         df['RSI_14'], df['MACD'], df['MACD_Signal']
         df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['OBV']
Sent:    {sentiment_list}
         (encoding: -1=tiêu cực, 0=trung lập, 1=tích cực)

### Data statistics
{data_stats}
"""

SYSTEM_PROMPT = """Bạn là senior quant researcher thiết kế formulaic alpha signals cho HOSE (Việt Nam).

## Nguyên tắc alpha CHẤT LƯỢNG CAO

### Signal design
- Alpha là DIRECTIONAL signal: dương = kỳ vọng tăng, âm = kỳ vọng giảm
- Kết hợp ≥ 2 components từ NGUỒN KHÁC NHAU (price + volume, technical + sentiment, v.v.)
- KHÔNG dùng 1 indicator đơn lẻ (ts_delta_ratio, ts_std, RSI một mình → IC ≈ 0)
- Normalize output: dùng ts_zscore_scale(s, w), ts_rank(s, w), tanh(), hoặc div()
- Alpha phải là CONTINUOUS signal, có output khác 0 ở phần lớn ngày (>60%)

### Chiến lược tạo alpha đa dạng
- Sáng tạo cách kết hợp MỚI giữa các operators và data fields
- Thử nhiều window lengths khác nhau (5, 10, 15, 20, 30) — KHÔNG chỉ dùng 20
- Kết hợp cross-indicator: ví dụ rank của indicator A − rank của indicator B
- Dùng ts_corr, ts_cov giữa 2 series khác nhau để bắt mối quan hệ động
- Kết hợp nhiều tầng: output của 1 operator làm input cho operator khác
- Mỗi alpha phải có LOGIC TÀI CHÍNH rõ ràng (giải thích được vì sao nó predict return)

### Cách dùng sentiment ĐÚNG (nếu chọn dùng)
Sentiment data thường SPARSE: encoding -1/0/1, phần lớn ngày = 0 (80-90% ngày).
→ KHÔNG nhân trực tiếp sentiment vào signal! Kết quả sẽ = 0 hầu hết ngày.

SAI (Sharpe âm, signal thưa):
  alpha = greater(..., 0) * df['TCB_S']          ← kết quả = 0 khi TCB_S = 0
  alpha = less(..., 0) * df['MBB_S']             ← same problem
  alpha = binary_A * binary_B * df['X_S']        ← 3 terms nhân → gần như luôn = 0

ĐÚNG (signal liên tục, sentiment modulate):
  # 1. Sentiment như amplifier — technical signal luôn chạy, sentiment chỉ tăng/giảm biên độ
  alpha = cwise_mul(technical_signal, add(1.0, cwise_mul(0.3, ts_ema(df['TCB_S'], 5))))
  
  # 2. Sentiment blend — weighted average giữa technical và smoothed sentiment
  alpha = add(cwise_mul(0.7, technical_signal), cwise_mul(0.3, ts_ema(df['TCB_S'], 5)))
  
  # 3. Sentiment divergence — so sánh sentiment giữa 2 mã cùng ngành
  alpha = cwise_mul(technical_signal, tanh(ts_zscore_scale(minus(df['ACB_S'], df['TCB_S']), 10)))

Nguyên tắc: technical signal là PHẦN CHÍNH (chạy mọi ngày), sentiment chỉ MODULATE (tăng/giảm
biên độ 10-30%). Luôn smooth sentiment trước khi dùng (ts_ema, ts_mean, ts_zscore_scale).

### Hypothesis-driven design
- Mỗi alpha cần 1 HYPOTHESIS rõ ràng: market inefficiency gì → mechanism → tại sao formula capture
- Idea và expression phải NHẤT QUÁN: nếu idea nói X thì expression phải implement X
- Nếu idea không cần sentiment → ĐỪNG thêm sentiment vào expression

### TRÁNH (gây EVAL_ERROR hoặc IC ≈ 0)
- ts_zscore_scale(s) thiếu window → lỗi! Phải: ts_zscore_scale(s, 20)
- ts_maxmin_scale(s) thiếu window → lỗi! Phải: ts_maxmin_scale(s, 20)
- 1 indicator đơn lẻ: ts_std, RSI, Momentum riêng lẻ → IC ≈ 0
- Raw price levels: close, SMA_20 → leaks price level bias
- Python if/else, for loop, lambda → lỗi eval
- greater() * less() * df['X_S'] → sparse signal (= 0 hầu hết ngày) → Sharpe âm

### Syntax bắt buộc
- Chỉ dùng hàm từ danh sách operator (xem signatures)
- Gán kết quả: alpha = <expression>
- add/minus/cwise_mul/div nhận float hoặc Series
"""


def _check_sentiment_quality(sent_cols, df=None):
    """
    Phân tích chất lượng từng cột sentiment.
    Trả về dict: col → {"ok": bool, "nonzero_ratio": float, "diversity": float,
                         "mean_abs": float, "reason": str}
    """
    if df is None:
        return {c: {"ok": True, "nonzero_ratio": 1.0, "diversity": 1.0,
                    "mean_abs": 0.5, "reason": "no df"} for c in sent_cols}
    result = {}
    for c in sent_cols:
        if c not in df.columns:
            result[c] = {"ok": False, "nonzero_ratio": 0.0, "diversity": 0.0,
                         "mean_abs": 0.0, "reason": "cột không tồn tại"}
            continue
        s = df[c].dropna()
        if len(s) == 0:
            result[c] = {"ok": False, "nonzero_ratio": 0.0, "diversity": 0.0,
                         "mean_abs": 0.0, "reason": "rỗng"}
            continue

        nonzero_ratio = float((s != 0).mean())
        # diversity: tỉ lệ ngày có giá trị không phải 0, tính trên tổng ngày
        diversity = float(s.nunique()) / max(float(len(s)), 1.0)
        mean_abs  = float(s.abs().mean())

        ok = nonzero_ratio >= 0.08 and diversity >= 0.02
        if not ok:
            if nonzero_ratio < 0.08:
                reason = f"quá nhiều ngày 0 ({100*(1-nonzero_ratio):.0f}% = 0)"
            else:
                reason = f"thiếu đa dạng (diversity={diversity:.3f})"
        else:
            reason = (f"OK — nonzero={nonzero_ratio:.0%}, "
                      f"diversity={diversity:.3f}, |mean|={mean_abs:.3f}")

        result[c] = {"ok": ok, "nonzero_ratio": nonzero_ratio,
                     "diversity": diversity, "mean_abs": mean_abs, "reason": reason}
    return result


def _summarise_sentiment_for_prompt(ticker: str, sent_cols: list, df,
                                    n_required_alphas: int = 5) -> str:
    """
    Sentiment section cho prompt:
    - Báo cáo chất lượng data
    - KHÔNG ép buộc dùng sentiment
    - Chỉ encourage khi có hypothesis hợp lý
    """
    quality = _check_sentiment_quality(sent_cols, df)
    target_col = f"{ticker}_S"
    target_industry = TICKER_INDUSTRY.get(ticker, "Khác")

    good = {c: q for c, q in quality.items() if q["ok"]}
    bad  = {c: q for c, q in quality.items() if not q["ok"]}

    # Phân loại cùng ngành / khác ngành
    same_industry = []
    for c in good:
        peer = c.replace('_S', '')
        if TICKER_INDUSTRY.get(peer) == target_industry:
            same_industry.append(c)

    lines = ["### Sentiment Data Available"]

    if good:
        lines.append(f"Cột sentiment có data tốt ({len(good)}/{len(sent_cols)}):")
        for c, q in sorted(good.items(), key=lambda x: -x[1]["nonzero_ratio"])[:8]:
            peer = c.replace('_S', '')
            industry = TICKER_INDUSTRY.get(peer, '?')
            same = " ← CÙNG NGÀNH" if c in same_industry else ""
            lines.append(f"   df['{c}'] ({industry}): nonzero={q['nonzero_ratio']:.0%}{same}")
    if bad:
        lines.append(f"Cột KHÔNG dùng ({len(bad)} — data quá thưa):")
        for c in list(bad.keys())[:3]:
            lines.append(f"   df['{c}']: SKIP")

    if good:
        good_list = ", ".join(f"df['{c}']" for c in list(good.keys())[:6])
        lines.append(f"""
### Hướng dẫn sử dụng Sentiment
Sentiment data CÓ SẴN: {good_list}
{ticker} thuộc ngành: {target_industry}. Các mã cùng ngành: {', '.join(same_industry) if same_industry else 'không có'}

NGUYÊN TẮC (QUAN TRỌNG — đọc kỹ):
1. Sentiment là OPTIONAL — chỉ dùng khi bạn có HYPOTHESIS rõ ràng tại sao nó cải thiện alpha.
2. Nếu dùng, phải giải thích TRONG PHẦN IDEA tại sao sentiment liên quan đến logic tài chính
   của alpha đó. Ví dụ: "Khi sentiment ngành ngân hàng đồng loạt tiêu cực, momentum giảm
   được khuếch đại → short signal mạnh hơn" — đây là hypothesis có logic.
3. KHÔNG chấp nhận: "Dùng sentiment để tăng cường tín hiệu" — quá chung chung, không có logic.
4. Nếu idea mention sentiment thì expression PHẢI chứa df['..._S']. Nếu không → alpha bị reject.
5. Alpha thuần technical (không dùng sentiment) hoàn toàn OK nếu có hypothesis tốt.
6. KHÔNG có quota bắt buộc — 0/5 alpha dùng sentiment cũng chấp nhận nếu chất lượng tốt.
""")
    else:
        lines.append("""
### Sentiment
Tất cả cột sentiment đều thưa data. KHÔNG dùng sentiment — tập trung technical + price-volume.
""")

    return "\n".join(lines)


def _build_seed_examples(ticker, sent_cols, same_industry_peers=None):
    return f"""
## Syntax reference — structural patterns (TỰ SÁNG TẠO nội dung + fields)

Pattern 1: div(operator_A(field_X, window), add(operator_B(field_Y, window), epsilon))
  → Chuẩn hoá output của operator A bằng operator B, epsilon tránh /0

Pattern 2: minus(ts_rank(field_X, window), ts_rank(field_Y, window))
  → So sánh percentile rank của 2 nguồn data khác nhau

Pattern 3: cwise_mul(operator_A(field_X, w1), operator_B(field_Y, w2))
  → Nhân 2 signals từ nguồn khác nhau (interaction term)

QUAN TRỌNG:
- Các pattern trên CHỈ là cấu trúc. Bạn PHẢI tự chọn operators, fields, windows KHÁC NHAU.
- Kết hợp sáng tạo: lồng operators, dùng operators ít phổ biến (ts_corr, ts_skew, ts_kurt,
  ts_argmaxmin_diff, ts_percentile, ts_cov, pow_sign, sign, greater, less...)
- KHÔNG lặp lại cùng 1 tổ hợp operator+field ở nhiều alpha.
- Sentiment columns (df['..._S']) là 1 data source như bất kỳ field nào khác — dùng khi có
  hypothesis hợp lý, KHÔNG cần pattern riêng.
"""


def build_seed_prompt(ticker, sent_cols, df=None):
    sent_list = ", ".join(f"df['{c}']" for c in sent_cols)
    data_stats = compute_data_stats(df) if df is not None else "(N/A)"
    fields = FIELD_TEMPLATE.format(sentiment_list=sent_list, data_stats=data_stats)

    # Lọc chất lượng Sentiment
    sent_quality = _check_sentiment_quality(sent_cols, df)
    good_sent = [c for c, ok in sent_quality.items() if ok]
    bad_sent  = [c for c, ok in sent_quality.items() if not ok]

    # Phân loại CÙNG NGÀNH vs KHÁC NGÀNH
    target_industry = TICKER_INDUSTRY.get(ticker, "Khác")
    same_industry_sent = []
    other_industry_sent = []
    
    for c in good_sent:
        peer_ticker = c.replace('_S', '')
        if TICKER_INDUSTRY.get(peer_ticker) == target_industry:
            same_industry_sent.append(c)
        else:
            other_industry_sent.append(c)

    # Sentiment info (đã xử lý trong _summarise_sentiment_for_prompt)
    sent_block = _summarise_sentiment_for_prompt(ticker, sent_cols, df, n_required_alphas=SEED_OVERSAMPLE)

    examples = _build_seed_examples(ticker, sent_cols, same_industry_sent)

    return f"""
{fields}
{OPERATOR_SIGNATURES}
{examples}
{sent_block}

## Yêu cầu
Sinh ĐÚNG {SEED_OVERSAMPLE} alpha formulas cho **{ticker}**.

### Cách viết mỗi alpha (hypothesis-driven):
Mỗi alpha cần có:
- **idea**: MỘT hypothesis cụ thể — giải thích MARKET INEFFICIENCY gì bạn đang khai thác
  và TẠI SAO formula của bạn capture được nó. Nếu dùng sentiment, giải thích chuỗi
  logic tài chính: sentiment → hành vi thị trường → price impact.
  Ví dụ TỐT: "Khi volume tăng đột biến nhưng price không tăng tương ứng (divergence),
  đây là dấu hiệu distribution → kỳ vọng giảm. RSI ở vùng overbought xác nhận."
  Ví dụ XẤU: "Kết hợp MACD với volume, dùng sentiment để tăng cường tín hiệu."
- **expression**: formula PHẢI NHẤT QUÁN với idea. Nếu idea mention sentiment
  thì expression PHẢI chứa df['..._S']. Nếu idea không cần sentiment thì ĐỪNG thêm vào.

### Đa dạng hoá — mỗi alpha khai thác MỘT khía cạnh riêng:
1. Trend/Momentum (slope, acceleration, momentum đa timeframe)
2. Mean-reversion (vị trí giá so với baseline, oversold/overbought phức hợp)
3. Volume dynamics (volume vs price divergence, volume regime change)
4. Volatility/Risk (risk-adjusted return, volatility regime, skewness)
5. MACD/Signal interaction (histogram dynamics, signal crossover strength)
6. Correlation-based (rolling corr giữa price và volume, hoặc giữa indicators)
7. Sentiment-driven (CHỈ KHI có hypothesis hợp lý — xem hướng dẫn sentiment ở trên)

BẮT BUỘC:
- MỖI alpha kết hợp ≥ 2 components từ NGUỒN KHÁC NHAU.
- SÁNG TẠO công thức MỚI — KHÔNG copy từ ví dụ syntax reference.
- Thử nhiều window lengths khác nhau (5, 10, 15, 20, 30).
- Không sử dụng các phép toán thừa thãi (minus(A, 0), cwise_mul(A, 1)).
- IDEA và EXPRESSION phải nhất quán: idea nói gì → expression phải implement đúng cái đó.

Output JSON:
{{
  "alphas": [
    {{"id": 1, "idea": "<hypothesis + logic>", "expression": "alpha = <expr>"}},
    ...
    {{"id": {SEED_OVERSAMPLE}, "idea": "...", "expression": "alpha = ..."}}
  ]
}}
"""


def build_refine_prompt(ticker, sent_cols, current_results, weak, corr_matrix, df=None):
    lines = []
    for r in current_results:
        if r["status"] == "OK":
            flip = " [flipped]" if r.get("flipped") else ""
            lines.append(
                f"  Alpha {r['id']}: IC={r['ic']:+.4f}, Sharpe={r['sharpe']:+.3f}, "
                f"Score={r.get('score', 0):.4f}{flip}\n"
                f"    Idea: {r['idea']}\n    Expr: {r['expression']}")
        else:
            lines.append(f"  Alpha {r['id']}: [ERROR] {r.get('error_reason','?')}\n"
                         f"    Expr: {r['expression']}")

    corr_text = corr_matrix.to_string() if not corr_matrix.empty else "N/A"
    weak_text = "\n".join(f"  - Alpha {aid}: {reason}" for aid, reason in weak)
    weak_ids  = [w[0] for w in weak]
    keep_ids  = [r["id"] for r in current_results if r["id"] not in weak_ids]
    keep_text = ", ".join(f"alpha_{i}" for i in keep_ids) if keep_ids else "không có"

    keep_ideas = [f"    alpha_{r['id']}: {r['idea']}"
                  for r in current_results if r["id"] in keep_ids and r["status"] == "OK"]

    sent_list  = ", ".join(f"df['{c}']" for c in sent_cols)
    data_stats = compute_data_stats(df) if df is not None else "(N/A)"
    fields     = FIELD_TEMPLATE.format(sentiment_list=sent_list, data_stats=data_stats)
    sent_block = _summarise_sentiment_for_prompt(ticker, sent_cols, df,
                                                 n_required_alphas=len(weak))

    return f"""
## Refinement cho {ticker}

### Kết quả hiện tại
{chr(10).join(lines)}

### Correlation matrix
{corr_text}

### Alpha cần thay (lý do + gợi ý)
{weak_text}

### Alpha GIỮ NGUYÊN
{keep_text}
{chr(10).join(keep_ideas)}

---
{fields}
{OPERATOR_SIGNATURES}
{sent_block}

## Yêu cầu
Sinh ĐÚNG {len(weak)} alpha MỚI.
- Kết hợp ≥ 2 components, ts_zscore_scale(s, w) CẦN 2 ARGS
- Ý tưởng KHÁC HOÀN TOÀN với alpha đang giữ
- ID cần thay: {weak_ids}
- Mỗi alpha cần HYPOTHESIS rõ ràng trong idea: market inefficiency gì, tại sao formula capture được
- Nếu idea mention sentiment → expression PHẢI chứa df['..._S']. Không thì alpha bị reject.

Output JSON:
{{
  "alphas": [{{"id": <id>, "idea": "...", "expression": "alpha = ..."}}]
}}
"""


# ===================================================================
# 4. LLM
# ===================================================================

def call_llm(prompt, expected_count, temperature=None):
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temp,
                max_tokens=LLM_MAX_TOKENS,
            )
            data   = json.loads(response.choices[0].message.content)
            alphas = data.get("alphas", [])
            if len(alphas) >= expected_count:
                return alphas[:expected_count]
            log.warning(f"  LLM trả {len(alphas)}/{expected_count}. Thử lại...")
        except Exception as e:
            log.warning(f"  LLM lỗi lần {attempt+1}: {e}")
    raise RuntimeError(f"Không lấy đủ {expected_count} alphas sau 3 lần thử")


# ===================================================================
# 5. SELECTION
# ===================================================================

def select_best_alphas(results, n=5):
    """Greedy selection: score cao nhất + low correlation."""
    ok = [r for r in results if r["status"] == "OK"]
    ok.sort(key=lambda r: r.get("score", 0), reverse=True)

    selected = []
    for cand in ok:
        if len(selected) >= n:
            break
        c_s = cand.get("series")
        if c_s is None:
            continue
        corr_ok = True
        for exist in selected:
            e_s = exist.get("series")
            if e_s is None:
                continue
            merged = pd.concat([c_s, e_s], axis=1).dropna()
            if len(merged) < 10:
                continue
            if abs(merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman")) >= CORR_THRESHOLD:
                corr_ok = False
                break
        if corr_ok:
            selected.append(cand)

    for i, r in enumerate(selected):
        r["id"] = i + 1

    if len(selected) < n:
        remaining = [r for r in ok if r not in selected]
        for r in remaining:
            if len(selected) >= n:
                break
            r["id"] = len(selected) + 1
            selected.append(r)

    return selected[:n]


# ===================================================================
# 6. LLM RESCUE — retry EVAL_ERROR bằng prompt đặc biệt
# ===================================================================

RESCUE_PROMPT_TEMPLATE = """
## Rescue: sửa alpha bị lỗi eval cho mã {ticker}

Các alpha sau bị EVAL_ERROR khi chạy. Hãy viết lại HOÀN TOÀN với logic MỚI.

### Alpha cần rescue
{error_details}

### Lỗi thường gặp cần tránh
- ts_zscore_scale(s) thiếu window → Phải: ts_zscore_scale(s, 20)
- ts_maxmin_scale(s) thiếu window → Phải: ts_maxmin_scale(s, 20)
- ts_corr(s1, s2) thiếu window → Phải: ts_corr(s1, s2, 20)
- Chia cho series có thể = 0 → Dùng div(a, b) thay vì a/b
- Series toàn NaN hoặc constant → Dùng operators có output variance (ts_rank, ts_zscore_scale)

### Data fields
{fields}

### Operators
{operators}

## Yêu cầu
Viết lại {n_errors} alpha với logic HOÀN TOÀN MỚI, ĐƠN GIẢN HƠN nhưng vẫn kết hợp ≥ 2 components.
Ưu tiên operators an toàn: ts_rank, ts_zscore_scale, minus, div, cwise_mul, tanh.

Output JSON:
{{
  "alphas": [{{"id": <id>, "idea": "...", "expression": "alpha = ..."}}]
}}
"""


def build_rescue_prompt(ticker: str, error_results: list[dict],
                        sent_cols: list[str], df=None) -> str:
    error_details = []
    for r in error_results:
        error_details.append(
            f"  Alpha {r['id']}: lỗi={r.get('error_reason', '?')}\n"
            f"    Expression cũ: {r['expression']}"
        )

    sent_list  = ", ".join(f"df['{c}']" for c in sent_cols[:5])
    data_stats = compute_data_stats(df) if df is not None else "(N/A)"
    fields     = FIELD_TEMPLATE.format(sentiment_list=sent_list, data_stats=data_stats)
    sent_block = _summarise_sentiment_for_prompt(ticker, sent_cols, df,
                                                 n_required_alphas=len(error_results))

    return RESCUE_PROMPT_TEMPLATE.format(
        ticker=ticker,
        error_details="\n".join(error_details),
        n_errors=len(error_results),
        fields=fields,
        operators=OPERATOR_SIGNATURES + "\n" + sent_block,
    )


def rescue_errors_with_llm(results: list[dict], ticker: str, sent_cols: list[str],
                           df: pd.DataFrame, fwd_ret, fwd_ret5,
                           max_attempts: int = 3) -> list[dict]:
    """Retry EVAL_ERROR alphas bằng LLM với prompt chuyên biệt."""
    for attempt in range(1, max_attempts + 1):
        error_results = [r for r in results if r["status"] == "EVAL_ERROR"]
        if not error_results:
            break

        log.info(f"[{ticker}] Rescue attempt {attempt}/{max_attempts} "
                 f"cho {len(error_results)} alpha lỗi: {[r['id'] for r in error_results]}")

        try:
            prompt = build_rescue_prompt(ticker, error_results, sent_cols, df)
            new_defs = call_llm(prompt, expected_count=len(error_results),
                                temperature=0.5)  # lower temp cho rescue = an toàn hơn

            for new_def in new_defs:
                new_r = eval_one(new_def, df, fwd_ret, fwd_ret5)
                if new_r["status"] == "OK":
                    idx = next((i for i, r in enumerate(results) if r["id"] == new_r["id"]), None)
                    if idx is not None:
                        results[idx] = new_r
                        log.info(f"  [RESCUED] Alpha {new_r['id']} → IC={new_r['ic']:+.4f} "
                                 f"Score={new_r['score']:.4f}")
                else:
                    log.info(f"  [STILL_ERROR] Alpha {new_r['id']}: {new_r.get('error_reason')}")

        except Exception as e:
            log.warning(f"  Rescue attempt {attempt} failed: {e}")

    return results


# ===================================================================
# 7. HELPERS
# ===================================================================

def _strip_series(results):
    return [{k: v for k, v in r.items() if k != "series"} for r in results]


def log_round(ticker, round_num, results, corr):
    log.info(f"\n{'='*60}")
    log.info(f"[{ticker}] VÒNG {round_num}")
    log.info(f"{'='*60}")
    for r in results:
        if r["status"] == "OK":
            flip = " [flipped]" if r.get("flipped") else ""
            log.info(f"  Alpha {r['id']} [OK]  IC={r['ic']:+.4f}  Sharpe={r['sharpe']:+.3f}  "
                     f"Score={r.get('score', 0):.4f}  Turn={r['turnover']:.3f}{flip}")
            log.info(f"    {r['idea']}")
        else:
            log.info(f"  Alpha {r['id']} [ERROR]  {r.get('error_reason', '?')}")
    if not corr.empty:
        log.info(f"\n  Corr:\n{corr.to_string()}")


# ===================================================================
# 8. PIPELINE
# ===================================================================

def run_single(ticker, max_rounds=MAX_ROUNDS, force=False):
    out_json = os.path.join(ALPHA_FORMULA, f"{ticker}_alphas.json")
    out_csv  = os.path.join(ALPHA_DIR, f"{ticker}_alpha_values.csv")

    if os.path.exists(out_json) and not force:
        log.info(f"[{ticker}] Skip (đã có). Dùng --force để chạy lại.")
        return json.load(open(out_json))

    return _run_single_impl(ticker, max_rounds=max_rounds, refine_only=False)


def run_single_refine_only(ticker, max_rounds=MAX_ROUNDS):
    out_json = os.path.join(ALPHA_FORMULA, f"{ticker}_alphas.json")
    if not os.path.exists(out_json):
        log.warning(f"[{ticker}] Không có alpha cũ để refinement-only, fallback sang generate mới.")
        return _run_single_impl(ticker, max_rounds=max_rounds, refine_only=False)
    return _run_single_impl(ticker, max_rounds=max_rounds, refine_only=True)


def _run_single_impl(ticker, max_rounds=MAX_ROUNDS, refine_only=False):
    out_json = os.path.join(ALPHA_FORMULA, f"{ticker}_alphas.json")
    out_csv  = os.path.join(ALPHA_DIR, f"{ticker}_alpha_values.csv")

    df       = load_data(ticker)
    fwd_ret  = make_forward_return(df, horizon=1)
    fwd_ret5 = make_forward_return(df, horizon=5)
    sent_cols = [c for c in df.columns if c.endswith("_S")]
    log.info(f"[{ticker}] Sentiment cols ({len(sent_cols)})")

    if refine_only:
        with open(out_json, "r", encoding="utf-8") as f:
            prev = json.load(f)
        prev_alphas = prev.get("alphas", [])
        if not prev_alphas:
            log.warning(f"[{ticker}] File alpha cũ rỗng, fallback sang generate mới.")
            refine_only = False

    if refine_only:
        log.info(f"\n[{ticker}] === VÒNG 1: Load alpha cũ để refinement ===")
        existing_defs = []
        for i, a in enumerate(prev_alphas, 1):
            existing_defs.append({
                "id": a.get("id", i),
                "idea": a.get("idea", ""),
                "expression": a.get("expression", ""),
            })
        results = [eval_one(a, df, fwd_ret, fwd_ret5) for a in existing_defs]
    else:
        # --- Vòng 1: Seed with oversampling ---
        log.info(f"\n[{ticker}] === VÒNG 1: Seed ({SEED_OVERSAMPLE} candidates) ===")
        seed_defs = call_llm(build_seed_prompt(ticker, sent_cols, df=df),
                             expected_count=SEED_OVERSAMPLE)

        all_results = [eval_one(a, df, fwd_ret, fwd_ret5) for a in seed_defs]
        for r in all_results:
            if r["status"] == "OK":
                log.info(f"  #{r['id']}: IC={r['ic']:+.4f} Sh={r['sharpe']:+.3f} "
                         f"Sc={r['score']:.4f}{'[F]' if r.get('flipped') else ''}")
            else:
                log.info(f"  #{r['id']}: [ERR] {r.get('error_reason','?')}")

        results = select_best_alphas(all_results, n=5)

    # Nếu thiếu alpha → gọi LLM bổ sung với error-aware prompt
    if len(results) < 5:
        n_need = 5 - len(results)
        # Collect error reasons to inform LLM
        if refine_only:
            error_reasons = [r.get("error_reason", "?") for r in results if r["status"] != "OK"]
        else:
            error_reasons = [r.get("error_reason", "?") for r in all_results if r["status"] != "OK"]
        unique_errors = list(set(error_reasons))[:3]

        log.warning(f"  Chỉ {len(results)}/5 OK, gọi LLM bổ sung {n_need} alpha")
        log.warning(f"  Error reasons: {unique_errors}")

        for supplement_attempt in range(5):  # tăng từ 3 → 5 attempts
            if len(results) >= 5:
                break
            try:
                existing_ideas = [r["idea"] for r in results if r["status"] == "OK"]
                n_still_need = 5 - len(results)

                # Error-aware supplement prompt
                supp_prompt = build_seed_prompt(ticker, sent_cols, df=df)
                supp_prompt += f"""

ĐÃ CÓ các alpha sau (KHÔNG trùng ý tưởng):
"""
                for idea in existing_ideas:
                    supp_prompt += f"  - {idea}\n"

                supp_prompt += f"""
CÁC ALPHA TRƯỚC ĐÓ BỊ REJECT VÌ:
"""
                for err in unique_errors:
                    supp_prompt += f"  - {err}\n"

                supp_prompt += f"""
QUAN TRỌNG: Sinh thêm ĐÚNG {n_still_need} alpha MỚI với ID từ {len(results)+1}.
- Alpha phải có OUTPUT LIÊN TỤC (khác 0 ở >60% ngày). TRÁNH binary × sentiment.
- Ưu tiên alpha THUẦN TECHNICAL nếu sentiment gây sparsity.
- Technical signal phải là PHẦN CHÍNH, sentiment chỉ modulate nhẹ (nếu dùng).
"""
                supp_defs = call_llm(supp_prompt, expected_count=n_still_need, temperature=0.9)
                for sd in supp_defs:
                    sd["id"] = len(results) + 1
                    sr = eval_one(sd, df, fwd_ret, fwd_ret5)
                    if sr["status"] == "OK":
                        results.append(sr)
                        log.info(f"  [SUPPLEMENT] Alpha {sr['id']} OK: IC={sr['ic']:+.4f}")
                    else:
                        log.info(f"  [SUPPLEMENT FAIL] Alpha {sr['id']}: {sr.get('error_reason','?')}")
                    if len(results) >= 5:
                        break
            except Exception as e:
                log.warning(f"  Supplement attempt {supplement_attempt+1} failed: {e}")

    # Final fallback: nếu vẫn < 5, thêm simple technical alphas
    if len(results) < 5:
        log.warning(f"  Vẫn chỉ {len(results)}/5 sau supplement. Thêm fallback technical alphas.")
        fallback_exprs = [
            ("Momentum multi-timeframe: so sánh momentum ngắn hạn vs dài hạn",
             "alpha = minus(ts_zscore_scale(df['Momentum_3'], 10), ts_zscore_scale(df['Momentum_10'], 20))"),
            ("Volume-price divergence: volume tăng relative nhưng return giảm",
             "alpha = minus(ts_rank(df['volume'], 20), ts_rank(ts_delta(df['close'], 5), 20))"),
            ("RSI mean-reversion: khoảng cách RSI so với neutral 50, đa timeframe",
             "alpha = tanh(ts_zscore_scale(minus(df['RSI_14'], 50.0), 15))"),
            ("BB position: vị trí giá trong kênh Bollinger",
             "alpha = ts_zscore_scale(div(minus(df['close'], df['BB_Lower']), add(minus(df['BB_Upper'], df['BB_Lower']), 0.001)), 20)"),
            ("MACD histogram momentum: tốc độ thay đổi histogram",
             "alpha = ts_zscore_scale(ts_delta(minus(df['MACD'], df['MACD_Signal']), 3), 10)"),
        ]
        for idea, expr in fallback_exprs:
            if len(results) >= 5:
                break
            fb_def = {"id": len(results) + 1, "idea": idea, "expression": expr}
            fb_r = eval_one(fb_def, df, fwd_ret, fwd_ret5)
            if fb_r["status"] == "OK":
                results.append(fb_r)
                log.info(f"  [FALLBACK] Alpha {fb_r['id']} OK: IC={fb_r['ic']:+.4f}")
            else:
                log.info(f"  [FALLBACK FAIL] Alpha {fb_r['id']}: {fb_r.get('error_reason','?')}")

    corr = compute_corr_matrix(results)
    log_round(ticker, 1, results, corr)
    history = [{"round": 1, "alphas": _strip_series(results)}]

    # --- Refinement ---
    for round_num in range(2, max_rounds + 1):
        weak = identify_weak_alphas(results, corr, fwd_ret=fwd_ret)
        if not weak:
            log.info(f"[{ticker}] Tất cả OK — dừng vòng {round_num - 1}.")
            break

        log.info(f"\n[{ticker}] === VÒNG {round_num}: Refinement ===")
        for aid, reason in weak:
            log.info(f"  Thay alpha {aid}: {reason}")

        replaced = 0
        keep_ids = {r["id"] for r in results if r["id"] not in [w[0] for w in weak]}

        pending = list(weak)
        for attempt in range(1, REFINE_ATTEMPTS_PER_ROUND + 1):
            if not pending:
                break

            if attempt > 1:
                log.info(
                    f"  Retry refinement attempt {attempt}/{REFINE_ATTEMPTS_PER_ROUND} "
                    f"for pending IDs {[w[0] for w in pending]}"
                )

            try:
                new_defs = call_llm(
                    build_refine_prompt(ticker, sent_cols, results, pending, corr, df=df),
                    expected_count=len(pending),
                    temperature=min(1.0, LLM_TEMPERATURE + 0.1 * (attempt - 1)),
                )
            except RuntimeError as e:
                log.warning(f"  LLM failed attempt {attempt}: {e}")
                continue

            new_results = [eval_one(a, df, fwd_ret, fwd_ret5) for a in new_defs]
            unresolved_ids = {w[0] for w in pending}

            for new_r in new_results:
                tid = new_r["id"]
                old_r = next((r for r in results if r["id"] == tid), None)
                if old_r is None:
                    continue

                can_replace, replace_reason = should_replace_alpha(old_r, new_r, fwd_ret=fwd_ret)
                if not can_replace:
                    log.info(f"  [KEEP] {tid}: {replace_reason}")
                    continue

                # Corr check
                new_s = new_r.get("series")
                corr_ok = True
                if new_s is not None:
                    for kid in keep_ids:
                        kept = next((r for r in results if r["id"] == kid), None)
                        if kept is None or kept.get("series") is None:
                            continue
                        m = pd.concat([new_s, kept["series"]], axis=1).dropna()
                        if len(m) >= 10 and abs(m.iloc[:, 0].corr(m.iloc[:, 1], method="spearman")) >= CORR_THRESHOLD:
                            log.info(f"  [REJECT] {tid}: high corr with alpha_{kid}")
                            corr_ok = False
                            break

                if corr_ok:
                    idx = next(i for i, r in enumerate(results) if r["id"] == tid)
                    results[idx] = new_r
                    keep_ids.add(tid)
                    replaced += 1
                    unresolved_ids.discard(tid)
                    old_score = old_r.get("score", 0) if old_r.get("status") == "OK" else 0.0
                    new_score = new_r.get("score", 0)
                    log.info(f"  [REPLACE] {tid}: {old_score:.4f} → {new_score:.4f} | {replace_reason}")

            pending = [w for w in pending if w[0] in unresolved_ids]
            if pending:
                log.info(f"  Pending after attempt {attempt}: {[w[0] for w in pending]}")

        corr = compute_corr_matrix(results)
        log_round(ticker, round_num, results, corr)
        history.append({"round": round_num, "replaced": replaced, "alphas": _strip_series(results)})

        if replaced == 0:
            log.info(f"[{ticker}] Chưa cải thiện ở vòng {round_num} — tiếp tục thử vòng sau.")

    # --- Rescue EVAL_ERROR bằng LLM retry ---
    results = rescue_errors_with_llm(results, ticker, sent_cols, df, fwd_ret, fwd_ret5)

    # --- Save ---
    series_dict = {}
    for r in results:
        col = f"alpha_{r['id']}"
        if r["status"] == "OK" and r.get("series") is not None:
            series_dict[col] = r["series"]
        else:
            series_dict[col] = pd.Series(0.0, index=df.index)

    df_vals = pd.DataFrame(series_dict, index=df.index)
    final = _strip_series(results)
    output = {"ticker": ticker, "n_rows": len(df), "n_rounds": len(history),
              "alphas": final, "history": history}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    if not df_vals.empty:
        df_vals.to_csv(out_csv)

    ok = [r for r in final if r["status"] == "OK"]
    log.info(f"\n{'='*60}")
    log.info(f"[{ticker}] FINAL: {len(ok)}/5 OK sau {len(history)} vòng")
    log.info(f"{'='*60}")

    # --- Sentiment usage audit ---
    sent_keywords = [c.replace("_S", "").lower() for c in sent_cols]
    n_using_sent = sum(
        1 for r in ok
        if any(f"df['{c}']" in r.get("expression", "") for c in sent_cols)
    )
    quality = _check_sentiment_quality(sent_cols, df)
    n_good_sent = sum(1 for q in quality.values() if q["ok"])
    log.info(f"[{ticker}] Sentiment: {n_good_sent}/{len(sent_cols)} cột quality OK | "
             f"{n_using_sent}/{len(ok)} alpha dùng sentiment")

    for r in ok:
        uses_sent = any(f"df['{c}']" in r.get("expression", "") for c in sent_cols)
        sent_tag = " 📊sent" if uses_sent else ""
        log.info(f"  Alpha {r['id']}  IC={r['ic']:+.4f}  Sharpe={r['sharpe']:+.3f}  "
                 f"Score={r.get('score',0):.4f}{'  [flipped]' if r.get('flipped') else ''}{sent_tag}")
        log.info(f"    {r['idea']}")

    return output


# ===================================================================
# 9. RUN ALL
# ===================================================================

def run_all(max_rounds=MAX_ROUNDS, force=False):
    summary = []
    for ticker in VN30_SYMBOLS:
        try:
            result = run_single(ticker, max_rounds=max_rounds, force=force)
            ok = [a for a in result["alphas"] if a["status"] == "OK"]
            avg_ic = np.mean([abs(a["ic"]) for a in ok if a.get("ic")]) if ok else 0
            summary.append({"ticker": ticker, "ok": len(ok), "avg_ic": round(avg_ic, 4)})
        except Exception as e:
            log.error(f"[{ticker}] FAILED: {e}")
            summary.append({"ticker": ticker, "ok": 0, "avg_ic": 0})

    log.info("\n" + "="*60 + "\nSUMMARY — VN30\n" + "="*60)
    for s in summary:
        log.info(f"  {s['ticker']}: {s['ok']}/5, avg|IC|={s['avg_ic']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gen Alpha")
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refine-only", action="store_true")
    args = parser.parse_args()

    if args.ticker:
        if args.refine_only:
            run_single_refine_only(args.ticker.upper(), max_rounds=args.rounds)
        else:
            run_single(args.ticker.upper(), max_rounds=args.rounds, force=args.force)
    elif args.all:
        run_all(max_rounds=args.rounds, force=args.force)
    else:
        parser.print_help()