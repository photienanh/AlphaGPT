# prompts/alpha_prompts.py

OPERATOR_SIGNATURES = """
## Operators có sẵn (alpha_operators.py) — dùng đúng tên hàm, đúng số args

### Time-series
shift(s, period)              delay(s, period)
ts_delta(s, period)           delta(s, period)          ts_delta_ratio(s, period)
ts_mean(s, w)                 ts_std(s, w)              stddev(s, w)
ts_sum(s, w)                  sum_op(s, w)
ts_min(s, w)                  ts_max(s, w)
ts_rank(s, w)                 ts_median(s, w)
ts_zscore_scale(s, w)         ts_maxmin_scale(s, w)
ts_skew(s, w)                 ts_kurt(s, w)
ts_corr(s1, s2, w)            correlation(s1, s2, w)
ts_cov(s1, s2, w)             covariance(s1, s2, w)
ts_ir(s, w)                   ts_linear_reg(s, w)
ts_ema(s, span)               ts_decayed_linear(s, w)   decay_linear(s, w)
ts_argmax(s, w)               ts_argmin(s, w)
ts_argmaxmin_diff(s, w)       ts_max_diff(s, w)         ts_min_diff(s, w)
ts_product(s, w)              product(s, w)

### Cross-sectional
rank(s)       # TIME-SERIES percentile (expanding) — KHÔNG phải cross-sectional
rank_ts(s)    # alias rõ ràng hơn cho time-series rank
rank_ts(s, w) # rolling time-series rank trong window w ngày
cs_rank(s)    # cross-sectional rank tại thời điểm hiện tại (single-point)
scale(s, a=1.0)   signed_power(s, exp)
zscore_scale(s)   winsorize_scale(s)
normed_rank(s)    cwise_max(s1, s2)    cwise_min(s1, s2)
indneutralize(s, df['industry'])   # demean within industry group

### Group-wise
grouped_mean(s, w)      grouped_std(s, w)
grouped_demean(s, w)    grouped_zscore_scale(s, w)

### Element-wise
add(s1, s2)    minus(s1, s2)    cwise_mul(s1, s2)    div(s1, s2)
relu(s)        neg(s)           abso(s)               sign(s)
log(s)         log1p(s)         tanh(s)               clip(s, lower, upper)
pow_op(s, exp) pow_sign(s, exp) signed_power(s, exp)
greater(s1, s2)   less(s1, s2)
normed_rank_diff(s1, s2)
"""

DATA_FIELDS_BLOCK = """
## Data fields có sẵn trong df (pd.DataFrame, index = datetime)

CẢNH BÁO: Chỉ được dùng ĐÚNG các tên field sau (lowercase).

### OHLCV
df['open']    df['high']    df['low']    df['close']    df['volume']

### Volume-derived
df['vwap']      — volume-weighted average price: (high+low+close)/3
df['adv20']     — average daily dollar volume 20 ngày: (close×volume).rolling(20).mean()
df['obv']       — On-Balance Volume

### Returns
df['returns']   — close.pct_change(1)

### Moving averages
df['sma_5']    df['sma_20']    df['ema_10']

### Momentum
df['momentum_3']    df['momentum_10']

### Oscillators
df['rsi_14']    df['macd']    df['macd_signal']

### Bollinger Bands
df['bb_upper']    df['bb_middle']    df['bb_lower']

### Industry (dùng với indneutralize)
df['industry']   — nếu có trong data

Chỉ dùng các tên trên. KHÔNG tự đặt tên khác.
"""

ALPHA_SYSTEM_PROMPT = """Bạn là Quant Developer — chuyên gia implement formulaic alpha signals.

Nhiệm vụ: Nhận trading hypothesis và implement thành Python expressions dùng bộ operators đã cho.

Nguyên tắc bắt buộc:
1. Expression PHẢI assign vào biến tên 'alpha': alpha = <expression>
2. Output phải là CONTINUOUS signal (không phải binary 0/1 thuần túy)
3. Kết thúc bằng normalization: ts_zscore_scale(s, w) hoặc tanh()
4. Tránh signal quá sparse (>65% zeros)
5. KHÔNG dùng if/else, for loop, lambda
6. Dùng div(a, b) thay vì a/b để tránh chia cho 0
7. ts_zscore_scale(s, w) LUÔN cần đúng 2 args
8. Field names phải lowercase: df['close'] không phải df['Close']

Phản hồi BẮT BUỘC là JSON hợp lệ."""

ALPHA_INITIAL_PROMPT = """
Hypothesis: {hypothesis}

Hãy implement {num_factors} alpha expressions. Mỗi expression implement MỘT khía cạnh khác nhau của hypothesis.

{rag_examples}

{data_fields}
{operators}

Lưu ý về examples: dùng làm inspiration về cấu trúc, không copy trực tiếp.

Trả về JSON:
{{
  "alphas": [
    {{
      "id": "alpha_1",
      "description": "mô tả ngắn signal này",
      "expression": "alpha = <công thức dùng operators và df['field']>"
    }},
    ...
  ]
}}
"""

ALPHA_ITERATION_PROMPT = """
Hypothesis: {hypothesis}

Alphas yếu cần thay thế (alpha seed của vòng trước, chưa qua GP):
{weak_alphas}

Alphas tốt đang giữ (tránh trùng lặp) (alpha seed của vòng trước, chưa qua GP):
{good_alphas}

Định hướng cải thiện từ analyst:
{refinement_directions}

{rag_examples}

Implement {num_factors} alpha expressions MỚI để cải thiện portfolio.

{data_fields}
{operators}

Trả về JSON:
{{
  "alphas": [
    {{
      "id": "alpha_X",
      "description": "mô tả ngắn",
      "expression": "alpha = <công thức>"
    }}
  ]
}}
"""