# prompts/alpha_prompts.py

OPERATOR_SIGNATURES = """
## Operators có sẵn (alpha_operators.py) — dùng đúng tên hàm, đúng số args

### Time-series
shift(s, period)
ts_delta(s, period)          ts_delta_ratio(s, period)
ts_mean(s, w)                ts_std(s, w)
ts_sum(s, w)                 ts_min(s, w)          ts_max(s, w)
ts_rank(s, w)                ts_median(s, w)
ts_zscore_scale(s, w)        ts_maxmin_scale(s, w)
ts_skew(s, w)                ts_kurt(s, w)
ts_corr(s1, s2, w)           ts_cov(s1, s2, w)
ts_ir(s, w)                  ts_linear_reg(s, w)
ts_ema(s, span)              ts_decayed_linear(s, w)
ts_argmaxmin_diff(s, w)      ts_max_diff(s, w)     ts_min_diff(s, w)

### Group-wise
grouped_mean(s, w)           grouped_std(s, w)
grouped_demean(s, w)         grouped_zscore_scale(s, w)

### Element-wise
add(s1, s2)    minus(s1, s2)    cwise_mul(s1, s2)    div(s1, s2)
relu(s)        neg(s)           abso(s)               sign(s)
log(s)         log1p(s)         tanh(s)               clip(s, lower, upper)
pow_op(s, exp) pow_sign(s, exp)
greater(s1, s2)   less(s1, s2)
cwise_max(s1, s2) cwise_min(s1, s2)
normed_rank(s)    normed_rank_diff(s1, s2)
zscore_scale(s)   winsorize_scale(s)
"""

DATA_FIELDS_BLOCK = """
## Data fields có sẵn trong df (pd.DataFrame, index = datetime)

CẢNH BÁO: Chỉ được dùng ĐÚNG các tên field sau. KHÔNG được tự đặt tên khác
(ví dụ: KHÔNG dùng SMA_50, MA50, SMA50, RSI, BB_Band — các tên đó không tồn tại).

### OHLCV
df['open']    df['high']    df['low']    df['close']    df['volume']

### Moving averages
df['SMA_5']    df['SMA_20']    df['EMA_10']

### Momentum
df['Momentum_3']    df['Momentum_10']

### Oscillators
df['RSI_14']    df['MACD']    df['MACD_Signal']

### Bollinger Bands
df['BB_Upper']    df['BB_Middle']    df['BB_Lower']

### Volume
df['OBV']

Danh sách đầy đủ và duy nhất — không có field nào khác ngoài danh sách này.
"""

ALPHA_SYSTEM_PROMPT = """Bạn là Quant Developer — chuyên gia implement formulaic alpha signals.

Nhiệm vụ: Nhận trading hypothesis và implement thành Python expressions dùng bộ operators đã cho.

Nguyên tắc bắt buộc:
1. Expression PHẢI assign vào biến tên 'alpha': alpha = <expression>
2. Output phải là CONTINUOUS signal (không phải binary 0/1 thuần túy)
3. Kết thúc bằng normalization: ts_zscore_scale(s, w) hoặc tanh()
4. Tránh signal quá sparse (>65% zeros)
5. KHÔNG dùng if/else, for loop, lambda
6. div(a, b) thay vì a/b để tránh chia cho 0
7. ts_zscore_scale(s, w) LUÔN cần đúng 2 args

QUAN TRỌNG — Các field KHÔNG tồn tại, KHÔNG được dùng:
Phản hồi BẮT BUỘC là JSON hợp lệ."""

ALPHA_INITIAL_PROMPT = """
Hypothesis: {hypothesis}

Hãy implement {num_factors} alpha expressions. Mỗi expression implement MỘT khía cạnh khác nhau của hypothesis.

{data_fields}
{operators}

Trả về JSON:
{{
  "alphas": [
    {{
      "id": "alpha_1",
      "family": "<momentum|mean_reversion|volume|volatility|technical|pattern>",
      "description": "mô tả ngắn signal này capture gì",
      "expression": "alpha = <công thức dùng operators và df['field']>"
    }},
    ...
  ]
}}
"""

ALPHA_ITERATION_PROMPT = """
Hypothesis: {hypothesis}

Analyst feedback từ vòng trước:
{analyst_feedback}

Alphas yếu cần thay thế:
{weak_alphas}

Alphas tốt đang giữ (tránh trùng lặp):
{good_alphas}

Implement {num_factors} alpha expressions MỚI để cải thiện portfolio.

{data_fields}
{operators}

Trả về JSON:
{{
  "alphas": [
    {{
      "id": "alpha_X",
      "family": "<family>",
      "description": "mô tả ngắn",
      "expression": "alpha = <công thức>"
    }}
  ]
}}
"""