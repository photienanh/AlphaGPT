"""
alpha_kb_data.py
Danh sách alphas từ Kakushadze (2016) "101 Formulaic Alphas".
Chỉ giữ những alpha dùng fields: open/high/low/close/volume/vwap/adv20/returns
và operators đã implement. Loại bỏ các alpha cần: cap, IndClass level chi tiết
(subindustry, industry) khi không có data, adv với d khác 20.

Mapping operator names:
  Ts_Rank         → ts_rank
  Ts_ArgMax       → ts_argmax
  Ts_ArgMin       → ts_argmin
  SignedPower      → signed_power
  decay_linear    → ts_decayed_linear (hoặc decay_linear alias)
  rank(x)         → rank(x)          (cross-sectional expanding rank)
  delay(x,d)      → shift(x,d)  hoặc delay(x,d) (alias đã có)
  delta(x,d)      → ts_delta(x,d) hoặc delta(x,d) (alias)
  correlation     → ts_corr hoặc correlation (alias)
  covariance      → ts_cov hoặc covariance (alias)
  stddev          → ts_std hoặc stddev (alias)
  sum(x,d)        → ts_sum(x,d) hoặc sum_op(x,d)
  scale(x)        → scale(x)
  sign(x)         → sign(x)
  abs(x)          → abso(x)
  log(x)          → log(x)
  indneutralize(x, industry) → indneutralize(x, df['industry'])
"""

ALPHA_KB: list = [
    {
        "id": "alpha002",
        "expression": "alpha = neg(ts_corr(rank(ts_delta(log(df['volume']), 2)), rank(div(minus(df['close'], df['open']), df['open'])), 6))",
        "description": "Negative correlation between volume change rank and price return rank over 6 days. Captures divergence between volume and price momentum.",
        "family": "volume",
    },
    {
        "id": "alpha003",
        "expression": "alpha = neg(ts_corr(rank(df['open']), rank(df['volume']), 10))",
        "description": "Negative correlation between open price rank and volume rank over 10 days.",
        "family": "volume",
    },
    {
        "id": "alpha004",
        "expression": "alpha = neg(ts_rank(rank(df['low']), 9))",
        "description": "Negative time-series rank of cross-sectional rank of low prices over 9 days.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha006",
        "expression": "alpha = neg(ts_corr(df['open'], df['volume'], 10))",
        "description": "Negative correlation between open price and volume over 10 days.",
        "family": "volume",
    },
    {
        "id": "alpha007",
        "expression": "alpha = ts_zscore_scale(neg(ts_rank(abso(ts_delta(df['close'], 7)), 60)), 20)",
        "description": "When volume exceeds adv20, short recent large moves. Captures mean-reversion after volatile periods.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha008",
        "expression": "alpha = neg(rank(minus(cwise_mul(ts_sum(df['open'], 5), ts_sum(df['returns'], 5)), delay(cwise_mul(ts_sum(df['open'], 5), ts_sum(df['returns'], 5)), 10))))",
        "description": "Negative rank of change in (sum_open * sum_returns) product over 10-day lag.",
        "family": "momentum",
    },
    {
        "id": "alpha011",
        "expression": "alpha = cwise_mul(add(rank(ts_max(minus(df['vwap'], df['close']), 3)), rank(ts_min(minus(df['vwap'], df['close']), 3))), rank(ts_delta(df['volume'], 3)))",
        "description": "Combines VWAP-close spread extremes with volume change. Captures liquidity-driven price patterns.",
        "family": "volume",
    },
    {
        "id": "alpha012",
        "expression": "alpha = cwise_mul(sign(ts_delta(df['volume'], 1)), neg(ts_delta(df['close'], 1)))",
        "description": "Sign of volume change times negative price change. Mean-reversion: rising volume with falling price.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha013",
        "expression": "alpha = neg(rank(ts_cov(rank(df['close']), rank(df['volume']), 5)))",
        "description": "Negative rank of covariance between close rank and volume rank. Stocks where price-volume correlation is high tend to revert.",
        "family": "volume",
    },
    {
        "id": "alpha014",
        "expression": "alpha = cwise_mul(neg(rank(ts_delta(df['returns'], 3))), ts_corr(df['open'], df['volume'], 10))",
        "description": "Return momentum change combined with open-volume correlation.",
        "family": "momentum",
    },
    {
        "id": "alpha015",
        "expression": "alpha = neg(ts_sum(rank(ts_corr(rank(df['high']), rank(df['volume']), 3)), 3))",
        "description": "Sum of negative rank of high-volume correlation. Contrarian on high-price high-volume periods.",
        "family": "volume",
    },
    {
        "id": "alpha016",
        "expression": "alpha = neg(rank(ts_cov(rank(df['high']), rank(df['volume']), 5)))",
        "description": "Negative rank of covariance between high price rank and volume rank over 5 days.",
        "family": "volume",
    },
    {
        "id": "alpha018",
        "expression": "alpha = neg(rank(add(add(ts_std(abso(minus(df['close'], df['open'])), 5), minus(df['close'], df['open'])), ts_corr(df['close'], df['open'], 10))))",
        "description": "Combines intraday range volatility, price change, and close-open correlation.",
        "family": "volatility",
    },
    {
        "id": "alpha022",
        "expression": "alpha = neg(cwise_mul(ts_delta(ts_corr(df['high'], df['volume'], 5), 5), rank(ts_std(df['close'], 20))))",
        "description": "Change in high-volume correlation times close volatility. Captures regime shifts.",
        "family": "volatility",
    },
    {
        "id": "alpha023",
        "expression": "alpha = ts_zscore_scale(cwise_mul(greater(df['high'], div(ts_sum(df['high'], 20), 20.0)), neg(ts_delta(df['high'], 2))), 20)",
        "description": "When high exceeds 20-day average high, short the recent high move.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha025",
        "expression": "alpha = rank(cwise_mul(cwise_mul(cwise_mul(neg(df['returns']), df['adv20']), df['vwap']), minus(df['high'], df['close'])))",
        "description": "Rank of negative returns × adv20 × vwap × (high-close). Multi-factor combination.",
        "family": "momentum",
    },
    {
        "id": "alpha026",
        "expression": "alpha = neg(ts_max(ts_corr(ts_rank(df['volume'], 5), ts_rank(df['high'], 5), 5), 3))",
        "description": "Negative max correlation between volume rank and high rank over 3 days.",
        "family": "volume",
    },
    {
        "id": "alpha028",
        "expression": "alpha = scale(add(ts_corr(df['adv20'], df['low'], 5), div(add(df['high'], df['low']), 2.0)))",
        "description": "Scale of adv20-low correlation plus midpoint price.",
        "family": "volume",
    },
    {
        "id": "alpha030",
        "expression": "alpha = cwise_mul(div(minus(1.0, rank(add(add(sign(minus(df['close'], delay(df['close'], 1))), sign(minus(delay(df['close'], 1), delay(df['close'], 2)))), sign(minus(delay(df['close'], 2), delay(df['close'], 3)))))), ts_sum(df['volume'], 5)), div(1.0, ts_sum(df['volume'], 20)))",
        "description": "Contrarian on consecutive price direction patterns weighted by volume ratio.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha033",
        "expression": "alpha = rank(neg(div(df['open'], df['close'])))",
        "description": "Rank of negative open/close ratio. Stocks opening high relative to close are expected to revert.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha034",
        "expression": "alpha = rank(add(minus(1.0, rank(div(ts_std(df['returns'], 2), ts_std(df['returns'], 5)))), minus(1.0, rank(ts_delta(df['close'], 1)))))",
        "description": "Combines short-term vs medium-term volatility ratio rank and price change rank.",
        "family": "volatility",
    },
    {
        "id": "alpha035",
        "expression": "alpha = cwise_mul(cwise_mul(ts_rank(df['volume'], 32), minus(1.0, ts_rank(add(df['close'], df['high']), 16))), minus(1.0, ts_rank(df['returns'], 32)))",
        "description": "Volume rank × inverse (close+high) rank × inverse return rank. Volume momentum with price reversion.",
        "family": "volume",
    },
    {
        "id": "alpha037",
        "expression": "alpha = add(rank(ts_corr(delay(minus(df['open'], df['close']), 1), df['close'], 200)), rank(minus(df['open'], df['close'])))",
        "description": "Intraday range correlation with close over 200 days plus current range.",
        "family": "momentum",
    },
    {
        "id": "alpha038",
        "expression": "alpha = cwise_mul(neg(ts_rank(df['close'], 10)), rank(div(df['close'], df['open'])))",
        "description": "Negative time-series rank of close times rank of close/open ratio.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha040",
        "expression": "alpha = cwise_mul(neg(rank(ts_std(df['high'], 10))), ts_corr(df['high'], df['volume'], 10))",
        "description": "Negative rank of high volatility times high-volume correlation.",
        "family": "volatility",
    },
    {
        "id": "alpha041",
        "expression": "alpha = minus(signed_power(cwise_mul(df['high'], df['low']), 0.5), df['vwap'])",
        "description": "Geometric mean of high-low minus VWAP. Measures intraday price center vs VWAP.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha042",
        "expression": "alpha = div(rank(minus(df['vwap'], df['close'])), rank(add(df['vwap'], df['close'])))",
        "description": "Ratio of VWAP-close difference rank to VWAP+close sum rank. Delay-0 mean-reversion.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha043",
        "expression": "alpha = cwise_mul(ts_rank(div(df['volume'], df['adv20']), 20), ts_rank(neg(ts_delta(df['close'], 7)), 8))",
        "description": "Volume ratio rank times inverse price momentum rank. High-volume contrarian.",
        "family": "volume",
    },
    {
        "id": "alpha044",
        "expression": "alpha = neg(ts_corr(df['high'], rank(df['volume']), 5))",
        "description": "Negative correlation between high price and volume rank over 5 days.",
        "family": "volume",
    },
    {
        "id": "alpha046",
        "expression": "alpha = ts_zscore_scale(ts_delta(df['close'], 1), 20)",
        "description": "Momentum/mean-reversion based on longer-term trend acceleration. Simplified from original ternary logic.",
        "family": "momentum",
    },
    {
        "id": "alpha050",
        "expression": "alpha = neg(ts_max(rank(ts_corr(rank(df['volume']), rank(df['vwap']), 5)), 5))",
        "description": "Negative max of rank of volume-VWAP correlation over 5 days.",
        "family": "volume",
    },
    {
        "id": "alpha052",
        "expression": "alpha = cwise_mul(cwise_mul(add(neg(ts_min(df['low'], 5)), delay(ts_min(df['low'], 5), 5)), rank(div(minus(ts_sum(df['returns'], 240), ts_sum(df['returns'], 20)), 220.0))), ts_rank(df['volume'], 5))",
        "description": "Low price bounce combined with long-term return minus short-term return, weighted by volume.",
        "family": "momentum",
    },
    {
        "id": "alpha053",
        "expression": "alpha = neg(ts_delta(div(minus(minus(df['close'], df['low']), minus(df['high'], df['close'])), add(minus(df['close'], df['low']), 1e-9)), 9))",
        "description": "Change in (close position within high-low range) over 9 days.",
        "family": "mean_reversion",
    },
    {
        "id": "alpha054",
        "expression": "alpha = neg(div(cwise_mul(minus(df['low'], df['close']), pow_op(df['open'], 5)), cwise_mul(minus(df['low'], df['high']), add(pow_op(df['close'], 5), 1e-9))))",
        "description": "Intraday price structure alpha based on open/close/high/low relationships.",
        "family": "pattern",
    },
    {
        "id": "alpha055",
        "expression": "alpha = neg(ts_corr(rank(div(minus(df['close'], ts_min(df['low'], 12)), add(minus(ts_max(df['high'], 12), ts_min(df['low'], 12)), 1e-9))), rank(df['volume']), 6))",
        "description": "Negative correlation between stochastic-like price rank and volume rank over 6 days.",
        "family": "volume",
    },
    {
        "id": "alpha060",
        "expression": "alpha = minus(scale(rank(cwise_mul(div(minus(minus(df['close'], df['low']), minus(df['high'], df['close'])), add(minus(df['high'], df['low']), 1e-9)), df['volume']))), scale(rank(ts_argmax(df['close'], 10))))",
        "description": "Scaled money flow pressure minus scaled recent high rank. Combines volume and momentum.",
        "family": "volume",
    },
    {
        "id": "alpha083",
        "expression": "alpha = div(cwise_mul(rank(delay(div(minus(df['high'], df['low']), add(div(ts_sum(df['close'], 5), 5.0), 1e-9)), 2)), rank(rank(df['volume']))), div(div(minus(df['high'], df['low']), add(div(ts_sum(df['close'], 5), 5.0), 1e-9)), add(minus(df['vwap'], df['close']), 1e-9)))",
        "description": "Ratio of lagged HL-range rank to current HL-range normalized by VWAP-close.",
        "family": "volatility",
    },
    {
        "id": "alpha101",
        "expression": "alpha = div(minus(df['close'], df['open']), add(minus(df['high'], df['low']), 0.001))",
        "description": "Intraday return normalized by intraday range. Positive means close > open relative to volatility.",
        "family": "momentum",
    },
]