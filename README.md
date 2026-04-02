# Alpha-GPT · VN30 Quant System

Hệ thống sinh alpha tự động kết hợp Human-AI interaction theo paper Alpha-GPT (2308.00016v2),
áp dụng cho thị trường VN30 Việt Nam.

## Kiến trúc

```
alpha_gpt/
├── pipelines/
│   ├── stock_data.py         ← Crawl OHLCV từ vnstock
│   ├── indicators.py         ← Tính technical indicators
│   ├── sentiment.py          ← Crawl tin tức + sentiment (GPT-4o-mini)
│   └── gen_alpha.py          ← Sinh công thức alpha (LLM + refinement loop)
├── app.py                    ← Flask REST API backend
├── core/
│   ├── alpha_operators.py    ← Toán tử alpha (ts_*, cross-sectional, element-wise)
│   ├── backtester.py         ← IC, Sharpe, Turnover, decay detection
│   ├── daily_runner.py       ← Pipeline hàng ngày: load alpha → compute → signal
│   ├── universe.py           ← VN30 symbols + industry map (shared constants)
│   └── paths.py              ← Shared filesystem paths
├── templates/
│   └── dashboard.html        ← Frontend SPA (vanilla JS + Chart.js)
└── data/
	├── alpha_formulas/       ← {TICKER}_alphas.json (công thức + metrics)
	├── alphas/               ← {TICKER}_alpha_values.csv (series values)
	├── daily_scores/         ← daily sentiment per symbol
	├── features/             ← {TICKER}.csv (OHLCV + technical indicators)
	├── price/                ← OHLCV raw prices
	├── raw_news/             ← crawled news titles by symbol
	├── sentiment_output/     ← {TICKER}_Full_Sentiment.csv
	└── signals/              ← signals_{YYYYMMDD}.csv (daily output)
```

## Quy trình đầy đủ

### Bước 1 — Thu thập dữ liệu (chạy 1 lần + update hàng ngày)

```bash
# Crawl giá cổ phiếu (3 năm)
python pipelines/stock_data.py

# Tính technical indicators
python pipelines/indicators.py

# Crawl tin tức + tính sentiment (cần OPENAI_API_KEY)
python pipelines/sentiment.py
```

### Bước 2 — Sinh alpha formulas (chạy 1 lần, refinement định kỳ)

```bash
# Sinh alpha cho 1 mã
python pipelines/gen_alpha.py --ticker STB

# Sinh alpha cho tất cả 30 mã VN30
python pipelines/gen_alpha.py --all

# Force re-generate (bỏ qua cache)
python pipelines/gen_alpha.py --ticker STB --force

# Tăng số vòng refinement
python pipelines/gen_alpha.py --ticker STB --rounds 5
```

### Bước 3 — Chạy dashboard

```bash
# Cài dependencies
pip install -r requirements.txt

# Khởi động API backend
python app.py
# → API tại http://localhost:5000/api

# Mở dashboard trong browser
open templates/dashboard.html
# Hoặc serve qua Flask: http://localhost:5000
```

### Bước 4 — Pipeline hàng ngày (cron job)

```bash
# Thêm vào crontab - chạy lúc 8:45 sáng mỗi ngày giao dịch
# 45 8 * * 1-5 cd /path/to/alpha_gpt && python -c "
# from core.daily_runner import run_daily
# import json
# tickers = json.load(open('data/alpha_formulas/VN30_list.json'))
# run_daily(tickers)
# "
```

## API Endpoints

| Endpoint | Mô tả |
|---|---|
| `GET /api/overview` | Tóm tắt tất cả 30 mã (signal, action, IC, Sharpe) |
| `GET /api/ticker/{TICKER}` | Chi tiết 1 mã: alphas, rolling IC, composite signal |
| `GET /api/signals` | Ranked buy/sell signals hôm nay |
| `GET /api/decay` | Các alpha đang bị decay |
| `POST /api/gen/{TICKER}` | Trigger re-generate alpha (async) |
| `GET /api/ticker/{TICKER}/history` | Lịch sử refinement rounds |

## Cách đọc tín hiệu giao dịch

```
Composite Signal > +1σ  → BUY  (mua ATO lúc 9:00)
Composite Signal < -1σ  → SELL (tránh / bán nếu đang giữ)
-1σ ≤ Signal ≤ +1σ     → HOLD
```

Composite signal = weighted average của 5 alpha, trọng số = composite score (IC × Sharpe).
Signal được z-score normalize trên rolling 60 ngày để so sánh tương đối giữa các mã.

## Alpha Decay Loop

Mỗi tuần/tháng, dashboard → tab "Alpha Decay" kiểm tra IC rolling 20d.
Nếu IC giảm > 30% so với lịch sử → alpha được đánh dấu cần refinement.
Click "Refinement" → trigger `pipelines/gen_alpha.py --ticker X --force` để sinh alpha mới.

## Môi trường

```bash
# .env
OPENAI_API_KEY=sk-...
```

## Liên quan

- Paper: Alpha-GPT: Human-AI Interactive Alpha Mining (arXiv 2308.00016)
- Repo tham khảo: https://github.com/parthmodi152/alpha-gpt
- Thực nghiệm: file `Thực_nghiệm_AlphaGPT.pdf`
