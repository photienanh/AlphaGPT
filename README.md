# Alpha-GPT VN30 — Project Structure & Guide

```
alpha_gpt/
│
├── app.py                          # Flask REST API + Dashboard server
│
├── core/                           # Core modules (stateless utilities)
│   ├── alpha_memory.py             # ★ NEW: RAG Knowledge Library
│   ├── alpha_operators.py          # All formulaic operators for alpha expressions
│   ├── backtester.py               # ★ UPDATED: Walk-forward OOS evaluation
│   ├── daily_runner.py             # Daily pipeline: refresh → recompute → signal
│   ├── genetic_search.py           # ★ NEW: GP enhancement (paper Section 4.2)
│   ├── paths.py                    # Filesystem paths
│   └── universe.py                 # VN30 symbols + industry mapping
│
├── pipelines/
│   ├── gen_alpha.py                # ★ UPDATED: Full Alpha-GPT pipeline (v3)
│   ├── indicators.py               # Technical indicator computation
│   ├── sentiment.py                # News crawl + sentiment scoring
│   └── stock_data.py               # OHLCV data download via vnstock
│
├── templates/
│   └── dashboard.html              # ★ UPDATED: Web dashboard (OOS metrics, GP tag)
│
└── data/                           # Auto-created by pipelines
    ├── price/                      # Raw OHLCV CSVs (SSI.csv, HPG.csv, ...)
    ├── features/                   # OHLCV + Technical indicators
    ├── raw_news/                   # Crawled news titles
    ├── daily_scores/               # Per-ticker daily sentiment scores
    ├── sentiment_output/           # Merged sentiment features
    ├── alpha_formulas/             # Alpha definitions + metrics (JSON)
    ├── alphas/                     # Alpha value time-series (CSV)
    ├── alpha_memory/               # ★ NEW: RAG memory store (JSON)
    └── signals/                    # Daily signal outputs (CSV)
```

## Quick Start

### 1. Install dependencies
```bash
pip install flask flask-cors openai vnstock pandas numpy python-dotenv beautifulsoup4 requests
```

### 2. Set API key
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Download data (one-time)
```bash
python pipelines/stock_data.py        # OHLCV for VN30
python pipelines/indicators.py        # Compute technical indicators
python pipelines/sentiment.py         # Crawl news + score sentiment (slow ~1hr)
```

### 4. Generate alphas
```bash
# Single ticker (recommended to start)
python pipelines/gen_alpha.py --ticker HPG

# Faster (skip GP enhancement)
python pipelines/gen_alpha.py --ticker HPG --no-gp

# All VN30 (long, ~30 min)
python pipelines/gen_alpha.py --all

# Refine existing alphas (keep good, replace weak)
python pipelines/gen_alpha.py --ticker HPG --refine-only
```

### 5. Launch dashboard
```bash
python app.py
# Open: http://localhost:5000
```

### 6. Daily update
```bash
python -c "
from core.daily_runner import refresh_market_data, refresh_alpha_values, run_daily
from core.universe import VN30_SYMBOLS
refresh_market_data(VN30_SYMBOLS)
refresh_alpha_values(VN30_SYMBOLS)
signals = run_daily(VN30_SYMBOLS)
print(signals[['ticker','signal','action','rank']].head(10))
"
```