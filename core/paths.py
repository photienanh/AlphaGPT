"""Shared filesystem paths used across the project."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

ALPHA_FORMULA_DIR = str(DATA_DIR / "alpha_formulas")
ALPHA_VALUES_DIR = str(DATA_DIR / "alphas")
ALPHA_MEMORY_DIR = str(DATA_DIR / "alpha_memory")
FEATURES_DIR = str(DATA_DIR / "features")
PRICE_DIR = str(DATA_DIR / "price")
RAW_NEWS_DIR = str(DATA_DIR / "raw_news")
DAILY_SCORES_DIR = str(DATA_DIR / "daily_scores")
SENTIMENT_OUTPUT_DIR = str(DATA_DIR / "sentiment_output")
SIGNALS_DIR = str(DATA_DIR / "signals")
