"""
core/daily_runner.py — Daily pipeline: load saved alphas → compute values → rank → signals
"""
import os
import json
import argparse
import logging
import sys
import importlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from core.universe import VN30_SYMBOLS
    from core.paths import (
        ALPHA_FORMULA_DIR,
        ALPHA_VALUES_DIR,
        FEATURES_DIR,
        PRICE_DIR,
        SENTIMENT_OUTPUT_DIR,
        DAILY_SCORES_DIR,
        RAW_NEWS_DIR,
        SIGNALS_DIR,
        BASE_DIR,
    )
except ModuleNotFoundError:
    from universe import VN30_SYMBOLS
    from paths import (
        ALPHA_FORMULA_DIR,
        ALPHA_VALUES_DIR,
        FEATURES_DIR,
        PRICE_DIR,
        SENTIMENT_OUTPUT_DIR,
        DAILY_SCORES_DIR,
        RAW_NEWS_DIR,
        SIGNALS_DIR,
        BASE_DIR,
    )

log = logging.getLogger(__name__)

SENTIMENT_DIR = SENTIMENT_OUTPUT_DIR

os.makedirs(SIGNALS_DIR, exist_ok=True)


# ─── Incremental market data refresh ─────────────────────────────────────────

def _fetch_price_history(symbol: str, lookback_days: int = 90) -> pd.DataFrame | None:
    """Fetch recent OHLCV from vnstock. Returns normalized time-indexed frame."""
    try:
        from vnstock import Quote
    except Exception:
        log.warning("vnstock not available. Skip market data refresh.")
        return None

    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        quote = Quote(symbol=symbol, source="KBS")
        raw = pd.DataFrame(
            quote.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="d",
            )
        )
    except Exception as e:
        log.warning(f"[{symbol}] fetch history failed: {e}")
        return None

    if raw.empty or "time" not in raw.columns:
        return None

    cols = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in raw.columns]
    df = raw[cols].copy()
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.dropna(subset=["time", "close"]).sort_values("time")
    return df.reset_index(drop=True)


def _latest_market_trading_day(anchor_symbol: str = "VCB") -> pd.Timestamp | None:
    """
    Use actual exchange data to infer latest trading day.
    This naturally excludes weekends/holidays without manual holiday calendar.
    """
    df = _fetch_price_history(anchor_symbol, lookback_days=30)
    if df is None or df.empty:
        return None
    return pd.to_datetime(df["time"]).max().normalize()


def _append_price_rows(ticker: str, target_day: pd.Timestamp) -> tuple[bool, int]:
    """Append only missing rows into existing price file. Returns (changed, n_new_rows)."""
    price_path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    os.makedirs(PRICE_DIR, exist_ok=True)

    old_df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    last_local_day = None
    if os.path.exists(price_path):
        old_df = pd.read_csv(price_path)
        if not old_df.empty and "time" in old_df.columns:
            old_df["time"] = pd.to_datetime(old_df["time"]).dt.normalize()
            last_local_day = old_df["time"].max().normalize()

    if last_local_day is not None and last_local_day >= target_day:
        return False, 0

    remote_df = _fetch_price_history(ticker, lookback_days=120)
    if remote_df is None or remote_df.empty:
        return False, 0

    if last_local_day is not None:
        new_rows = remote_df[remote_df["time"] > last_local_day]
    else:
        new_rows = remote_df

    if new_rows.empty:
        return False, 0

    merged = pd.concat([old_df, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["time"], keep="last").sort_values("time")
    merged.to_csv(price_path, index=False)
    return True, len(new_rows)


def _last_day_in_csv(path: str, time_col: str = "time") -> pd.Timestamp | None:
    """Return latest normalized day from CSV time column, or None."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=[time_col])
        if df.empty or time_col not in df.columns:
            return None
        s = pd.to_datetime(df[time_col], errors="coerce").dropna()
        if s.empty:
            return None
        return s.max().normalize()
    except Exception:
        return None


def _refresh_features_for_ticker(ticker: str) -> bool:
    """Rebuild feature file from updated price file and overwrite same file."""
    try:
        base_path = str(BASE_DIR)
        if base_path not in sys.path:
            sys.path.insert(0, base_path)
        mod = importlib.import_module("pipelines.indicators")
        add_technical_indicators = mod.add_technical_indicators
    except Exception as e:
        log.warning(f"[{ticker}] cannot import indicators module: {e}")
        return False

    price_path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")

    if not os.path.exists(price_path):
        return False

    try:
        src = pd.read_csv(price_path)
        feat = add_technical_indicators(src)
        os.makedirs(FEATURES_DIR, exist_ok=True)
        feat.to_csv(feat_path, index=False)
        return True
    except Exception as e:
        log.warning(f"[{ticker}] rebuild features failed: {e}")
        return False


def refresh_latest_market_data(tickers: list[str]) -> dict:
    """
    Ensure local price/features include latest trading day.
    - No separate bulky files.
    - Append into existing price files.
    - Overwrite same features file after recomputation.
    """
    target_day = _latest_market_trading_day()
    if target_day is None:
        log.warning("Cannot determine latest market trading day. Skip refresh.")
        return {"target_day": None, "updated": [], "skipped": tickers, "failed": []}

    updated = []
    refreshed_only = []
    skipped = []
    failed = []

    log.info(f"Latest market trading day inferred: {target_day.date()}")
    for ticker in tickers:
        try:
            changed, n_new = _append_price_rows(ticker, target_day)
            price_path = os.path.join(PRICE_DIR, f"{ticker}.csv")
            feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
            price_last = _last_day_in_csv(price_path)
            feat_last = _last_day_in_csv(feat_path)

            needs_feature_refresh = changed or (price_last is not None and (feat_last is None or feat_last < price_last))

            if not needs_feature_refresh:
                skipped.append(ticker)
                continue

            if _refresh_features_for_ticker(ticker):
                if changed:
                    log.info(f"[{ticker}] appended {n_new} new row(s) and refreshed features")
                    updated.append(ticker)
                else:
                    log.info(f"[{ticker}] refreshed stale features (price_last={price_last}, feat_last={feat_last})")
                    refreshed_only.append(ticker)
            else:
                failed.append(ticker)
        except Exception as e:
            log.warning(f"[{ticker}] refresh failed: {e}")
            failed.append(ticker)

    return {
        "target_day": str(target_day.date()),
        "updated": updated,
        "refreshed_only": refreshed_only,
        "skipped": skipped,
        "failed": failed,
    }


def _load_daily_score_value(score_cache: dict, ticker_col: str, target_day: pd.Timestamp) -> float:
    """Read sentiment score for one ticker column (e.g. VCB_S) on target day from daily_scores."""
    peer = ticker_col.replace("_S", "")
    if peer not in score_cache:
        score_path = os.path.join(DAILY_SCORES_DIR, f"{peer}.csv")
        if os.path.exists(score_path):
            df = pd.read_csv(score_path)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"]).dt.normalize()
            score_cache[peer] = df
        else:
            score_cache[peer] = pd.DataFrame()

    df = score_cache[peer]
    if df.empty:
        return 0.0

    if ticker_col not in df.columns or "time" not in df.columns:
        return 0.0

    day_rows = df[df["time"] == target_day]
    if day_rows.empty:
        return 0.0

    v = day_rows.iloc[-1][ticker_col]
    try:
        fv = float(v)
        if np.isfinite(fv):
            return fv
    except Exception:
        pass
    return 0.0


def _load_sentiment_module():
    """Load sentiment pipeline module lazily to reuse parse/scoring logic."""
    try:
        base_path = str(BASE_DIR)
        if base_path not in sys.path:
            sys.path.insert(0, base_path)
        return importlib.import_module("pipelines.sentiment")
    except Exception as e:
        log.warning(f"Cannot load sentiment module: {e}")
        return None


def _crawl_incremental_raw_news(symbol: str, sent_mod, max_pages: int = 30) -> tuple[pd.DataFrame, set[str], int]:
    """
    Crawl incremental news from last date in raw_news/{symbol}.csv to today.
    Deduplicate by (date, title). Returns (updated_raw_df, affected_dates, new_rows_count).
    """
    raw_path = os.path.join(RAW_NEWS_DIR, f"{symbol}.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    raw_df = pd.DataFrame(columns=["ticker", "date", "title"])
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        try:
            raw_df = pd.read_csv(raw_path)
        except Exception:
            raw_df = pd.DataFrame(columns=["ticker", "date", "title"])

    for col in ["ticker", "date", "title"]:
        if col not in raw_df.columns:
            raw_df[col] = ""

    raw_df = raw_df[["ticker", "date", "title"]].copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    raw_df = raw_df.dropna(subset=["date"])

    today = pd.Timestamp.today().normalize()
    if raw_df.empty:
        crawl_from_day = (today - pd.Timedelta(days=7)).normalize()
    else:
        crawl_from_day = pd.to_datetime(raw_df["date"]).max().normalize()

    existing_keys = set(
        (str(d), str(t).strip().lower())
        for d, t in zip(raw_df["date"], raw_df["title"])
        if pd.notna(d) and pd.notna(t)
    )

    new_rows: list[dict] = []
    affected_dates: set[str] = set()

    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                sent_mod.CAFEF_AJAX_URL,
                params={
                    "symbol": symbol,
                    "floorID": "0",
                    "configID": "0",
                    "PageIndex": str(page),
                    "PageSize": "30",
                    "Type": "2",
                },
                headers=sent_mod.HEADERS,
                timeout=10,
            )
            body = resp.text.strip()
            if not body or "<li" not in body:
                break

            soup = BeautifulSoup(body, "html.parser")
            seen_older = False
            has_new_in_page = False

            for item in soup.find_all("li"):
                a_tag = item.find("a", class_="docnhanhTitle") or item.find("a")
                span_tag = item.find("span", class_="timeTitle")
                if not a_tag or not span_tag:
                    continue

                title = (a_tag.get("title") or a_tag.get_text(strip=True) or "").strip()
                date_str = sent_mod.parse_date(span_tag.get_text(strip=True))
                if not title or not date_str:
                    continue

                dt = pd.Timestamp(date_str).normalize()
                if dt < crawl_from_day:
                    seen_older = True
                    continue
                if dt > today:
                    continue

                k = (date_str, title.lower())
                if k in existing_keys:
                    continue

                existing_keys.add(k)
                new_rows.append({"ticker": symbol, "date": date_str, "title": title})
                affected_dates.add(date_str)
                has_new_in_page = True

            # Most Cafef pages are reverse-chronological; stop once we're clearly past window.
            if seen_older and not has_new_in_page and page > 1:
                break
        except Exception as e:
            log.warning(f"[{symbol}] crawl page {page} failed: {e}")
            break

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        merged = pd.concat([raw_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date", "title"], keep="last")
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        merged = merged.dropna(subset=["date"]).sort_values("date")
        merged.to_csv(raw_path, index=False)
        return merged, affected_dates, len(new_rows)

    if not raw_df.empty:
        raw_df = raw_df.sort_values("date")
    raw_df.to_csv(raw_path, index=False)
    return raw_df, affected_dates, 0


def _score_affected_dates(symbol: str, raw_df: pd.DataFrame, affected_dates: set[str], sent_mod) -> int:
    """Score affected dates and upsert into daily_scores/{symbol}.csv. Returns number of updated dates."""
    if not affected_dates:
        return 0

    score_path = os.path.join(DAILY_SCORES_DIR, f"{symbol}.csv")
    os.makedirs(DAILY_SCORES_DIR, exist_ok=True)

    col_name = f"{symbol}_S"
    score_df = pd.DataFrame(columns=["time", col_name])
    if os.path.exists(score_path) and os.path.getsize(score_path) > 0:
        try:
            score_df = pd.read_csv(score_path)
        except Exception:
            score_df = pd.DataFrame(columns=["time", col_name])

    if "time" not in score_df.columns:
        score_df["time"] = pd.Series(dtype="str")
    if col_name not in score_df.columns:
        score_df[col_name] = np.nan

    score_df["time"] = pd.to_datetime(score_df["time"], errors="coerce").dt.normalize()
    score_df = score_df.dropna(subset=["time"])

    updated_days = 0
    for d in sorted(affected_dates):
        titles = raw_df.loc[raw_df["date"] == d, "title"].dropna().astype(str).tolist()
        if not titles:
            day_score = 0.0
        else:
            raw_scores: list[int] = []
            batch_size = 20
            for i in range(0, len(titles), batch_size):
                batch = titles[i : i + batch_size]
                raw_scores.extend(sent_mod.analyze_sentiment_batch(batch))
            mapped = [float(sent_mod.SENTIMENT_MAP.get(s, 0)) for s in raw_scores]
            day_score = float(np.mean(mapped)) if mapped else 0.0

        day_ts = pd.Timestamp(d).normalize()
        mask = score_df["time"] == day_ts
        if mask.any():
            score_df.loc[mask, col_name] = day_score
        else:
            score_df = pd.concat(
                [score_df, pd.DataFrame([{"time": day_ts, col_name: day_score}])],
                ignore_index=True,
            )
        updated_days += 1

    score_df = score_df.drop_duplicates(subset=["time"], keep="last").sort_values("time")
    score_df["time"] = pd.to_datetime(score_df["time"]).dt.strftime("%Y-%m-%d")
    score_df.to_csv(score_path, index=False)
    return updated_days


def _collect_needed_sentiment_symbols(tickers: list[str]) -> set[str]:
    """
    Collect all symbols needed by selected sentiment_output files.
    This ensures related *_S columns are also refreshed before merge.
    """
    symbols: set[str] = set(tickers)
    for ticker in tickers:
        path = os.path.join(SENTIMENT_DIR, f"{ticker}_Full_Sentiment.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, nrows=1)
            for c in df.columns:
                if c.endswith("_S"):
                    symbols.add(c.replace("_S", ""))
        except Exception:
            continue
    return symbols


def _symbol_sentiment_up_to_date(symbol: str, target_day: pd.Timestamp) -> bool:
    """
    Fast skip guard: if both raw_news and daily_scores already have target day, skip crawl+score.
    """
    raw_path = os.path.join(RAW_NEWS_DIR, f"{symbol}.csv")
    score_path = os.path.join(DAILY_SCORES_DIR, f"{symbol}.csv")
    if not os.path.exists(raw_path) or not os.path.exists(score_path):
        return False

    try:
        raw = pd.read_csv(raw_path, usecols=["date"])
        if raw.empty:
            return False
        raw_last = pd.to_datetime(raw["date"], errors="coerce").dropna().max()
        if pd.isna(raw_last) or raw_last.normalize() < target_day:
            return False

        score = pd.read_csv(score_path, usecols=["time"])
        if score.empty:
            return False
        score_last = pd.to_datetime(score["time"], errors="coerce").dropna().max()
        if pd.isna(score_last):
            return False
        return score_last.normalize() >= target_day
    except Exception:
        return False


def refresh_latest_sentiment_data(tickers: list[str], target_day: pd.Timestamp | None) -> dict:
    """
    Ensure sentiment_output has the latest trading day row.
    Uses existing daily_scores files (no new bulky files).
    Missing values default to neutral 0.0.
    """
    sent_mod = _load_sentiment_module()
    if sent_mod is None:
        log.warning("Sentiment refresh skipped: sentiment module unavailable")
        return {
            "updated": [],
            "skipped": tickers,
            "failed": tickers,
            "symbols_crawled": 0,
            "news_rows_added": 0,
            "score_days_updated": 0,
        }

    if target_day is None:
        target_day = pd.Timestamp.today().normalize()

    symbols_to_refresh = sorted(_collect_needed_sentiment_symbols(tickers))
    total_new_rows = 0
    total_score_days = 0
    crawled_symbols = 0
    skipped_symbols = 0
    for i, symbol in enumerate(symbols_to_refresh, start=1):
        try:
            if _symbol_sentiment_up_to_date(symbol, target_day):
                skipped_symbols += 1
                if i % 10 == 0 or i == len(symbols_to_refresh):
                    log.info(
                        "Sentiment progress: %d/%d symbols processed (skip_up_to_date=%d)",
                        i,
                        len(symbols_to_refresh),
                        skipped_symbols,
                    )
                continue

            raw_df, affected_dates, new_rows = _crawl_incremental_raw_news(symbol, sent_mod)
            updated_days = _score_affected_dates(symbol, raw_df, affected_dates, sent_mod)
            total_new_rows += new_rows
            total_score_days += updated_days
            crawled_symbols += 1
            if new_rows > 0 or updated_days > 0:
                log.info(
                    "[%s] incremental sentiment: new_news=%d, updated_days=%d",
                    symbol,
                    new_rows,
                    updated_days,
                )
            if i % 10 == 0 or i == len(symbols_to_refresh):
                log.info(
                    "Sentiment progress: %d/%d symbols processed (crawled=%d, skip_up_to_date=%d)",
                    i,
                    len(symbols_to_refresh),
                    crawled_symbols,
                    skipped_symbols,
                )
        except Exception as e:
            log.warning(f"[{symbol}] incremental sentiment refresh failed: {e}")

    updated = []
    skipped = []
    failed = []
    score_cache: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        path = os.path.join(SENTIMENT_DIR, f"{ticker}_Full_Sentiment.csv")
        if not os.path.exists(path):
            failed.append(ticker)
            log.warning(f"[{ticker}] Missing sentiment file: {path}")
            continue

        try:
            df = pd.read_csv(path)
            if "time" not in df.columns:
                failed.append(ticker)
                log.warning(f"[{ticker}] Invalid sentiment file (no time column)")
                continue

            df["time"] = pd.to_datetime(df["time"]).dt.normalize()
            if (df["time"] == target_day).any():
                skipped.append(ticker)
                continue

            row = {"time": target_day}
            for c in df.columns:
                if c == "time":
                    continue
                row[c] = _load_daily_score_value(score_cache, c, target_day)

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time")
            df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
            df.to_csv(path, index=False)
            updated.append(ticker)
        except Exception as e:
            failed.append(ticker)
            log.warning(f"[{ticker}] sentiment refresh failed: {e}")

    return {
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "symbols_crawled": crawled_symbols,
        "symbols_skipped_up_to_date": skipped_symbols,
        "news_rows_added": total_new_rows,
        "score_days_updated": total_score_days,
    }


# ─── Load saved alpha formulas ───────────────────────────────────────────────

def load_alpha_meta(ticker: str) -> dict | None:
    path = os.path.join(ALPHA_FORMULA_DIR, f"{ticker}_alphas.json")
    if not os.path.exists(path):
        log.warning(f"[{ticker}] No alpha meta found at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_alpha_values(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(ALPHA_VALUES_DIR, f"{ticker}_alpha_values.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df


# ─── Compute alpha values from formula on new data ───────────────────────────

def compute_alpha_for_date(ticker: str, df_full: pd.DataFrame) -> pd.DataFrame | None:
    """
    Recompute alpha values using saved formulas.
    df_full = full feature + sentiment dataframe up to today.
    Returns DataFrame with columns alpha_1..alpha_5.
    """
    try:
        base_path = str(BASE_DIR)
        if base_path not in sys.path:
            sys.path.insert(0, base_path)
        op = importlib.import_module("core.alpha_operators")
    except Exception:
        log.error("core/alpha_operators.py not found")
        return None

    meta = load_alpha_meta(ticker)
    if meta is None:
        return None

    results = {}
    for a in meta["alphas"]:
        if a.get("status") != "OK":
            continue
        col = f"alpha_{a['id']}"
        expr = a["expression"]
        namespace = {name: getattr(op, name) for name in dir(op) if not name.startswith("_")}
        namespace.update({"df": df_full, "np": np, "pd": pd, "op": op})
        try:
            exec(expr, namespace)
            series = namespace.get("alpha")
            if isinstance(series, pd.Series):
                # Normalize
                norm = (series - series.mean()) / (series.std() + 1e-9)
                if a.get("flipped"):
                    norm = -norm
                results[col] = norm.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            log.warning(f"[{ticker}] Alpha {a['id']} eval error: {e}")
            results[col] = pd.Series(np.nan, index=df_full.index)

    if not results:
        return None
    return pd.DataFrame(results, index=df_full.index)


def _build_full_input_for_alpha(ticker: str) -> pd.DataFrame | None:
    feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
    sent_path = os.path.join(SENTIMENT_DIR, f"{ticker}_Full_Sentiment.csv")

    if not os.path.exists(feat_path) or not os.path.exists(sent_path):
        return None

    df_feat = pd.read_csv(feat_path)
    if "time" not in df_feat.columns:
        return None
    df_feat["time"] = pd.to_datetime(df_feat["time"]).dt.normalize()
    df_feat = df_feat.set_index("time").sort_index()

    df_sent = pd.read_csv(sent_path)
    if "time" not in df_sent.columns:
        return None
    df_sent["time"] = pd.to_datetime(df_sent["time"]).dt.normalize()
    df_sent = df_sent.set_index("time").sort_index()

    df_full = df_feat.join(df_sent, how="left").fillna(0.0)
    return df_full


def refresh_alpha_values_from_existing_formulas(tickers: list[str]) -> dict:
    """Recompute alphas/*.csv from existing formulas + latest features/sentiment."""
    updated = []
    failed = []

    for ticker in tickers:
        try:
            df_full = _build_full_input_for_alpha(ticker)
            if df_full is None or df_full.empty:
                failed.append(ticker)
                log.warning(f"[{ticker}] alpha refresh skipped (missing feature/sentiment inputs)")
                continue

            df_alpha = compute_alpha_for_date(ticker, df_full)
            if df_alpha is None or df_alpha.empty:
                failed.append(ticker)
                log.warning(f"[{ticker}] alpha refresh failed (no computed alpha series)")
                continue

            out_path = os.path.join(ALPHA_VALUES_DIR, f"{ticker}_alpha_values.csv")
            os.makedirs(ALPHA_VALUES_DIR, exist_ok=True)
            out_df = df_alpha.copy()
            out_df.index.name = "time"
            out_df.to_csv(out_path)
            updated.append(ticker)
        except Exception as e:
            failed.append(ticker)
            log.warning(f"[{ticker}] alpha refresh failed: {e}")

    return {"updated": updated, "failed": failed}


# ─── Composite signal per ticker ─────────────────────────────────────────────

def compute_composite(ticker: str, alpha_values: pd.DataFrame) -> pd.Series:
    meta = load_alpha_meta(ticker)
    if meta is None:
        return pd.Series(dtype=float)

    ok_alphas = [a for a in meta["alphas"] if a.get("status") == "OK"]
    scores = [a.get("score", 0.0) for a in ok_alphas]
    ids    = [a["id"] for a in ok_alphas]

    total = sum(scores) or 1.0
    weights = [s / total for s in scores]

    signal = pd.Series(0.0, index=alpha_values.index)
    for i, aid in enumerate(ids):
        col = f"alpha_{aid}"
        if col in alpha_values.columns:
            signal += weights[i] * alpha_values[col].fillna(0.0)

    # Z-score normalize over rolling 60d
    mu  = signal.rolling(60, min_periods=10).mean()
    std = signal.rolling(60, min_periods=10).std()
    return ((signal - mu) / (std + 1e-9)).rename(f"{ticker}_signal")


# ─── Run daily for all tickers ────────────────────────────────────────────────

def run_daily(tickers: list[str]) -> pd.DataFrame:
    """
    Returns DataFrame: index=ticker, cols=[signal, action, strength, ic_avg, ...]
    """
    rows = []

    for ticker in tickers:
        try:
            alpha_values = load_alpha_values(ticker)
            if alpha_values is None or alpha_values.empty:
                log.warning(f"[{ticker}] No alpha values, skipping")
                continue

            composite = compute_composite(ticker, alpha_values)
            if composite.empty:
                continue

            latest_signal = float(composite.iloc[-1]) if not np.isnan(composite.iloc[-1]) else 0.0
            latest_date   = composite.index[-1]

            # Action based on z-score thresholds
            if latest_signal > 1.0:
                action = "BUY"
            elif latest_signal < -1.0:
                action = "SELL"
            else:
                action = "HOLD"

            strength = min(100, int(abs(latest_signal) * 40))

            # Get meta info
            meta = load_alpha_meta(ticker)
            ok_alphas = [a for a in meta["alphas"] if a.get("status") == "OK"] if meta else []
            avg_ic = np.mean([abs(a["ic"]) for a in ok_alphas if a.get("ic")]) if ok_alphas else 0.0
            avg_sharpe = np.mean([a["sharpe"] for a in ok_alphas if a.get("sharpe")]) if ok_alphas else 0.0

            rows.append({
                "ticker":         ticker,
                "date":           latest_date,
                "signal":         round(latest_signal, 4),
                "action":         action,
                "strength":       strength,
                "avg_ic":         round(avg_ic, 4),
                "avg_sharpe":     round(avg_sharpe, 4),
                "n_alphas_ok":    len(ok_alphas),
                "composite":      composite,
            })

        except Exception as e:
            log.error(f"[{ticker}] Daily run error: {e}")

    if not rows:
        return pd.DataFrame()

    # Sort by signal (descending for BUY candidates)
    result = pd.DataFrame([{k: v for k, v in r.items() if k != "composite"} for r in rows])
    result["rank"] = result["signal"].rank(ascending=False).astype(int)
    result = result.sort_values("signal", ascending=False).reset_index(drop=True)

    # Save
    ts = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(SIGNALS_DIR, f"signals_{ts}.csv")
    result.drop(columns=["composite"] if "composite" in result.columns else []).to_csv(out_path, index=False)
    log.info(f"Signals saved → {out_path}")

    return result


# ─── Periodic decay check ─────────────────────────────────────────────────────

def check_all_decay(tickers: list[str]) -> list[dict]:
    """
    Check IC decay for all tickers. Returns list of tickers needing refinement.
    """
    from core.backtester import detect_decay

    needs_refinement = []

    for ticker in tickers:
        try:
            alpha_values = load_alpha_values(ticker)
            meta = load_alpha_meta(ticker)
            if alpha_values is None or meta is None:
                continue

            feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
            if not os.path.exists(feat_path):
                continue
            df_feat = pd.read_csv(feat_path)
            df_feat["time"] = pd.to_datetime(df_feat["time"]).dt.normalize()
            df_feat = df_feat.set_index("time").sort_index()
            fwd_ret = df_feat["close"].pct_change(1).shift(-1)

            decaying_alphas = []
            for a in meta["alphas"]:
                if a.get("status") != "OK":
                    continue
                col = f"alpha_{a['id']}"
                if col not in alpha_values.columns:
                    continue
                decay = detect_decay(alpha_values[col], fwd_ret)
                if decay.get("decaying"):
                    decaying_alphas.append({
                        "alpha_id": a["id"],
                        "decay": decay,
                    })

            if decaying_alphas:
                needs_refinement.append({
                    "ticker": ticker,
                    "decaying_alphas": decaying_alphas,
                    "n_decaying": len(decaying_alphas),
                })

        except Exception as e:
            log.error(f"[{ticker}] Decay check error: {e}")

    return needs_refinement


def parse_tickers(raw: str | None, use_all: bool) -> list[str]:
    if use_all or not raw:
        return VN30_SYMBOLS
    out = [t.strip().upper() for t in raw.split(",") if t.strip()]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily signal runner")
    parser.add_argument("--all", action="store_true", help="Run for all VN30 tickers")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers, e.g. BID,GAS,FPT",
    )
    parser.add_argument(
        "--check-decay",
        action="store_true",
        help="Also run decay check and print tickers needing refinement",
    )
    parser.add_argument(
        "--skip-market-refresh",
        action="store_true",
        help="Skip auto refresh of latest trading day data before running signals",
    )
    parser.add_argument(
        "--skip-sentiment-refresh",
        action="store_true",
        help="Skip appending latest trading day into sentiment_output from daily_scores",
    )
    parser.add_argument(
        "--skip-alpha-refresh",
        action="store_true",
        help="Skip recomputing alphas/*.csv from existing formulas",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = parse_tickers(args.tickers, args.all)
    if not tickers:
        log.warning("No tickers selected")
        return

    log.info(f"Running daily runner for {len(tickers)} tickers")

    target_day = None
    if not args.skip_market_refresh:
        refresh_stats = refresh_latest_market_data(tickers)
        if refresh_stats.get("target_day"):
            target_day = pd.Timestamp(refresh_stats["target_day"]).normalize()
        log.info(
            "Market refresh: target_day=%s, updated=%d, refreshed_only=%d, skipped=%d, failed=%d",
            refresh_stats.get("target_day"),
            len(refresh_stats.get("updated", [])),
            len(refresh_stats.get("refreshed_only", [])),
            len(refresh_stats.get("skipped", [])),
            len(refresh_stats.get("failed", [])),
        )

    if not args.skip_sentiment_refresh:
        sentiment_stats = refresh_latest_sentiment_data(tickers, target_day)
        log.info(
            "Sentiment refresh: crawled=%d, up_to_date=%d, new_news=%d, score_days=%d, merged=%d, skipped=%d, failed=%d",
            sentiment_stats.get("symbols_crawled", 0),
            sentiment_stats.get("symbols_skipped_up_to_date", 0),
            sentiment_stats.get("news_rows_added", 0),
            sentiment_stats.get("score_days_updated", 0),
            len(sentiment_stats.get("updated", [])),
            len(sentiment_stats.get("skipped", [])),
            len(sentiment_stats.get("failed", [])),
        )

    if not args.skip_alpha_refresh:
        alpha_stats = refresh_alpha_values_from_existing_formulas(tickers)
        log.info(
            "Alpha refresh: updated=%d, failed=%d",
            len(alpha_stats.get("updated", [])),
            len(alpha_stats.get("failed", [])),
        )

    result = run_daily(tickers)
    if result.empty:
        log.warning("No daily signals produced")
    else:
        cols = ["ticker", "signal", "action", "rank"]
        log.info("Top signals:\n%s", result[cols].head(10).to_string(index=False))

    if args.check_decay:
        decay = check_all_decay(tickers)
        if not decay:
            log.info("No decaying alphas detected")
        else:
            log.info("Tickers needing refinement: %s", ", ".join(d["ticker"] for d in decay))


if __name__ == "__main__":
    main()
