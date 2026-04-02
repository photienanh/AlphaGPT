"""
app.py — Alpha-GPT Dashboard Backend (Flask)
Serves REST API consumed by the frontend SPA.
"""
import os
import json
import glob
import logging
import threading
import subprocess
import uuid
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask import render_template
from flask_cors import CORS

from core.universe import TICKER_INDUSTRY, VN30_SYMBOLS
from core.paths import ALPHA_FORMULA_DIR, ALPHA_VALUES_DIR, SIGNALS_DIR, FEATURES_DIR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

GEN_JOBS: dict[str, dict] = {}
GEN_JOBS_LOCK = threading.Lock()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_meta(ticker: str) -> dict | None:
    path = os.path.join(ALPHA_FORMULA_DIR, f"{ticker}_alphas.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_alpha_values(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(ALPHA_VALUES_DIR, f"{ticker}_alpha_values.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    # Legacy alpha files may contain inf/-inf from earlier runs.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def load_close(ticker: str) -> pd.Series | None:
    path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    return df.set_index("time")["close"].sort_index()


def compute_composite(ticker: str, alpha_values: pd.DataFrame) -> pd.Series:
    meta = load_meta(ticker)
    if meta is None:
        return pd.Series(dtype=float)
    ok = [a for a in meta["alphas"] if a.get("status") == "OK"]
    raw_scores = []
    for a in ok:
        s = a.get("score", 0.0)
        raw_scores.append(float(s) if s is not None and np.isfinite(s) else 0.0)
    total = sum(raw_scores) or 1.0
    signal = pd.Series(0.0, index=alpha_values.index)
    for i, a in enumerate(ok):
        col = f"alpha_{a['id']}"
        if col in alpha_values.columns:
            w = raw_scores[i] / total
            clean_col = alpha_values[col].replace([np.inf, -np.inf], np.nan)
            signal += w * clean_col.fillna(0.0)
    mu  = signal.rolling(60, min_periods=10).mean()
    std = signal.rolling(60, min_periods=10).std()
    return ((signal - mu) / (std + 1e-9))


def safe_float(v):
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    if not np.isfinite(fv):
        return None
    return fv


def finite_mean(values) -> float | None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


# ─── API: Overview ────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/overview")
def api_overview():
    """Summary stats for all 30 tickers."""
    rows = []
    available = []

    for ticker in VN30_SYMBOLS:
        meta = load_meta(ticker)
        if meta is None:
            rows.append({
                "ticker": ticker,
                "industry": TICKER_INDUSTRY.get(ticker, "Khác"),
                "status": "no_data",
                "signal": None,
                "action": "—",
                "strength": 0,
                "avg_ic": None,
                "avg_sharpe": None,
                "n_alphas": 0,
            })
            continue

        available.append(ticker)
        ok = [a for a in meta["alphas"] if a.get("status") == "OK"]
        avg_ic = finite_mean(abs(a.get("ic")) for a in ok if a.get("ic") is not None)
        avg_sharpe = finite_mean(a.get("sharpe") for a in ok if a.get("sharpe") is not None)

        alpha_values = load_alpha_values(ticker)
        signal_val = None
        action = "HOLD"
        strength = 0

        if alpha_values is not None and not alpha_values.empty:
            composite = compute_composite(ticker, alpha_values)
            if not composite.empty:
                latest = float(composite.iloc[-1])
                if not np.isnan(latest):
                    signal_val = round(latest, 4)
                    if latest > 1.0:
                        action = "BUY"
                    elif latest < -1.0:
                        action = "SELL"
                    strength = min(100, int(abs(latest) * 40))

        rows.append({
            "ticker":      ticker,
            "industry":    TICKER_INDUSTRY.get(ticker, "Khác"),
            "status":      "ok",
            "signal":      signal_val,
            "action":      action,
            "strength":    strength,
            "avg_ic":      round(float(avg_ic), 4) if avg_ic is not None else None,
            "avg_sharpe":  round(float(avg_sharpe), 4) if avg_sharpe is not None else None,
            "n_alphas":    len(ok),
        })

    rows.sort(key=lambda r: r["signal"] or 0, reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return jsonify({
        "tickers": rows,
        "n_available": len(available),
        "n_total": len(VN30_SYMBOLS),
        "updated_at": datetime.now().isoformat(),
    })


# ─── API: Ticker detail ───────────────────────────────────────────────────────

@app.route("/api/ticker/<ticker>")
def api_ticker(ticker: str):
    ticker = ticker.upper()
    meta = load_meta(ticker)
    if meta is None:
        return jsonify({"error": f"No data for {ticker}"}), 404

    alpha_values = load_alpha_values(ticker)
    close = load_close(ticker)

    # Alpha details
    alphas_out = []
    for a in meta["alphas"]:
        entry = {
            "id":         a["id"],
            "idea":       a.get("idea", ""),
            "expression": a.get("expression", ""),
            "ic":         safe_float(a.get("ic")),
            "ic_5d":      safe_float(a.get("ic_5d")),
            "sharpe":     safe_float(a.get("sharpe")),
            "turnover":   safe_float(a.get("turnover")),
            "score":      safe_float(a.get("score")),
            "status":     a.get("status", "?"),
            "flipped":    a.get("flipped", False),
        }
        alphas_out.append(entry)

    # Rolling IC series (last 200 rows)
    rolling_ic_data = []
    if alpha_values is not None:
        feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
        if os.path.exists(feat_path):
            df_feat = pd.read_csv(feat_path, parse_dates=["time"])
            df_feat["time"] = pd.to_datetime(df_feat["time"]).dt.normalize()
            df_feat = df_feat.set_index("time").sort_index()
            fwd_ret = df_feat["close"].pct_change(1).shift(-1)

            for a in meta["alphas"]:
                if a.get("status") != "OK":
                    continue
                col = f"alpha_{a['id']}"
                if col not in alpha_values.columns:
                    continue
                merged = pd.concat([alpha_values[col], fwd_ret], axis=1).dropna()
                merged.columns = ["a", "r"]
                rolling = merged["a"].rolling(20).corr(merged["r"]).dropna().tail(200)
                rolling_ic_data.append({
                    "alpha_id": a["id"],
                    "dates": [d.strftime("%Y-%m-%d") for d in rolling.index],
                    "values": [round(float(v), 4) for v in rolling.values if np.isfinite(v)],
                })

    # Composite signal (last 200 rows)
    composite_data = []
    if alpha_values is not None:
        composite = compute_composite(ticker, alpha_values).tail(200)
        composite_data = [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
            for d, v in composite.items()
            if np.isfinite(v)
        ]

    # Close price (last 200 rows)
    close_data = []
    if close is not None:
        for d, v in close.tail(200).items():
            if np.isfinite(v):
                close_data.append({"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)})

    # Latest signal
    latest_signal = 0.0
    action = "HOLD"
    if composite_data:
        latest_signal = composite_data[-1]["value"]
        if latest_signal > 1.0:
            action = "BUY"
        elif latest_signal < -1.0:
            action = "SELL"

    return jsonify({
        "ticker":       ticker,
        "industry":     TICKER_INDUSTRY.get(ticker, "Khác"),
        "n_rounds":     meta.get("n_rounds", 1),
        "n_rows":       meta.get("n_rows", 0),
        "alphas":       alphas_out,
        "rolling_ic":   rolling_ic_data,
        "composite":    composite_data,
        "close":        close_data,
        "latest_signal":latest_signal,
        "action":       action,
        "strength":     min(100, int(abs(latest_signal) * 40)),
    })


# ─── API: Alpha values (for chart) ───────────────────────────────────────────

@app.route("/api/ticker/<ticker>/alpha/<int:alpha_id>")
def api_alpha_series(ticker: str, alpha_id: int):
    ticker = ticker.upper()
    alpha_values = load_alpha_values(ticker)
    if alpha_values is None:
        return jsonify({"error": "No alpha values"}), 404

    col = f"alpha_{alpha_id}"
    if col not in alpha_values.columns:
        return jsonify({"error": f"Alpha {alpha_id} not found"}), 404

    series = alpha_values[col].tail(300).dropna()
    return jsonify({
        "dates":  [d.strftime("%Y-%m-%d") for d in series.index],
        "values": [round(float(v), 4) for v in series.values],
    })


# ─── API: Trade signals (ranked list) ────────────────────────────────────────

@app.route("/api/signals")
def api_signals():
    """Return ranked buy/sell signals for today."""
    rows = []
    for ticker in VN30_SYMBOLS:
        alpha_values = load_alpha_values(ticker)
        if alpha_values is None or alpha_values.empty:
            continue
        composite = compute_composite(ticker, alpha_values)
        if composite.empty:
            continue
        latest = float(composite.iloc[-1])
        if not np.isfinite(latest):
            continue
        close = load_close(ticker)
        latest_price = float(close.iloc[-1]) if close is not None and not close.empty else None

        rows.append({
            "ticker":   ticker,
            "industry": TICKER_INDUSTRY.get(ticker, "Khác"),
            "signal":   round(latest, 4),
            "action":   "BUY" if latest > 1.0 else ("SELL" if latest < -1.0 else "HOLD"),
            "strength": min(100, int(abs(latest) * 40)),
            "price":    latest_price,
        })

    rows.sort(key=lambda r: r["signal"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    top_buy  = [r for r in rows if r["action"] == "BUY"][:5]
    top_sell = [r for r in rows if r["action"] == "SELL"][:5]

    return jsonify({
        "all":      rows,
        "top_buy":  top_buy,
        "top_sell": top_sell,
        "date":     datetime.now().strftime("%Y-%m-%d"),
    })


# ─── API: Decay check ─────────────────────────────────────────────────────────

@app.route("/api/decay")
def api_decay():
    """Check which tickers have decaying alphas."""
    from core.backtester import detect_decay
    results = []
    for ticker in VN30_SYMBOLS:
        meta = load_meta(ticker)
        alpha_values = load_alpha_values(ticker)
        if meta is None or alpha_values is None:
            continue
        feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
        if not os.path.exists(feat_path):
            continue
        try:
            df = pd.read_csv(feat_path, parse_dates=["time"])
            df["time"] = pd.to_datetime(df["time"]).dt.normalize()
            df = df.set_index("time").sort_index()
            fwd_ret = df["close"].pct_change(1).shift(-1)
            fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan)
            decaying = []
            for a in meta["alphas"]:
                if a.get("status") != "OK":
                    continue
                col = f"alpha_{a['id']}"
                if col not in alpha_values.columns:
                    continue
                alpha_series = alpha_values[col].replace([np.inf, -np.inf], np.nan)
                d = detect_decay(alpha_series, fwd_ret)
                if d.get("decaying"):
                    decaying.append({"alpha_id": a["id"], **d})
            if decaying:
                results.append({"ticker": ticker, "decaying": decaying})
        except Exception as e:
            log.error(f"[{ticker}] Decay API error: {e}")

    return jsonify({"needs_refinement": results, "count": len(results)})


# ─── API: Gen alpha (trigger) ─────────────────────────────────────────────────

@app.route("/api/gen/<ticker>", methods=["POST"])
def api_gen_alpha(ticker: str):
    """Trigger alpha generation for a ticker (async via thread)."""
    ticker = ticker.upper()
    if ticker not in VN30_SYMBOLS:
        return jsonify({"error": "Unknown ticker"}), 400

    force = request.json.get("force", False) if request.is_json else False
    refine_only = request.json.get("refine_only", False) if request.is_json else False
    job_id = str(uuid.uuid4())

    with GEN_JOBS_LOCK:
        GEN_JOBS[job_id] = {
            "job_id": job_id,
            "ticker": ticker,
            "status": "running",
            "force": force,
            "refine_only": refine_only,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "return_code": None,
            "error": None,
        }

    def _run():
        cmd = [sys.executable, "pipelines/gen_alpha.py", "--ticker", ticker]
        if refine_only:
            cmd.append("--refine-only")
        elif force:
            cmd.append("--force")
        try:
            log.info(f"[GEN_JOB {job_id}] Started ticker={ticker} refine_only={refine_only} force={force}")
            completed = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parent),
                check=False,
            )
            with GEN_JOBS_LOCK:
                GEN_JOBS[job_id]["return_code"] = completed.returncode
                GEN_JOBS[job_id]["finished_at"] = datetime.now().isoformat()
                if completed.returncode == 0:
                    GEN_JOBS[job_id]["status"] = "completed"
                    log.info(f"[GEN_JOB {job_id}] Completed successfully")
                else:
                    GEN_JOBS[job_id]["status"] = "failed"
                    GEN_JOBS[job_id]["error"] = f"pipelines/gen_alpha.py returned non-zero exit code: {completed.returncode}"
                    log.error(f"[GEN_JOB {job_id}] Failed with return code {completed.returncode}")
        except Exception as e:
            with GEN_JOBS_LOCK:
                GEN_JOBS[job_id]["status"] = "failed"
                GEN_JOBS[job_id]["finished_at"] = datetime.now().isoformat()
                GEN_JOBS[job_id]["error"] = str(e)
            log.exception(f"[GEN_JOB {job_id}] Exception while running generation job")

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({
        "status": "started",
        "job_id": job_id,
        "ticker": ticker,
        "force": force,
        "refine_only": refine_only,
    })


@app.route("/api/gen/status/<job_id>")
def api_gen_status(job_id: str):
    with GEN_JOBS_LOCK:
        job = GEN_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)


# ─── API: History ─────────────────────────────────────────────────────────────

@app.route("/api/ticker/<ticker>/history")
def api_history(ticker: str):
    ticker = ticker.upper()
    meta = load_meta(ticker)
    if meta is None:
        return jsonify({"error": "No data"}), 404
    return jsonify({"history": meta.get("history", [])})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
