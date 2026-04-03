"""
app.py — Alpha-GPT Dashboard Backend (Flask)
"""
import os, json, glob, logging, threading, subprocess, uuid, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from core.universe import TICKER_INDUSTRY, VN30_SYMBOLS
from core.paths import ALPHA_FORMULA_DIR, ALPHA_VALUES_DIR, SIGNALS_DIR, FEATURES_DIR
from core.daily_runner import compute_composite

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

GEN_JOBS: dict[str, dict] = {}
GEN_JOBS_LOCK = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────

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
    return df.replace([np.inf, -np.inf], np.nan)


def load_close(ticker: str) -> pd.Series | None:
    path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    return df.set_index("time")["close"].sort_index()


def safe_float(v):
    if v is None:
        return None
    try:
        fv = float(v)
        return fv if np.isfinite(fv) else None
    except Exception:
        return None


def finite_mean(values) -> float | None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else None


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/overview")
def api_overview():
    rows = []
    available = []
    for ticker in VN30_SYMBOLS:
        meta = load_meta(ticker)
        if meta is None:
            rows.append({"ticker": ticker, "industry": TICKER_INDUSTRY.get(ticker,"Khác"),
                         "status": "no_data", "signal": None, "action": "—",
                         "strength": 0, "avg_ic_oos": None, "avg_sharpe_oos": None,
                         "n_alphas": 0})
            continue
        available.append(ticker)
        ok = [a for a in meta["alphas"] if a.get("status") == "OK"]
        # Prefer OOS metrics
        avg_ic  = finite_mean(abs(a.get("ic_oos") or a.get("ic") or 0) for a in ok)
        avg_sh  = finite_mean(a.get("sharpe_oos") or a.get("sharpe") or 0 for a in ok)
        av = load_alpha_values(ticker)
        signal_val, action, strength = None, "HOLD", 0
        if av is not None and not av.empty:
            comp = compute_composite(ticker, av)
            if not comp.empty:
                latest = float(comp.iloc[-1])
                if np.isfinite(latest):
                    signal_val = round(latest, 4)
                    action = "BUY" if latest > 1.0 else ("SELL" if latest < -1.0 else "HOLD")
                    strength = min(100, int(abs(latest) * 40))
        rows.append({
            "ticker":        ticker,
            "industry":      TICKER_INDUSTRY.get(ticker,"Khác"),
            "status":        "ok",
            "signal":        signal_val,
            "action":        action,
            "strength":      strength,
            "avg_ic_oos":    round(float(avg_ic), 4) if avg_ic is not None else None,
            "avg_sharpe_oos":round(float(avg_sh), 4) if avg_sh  is not None else None,
            "n_alphas":      len(ok),
        })
    rows.sort(key=lambda r: r["signal"] or 0, reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return jsonify({"tickers": rows, "n_available": len(available),
                    "n_total": len(VN30_SYMBOLS),
                    "updated_at": datetime.now().isoformat()})


@app.route("/api/ticker/<ticker>")
def api_ticker(ticker: str):
    ticker = ticker.upper()
    meta   = load_meta(ticker)
    if meta is None:
        return jsonify({"error": f"No data for {ticker}"}), 404
    av    = load_alpha_values(ticker)
    close = load_close(ticker)

    alphas_out = []
    for a in meta["alphas"]:
        alphas_out.append({
            "id":         a["id"],
            "family":     a.get("family", "?"),
            "idea":       a.get("idea", ""),
            "hypothesis": a.get("hypothesis", ""),
            "expression": a.get("expression", ""),
            "ic":         safe_float(a.get("ic")),
            "ic_oos":     safe_float(a.get("ic_oos")),
            "ic_5d":      safe_float(a.get("ic_5d")),
            "sharpe":     safe_float(a.get("sharpe")),
            "sharpe_oos": safe_float(a.get("sharpe_oos")),
            "turnover":   safe_float(a.get("turnover")),
            "score":      safe_float(a.get("score")),
            "status":     a.get("status", "?"),
            "flipped":    a.get("flipped", False),
            "gp_enhanced":a.get("gp_enhanced", False),
        })

    # Rolling IC series
    rolling_ic_data = []
    if av is not None:
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
                if col not in av.columns:
                    continue
                merged = pd.concat([av[col], fwd_ret], axis=1).dropna()
                merged.columns = ["a","r"]
                rolling = merged["a"].rolling(20).corr(merged["r"]).dropna().tail(200)
                rolling_ic_data.append({
                    "alpha_id": a["id"],
                    "family":   a.get("family","?"),
                    "dates":    [d.strftime("%Y-%m-%d") for d in rolling.index],
                    "values":   [round(float(v), 4) for v in rolling.values if np.isfinite(v)],
                })

    # Composite signal
    composite_data = []
    if av is not None:
        comp = compute_composite(ticker, av).tail(200)
        composite_data = [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
            for d, v in comp.items() if np.isfinite(v)
        ]

    # Close price
    close_data = []
    if close is not None:
        for d, v in close.tail(200).items():
            if np.isfinite(v):
                close_data.append({"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)})

    latest_signal = composite_data[-1]["value"] if composite_data else 0.0
    action = "BUY" if latest_signal > 1.0 else ("SELL" if latest_signal < -1.0 else "HOLD")

    return jsonify({
        "ticker":        ticker,
        "industry":      TICKER_INDUSTRY.get(ticker,"Khác"),
        "n_rounds":      meta.get("n_rounds", 1),
        "n_rows":        meta.get("n_rows", 0),
        "alphas":        alphas_out,
        "rolling_ic":    rolling_ic_data,
        "composite":     composite_data,
        "close":         close_data,
        "latest_signal": latest_signal,
        "action":        action,
        "strength":      min(100, int(abs(latest_signal) * 40)),
    })


@app.route("/api/signals")
def api_signals():
    rows = []
    for ticker in VN30_SYMBOLS:
        av = load_alpha_values(ticker)
        if av is None or av.empty:
            continue
        comp = compute_composite(ticker, av)
        if comp.empty:
            continue
        latest = float(comp.iloc[-1])
        if not np.isfinite(latest):
            continue
        close = load_close(ticker)
        price = float(close.iloc[-1]) if close is not None and not close.empty else None
        rows.append({
            "ticker":   ticker,
            "industry": TICKER_INDUSTRY.get(ticker,"Khác"),
            "signal":   round(latest, 4),
            "action":   "BUY" if latest > 1.0 else ("SELL" if latest < -1.0 else "HOLD"),
            "strength": min(100, int(abs(latest) * 40)),
            "price":    price,
        })
    rows.sort(key=lambda r: r["signal"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return jsonify({
        "all":      rows,
        "top_buy":  [r for r in rows if r["action"] == "BUY"][:5],
        "top_sell": [r for r in rows if r["action"] == "SELL"][:5],
        "date":     datetime.now().strftime("%Y-%m-%d"),
    })


@app.route("/api/decay")
def api_decay():
    from core.backtester import detect_decay
    results = []
    for ticker in VN30_SYMBOLS:
        meta = load_meta(ticker)
        av   = load_alpha_values(ticker)
        if meta is None or av is None:
            continue
        feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
        if not os.path.exists(feat_path):
            continue
        try:
            df = pd.read_csv(feat_path, parse_dates=["time"])
            df["time"] = pd.to_datetime(df["time"]).dt.normalize()
            df = df.set_index("time").sort_index()
            fwd_ret = df["close"].pct_change(1).shift(-1).replace([np.inf,-np.inf], np.nan)
            decaying = []
            for a in meta["alphas"]:
                if a.get("status") != "OK":
                    continue
                col = f"alpha_{a['id']}"
                if col not in av.columns:
                    continue
                d = detect_decay(av[col].replace([np.inf,-np.inf], np.nan), fwd_ret)
                if d.get("decaying"):
                    decaying.append({"alpha_id": a["id"], **d})
            if decaying:
                results.append({"ticker": ticker, "decaying": decaying})
        except Exception as e:
            log.error(f"[{ticker}] decay API error: {e}")
    return jsonify({"needs_refinement": results, "count": len(results)})


@app.route("/api/memory/stats")
def api_memory_stats():
    """Show RAG memory stats — how many examples stored."""
    from core.alpha_memory import AlphaMemory
    from core.paths import ALPHA_MEMORY_DIR
    mem = AlphaMemory(ALPHA_MEMORY_DIR)
    stats = {"global": mem.stats(), "tickers": {}}
    for ticker in VN30_SYMBOLS:
        s = mem.stats(ticker)
        if s["ticker_count"] > 0:
            stats["tickers"][ticker] = s
    return jsonify(stats)


@app.route("/api/gen/<ticker>", methods=["POST"])
def api_gen_alpha(ticker: str):
    ticker = ticker.upper()
    if ticker not in VN30_SYMBOLS:
        return jsonify({"error": "Unknown ticker"}), 400
    force       = request.json.get("force", False) if request.is_json else False
    refine_only = request.json.get("refine_only", False) if request.is_json else False
    no_gp       = request.json.get("no_gp", False) if request.is_json else False
    job_id = str(uuid.uuid4())
    with GEN_JOBS_LOCK:
        GEN_JOBS[job_id] = {
            "job_id": job_id, "ticker": ticker, "status": "running",
            "started_at": datetime.now().isoformat(), "finished_at": None,
            "return_code": None, "error": None,
        }

    def _run():
        cmd = [sys.executable, "pipelines/gen_alpha.py", "--ticker", ticker]
        if refine_only:
            cmd.append("--refine-only")
        elif force:
            cmd.append("--force")
        if no_gp:
            cmd.append("--no-gp")
        try:
            completed = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), check=False)
            with GEN_JOBS_LOCK:
                GEN_JOBS[job_id]["return_code"]  = completed.returncode
                GEN_JOBS[job_id]["finished_at"]  = datetime.now().isoformat()
                GEN_JOBS[job_id]["status"] = "completed" if completed.returncode == 0 else "failed"
                if completed.returncode != 0:
                    GEN_JOBS[job_id]["error"] = f"Exit code {completed.returncode}"
        except Exception as e:
            with GEN_JOBS_LOCK:
                GEN_JOBS[job_id]["status"] = "failed"
                GEN_JOBS[job_id]["error"]  = str(e)
                GEN_JOBS[job_id]["finished_at"] = datetime.now().isoformat()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started", "job_id": job_id, "ticker": ticker})


@app.route("/api/gen/status/<job_id>")
def api_gen_status(job_id: str):
    with GEN_JOBS_LOCK:
        job = GEN_JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/ticker/<ticker>/history")
def api_history(ticker: str):
    ticker = ticker.upper()
    meta = load_meta(ticker)
    if meta is None:
        return jsonify({"error": "No data"}), 404
    return jsonify({"history": meta.get("history", [])})


if __name__ == "__main__":
    app.run(debug=True, port=5000)