"""
database/db.py
SQLite database cho Alpha-GPT — thay thế PostgreSQL.
Schema: hypotheses → alphas → backtest_results
"""
import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

DB_PATH = os.environ.get("ALPHAGPT_DB", "alphagpt.db")


def _ensure_columns(conn: sqlite3.Connection, table: str,
                    required_columns: Dict[str, str]) -> None:
    """Add missing columns for existing tables (SQLite-friendly migration)."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {r[1] for r in rows}
    for col, col_type in required_columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def init_db(db_path: str = DB_PATH) -> None:
    """Tạo tables nếu chưa tồn tại."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS hypotheses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id   TEXT NOT NULL,
            trading_idea TEXT NOT NULL,
            hypothesis  TEXT NOT NULL,
            reason      TEXT,
            concise_reason TEXT,
            concise_observation TEXT,
            concise_justification TEXT,
            concise_knowledge TEXT,
            iteration   INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS alphas (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id       TEXT NOT NULL,
            hypothesis_id   INTEGER NOT NULL REFERENCES hypotheses(id),
            alpha_id        TEXT NOT NULL,
            expression      TEXT,
            description     TEXT,
            family          TEXT,
            ic_is           REAL,
            ic_oos          REAL,
            sharpe_oos      REAL,
            return_oos      REAL,
            mdd             REAL,          
            turnover        REAL,
            gp_enhanced     INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS backtest_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id   TEXT NOT NULL,
            alpha_id    INTEGER NOT NULL REFERENCES alphas(id),
            ic_is       REAL,
            ic_oos      REAL,
            sharpe_oos  REAL,
            return_oos  REAL,
            mdd         REAL,
            turnover    REAL,
            is_sota     INTEGER DEFAULT 0,
            extra_json  TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_hyp_thread ON hypotheses(thread_id);
        CREATE INDEX IF NOT EXISTS idx_alpha_thread ON alphas(thread_id);
        CREATE INDEX IF NOT EXISTS idx_alpha_hyp ON alphas(hypothesis_id);
    """)

    # Migration for existing DB files created with older schema.
    _ensure_columns(conn, "alphas", {
        "return_oos": "REAL",
        "mdd": "REAL",
        "gp_enhanced": "INTEGER DEFAULT 0",
    })
    _ensure_columns(conn, "backtest_results", {
        "return_oos": "REAL",
        "mdd": "REAL",
    })

    conn.commit()
    conn.close()


def get_db(db_path: str = DB_PATH) -> "AlphaGPTDB":
    init_db(db_path)
    return AlphaGPTDB(db_path)


class AlphaGPTDB:
    """Thin wrapper quanh sqlite3 connection."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Hypothesis ────────────────────────────────────────────────────

    def save_hypothesis(self, thread_id: str, state_data: Dict[str, Any]) -> int:
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO hypotheses
                    (thread_id, trading_idea, hypothesis, reason,
                     concise_reason, concise_observation,
                     concise_justification, concise_knowledge, iteration)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                thread_id,
                state_data.get("trading_idea", ""),
                state_data.get("hypothesis", ""),
                state_data.get("reason", ""),
                state_data.get("concise_reason", ""),
                state_data.get("concise_observation", ""),
                state_data.get("concise_justification", ""),
                state_data.get("concise_knowledge", ""),
                state_data.get("iteration", 0),
            ))
            return cur.lastrowid

    def get_hypothesis_history(self, thread_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE thread_id=? ORDER BY iteration",
                (thread_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Alpha ─────────────────────────────────────────────────────────

    def save_alpha(self, thread_id: str, hypothesis_id: int,
                   alpha: Dict[str, Any]) -> int:
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO alphas
                    (thread_id, hypothesis_id, alpha_id, expression,
                    description, family, ic_is, ic_oos, sharpe_oos,
                    return_oos, mdd, turnover, gp_enhanced)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                thread_id, hypothesis_id,
                alpha.get("id", ""),
                alpha.get("expression", ""),
                alpha.get("description", ""),
                alpha.get("family", ""),
                alpha.get("ic_is"),
                alpha.get("ic_oos"),
                alpha.get("sharpe_oos"),
                alpha.get("return_oos"),
                alpha.get("mdd"),
                alpha.get("turnover"),
                1 if alpha.get("gp_enhanced") else 0,
            ))
            return cur.lastrowid
    
    def save_backtest(self, thread_id: str, alpha_db_id: int,
                      result: Dict[str, Any], is_sota: bool = False) -> None:
        extra = {k: v for k, v in result.items()
                 if k not in ("ic_is", "ic_oos", "sharpe_oos",
                              "return_oos", "mdd", "turnover", "score")}
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO backtest_results
                    (thread_id, alpha_id, ic_is, ic_oos, sharpe_oos,
                    return_oos, mdd, turnover, is_sota, extra_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                thread_id, alpha_db_id,
                result.get("ic_is"),
                result.get("ic_oos"),
                result.get("sharpe_oos"),
                result.get("return_oos"),
                result.get("mdd"),
                result.get("turnover"),
                1 if is_sota else 0,
                json.dumps(extra),
            ))

    def get_sota_alphas(self, thread_id: str, limit: int = 10) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT a.*, b.ic_oos as b_ic_oos, b.sharpe_oos as b_sharpe,
                       b.return_oos as b_return_oos, b.mdd as b_mdd
                FROM alphas a
                JOIN backtest_results b ON b.alpha_id = a.id
                WHERE a.thread_id=? AND b.is_sota=1
                ORDER BY b.ic_oos DESC, b.sharpe_oos DESC, b.id DESC LIMIT ?
            """, (thread_id, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_alphas_for_hypothesis(self, hypothesis_id: int) -> List[Dict]:
        """Lấy tất cả alphas thuộc một hypothesis."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alphas WHERE hypothesis_id=? ORDER BY id",
                (hypothesis_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_backtest_results_for_alpha(self, alpha_db_id: int) -> List[Dict]:
        """Lấy tất cả backtest results của một alpha."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_results WHERE alpha_id=? ORDER BY id",
                (alpha_db_id,)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                if d.get("extra_json"):
                    try:
                        import json as _json
                        d["extra"] = _json.loads(d["extra_json"])
                    except Exception:
                        pass
                results.append(d)
            return results