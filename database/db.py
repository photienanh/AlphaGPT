"""
database/db.py
SQLite database cho Alpha-GPT.
Schema: hypotheses → alphas → backtest_results
"""
import sqlite3
import os
import json
from typing import Optional, List, Dict, Any

DB_PATH = os.environ.get("ALPHAGPT_DB", "alphagpt.db")


def _ensure_columns(conn: sqlite3.Connection, table: str,
                    required_columns: Dict[str, str]) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {r[1] for r in rows}
    for col, col_type in required_columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def init_db(db_path: str = DB_PATH) -> None:
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
        CREATE INDEX IF NOT EXISTS idx_bt_sota ON backtest_results(is_sota);
    """)
    _ensure_columns(conn, "alphas", {
        "return_oos": "REAL", "mdd": "REAL",
    })
    _ensure_columns(conn, "backtest_results", {
        "return_oos": "REAL", "mdd": "REAL",
    })
    conn.commit()
    conn.close()


def get_db(db_path: str = DB_PATH) -> "AlphaGPTDB":
    init_db(db_path)
    return AlphaGPTDB(db_path)


class AlphaGPTDB:
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
                    return_oos, mdd, turnover)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
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
                result.get("ic_is"), result.get("ic_oos"),
                result.get("sharpe_oos"), result.get("return_oos"),
                result.get("mdd"), result.get("turnover"),
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
                ORDER BY b.ic_oos DESC, b.sharpe_oos DESC LIMIT ?
            """, (thread_id, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_all_sota_alphas(self, min_ic_oos: float = 0.03,
                            limit: int = 200) -> List[Dict]:
        """
        Lấy tất cả sota alphas từ mọi run, dùng cho cross-run RAG.
        Lọc theo min_ic_oos để đảm bảo chất lượng.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    a.alpha_id   AS id,
                    a.expression,
                    a.description,
                    a.family,
                    a.thread_id,
                    b.ic_oos,
                    b.sharpe_oos,
                    b.return_oos,
                    b.created_at
                FROM alphas a
                JOIN backtest_results b ON b.alpha_id = a.id
                WHERE b.is_sota = 1
                  AND b.ic_oos >= ?
                  AND a.expression IS NOT NULL
                  AND a.expression != ''
                ORDER BY b.ic_oos DESC
                LIMIT ?
            """, (min_ic_oos, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_alphas_for_hypothesis(self, hypothesis_id: int) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alphas WHERE hypothesis_id=? ORDER BY id",
                (hypothesis_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_backtest_results_for_alpha(self, alpha_db_id: int) -> List[Dict]:
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
                        d["extra"] = json.loads(d["extra_json"])
                    except Exception:
                        pass
                results.append(d)
            return results