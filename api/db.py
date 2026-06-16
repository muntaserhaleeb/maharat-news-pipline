import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "storage" / "maharat_ops.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config_versions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                config_name TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                saved_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                note        TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                command     TEXT    NOT NULL,
                status      TEXT    NOT NULL,
                started_at  TEXT    NOT NULL,
                finished_at TEXT,
                output      TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generation_jobs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id       TEXT    NOT NULL UNIQUE,
                topic        TEXT    NOT NULL,
                mode         TEXT,
                article_type TEXT,
                status       TEXT    NOT NULL DEFAULT 'pending',
                created_at   TEXT    NOT NULL,
                finished_at  TEXT,
                result_json  TEXT,
                error        TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS media_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id        TEXT    NOT NULL UNIQUE,
                folder_name     TEXT    NOT NULL,
                event_name      TEXT    NOT NULL,
                event_date      TEXT,
                base_dir        TEXT    NOT NULL,
                status          TEXT    NOT NULL DEFAULT 'pending',
                image_count     INTEGER DEFAULT 0,
                scores_json     TEXT,
                duplicates_json TEXT,
                hero_filename   TEXT,
                gallery_json    TEXT,
                rejected_json   TEXT,
                metadata_json   TEXT,
                ai_json         TEXT,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # Migrations: add review columns if they don't exist yet
        for ddl in [
            "ALTER TABLE generation_jobs ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending_review'",
            "ALTER TABLE generation_jobs ADD COLUMN draft_json TEXT",
        ]:
            try:
                conn.execute(ddl)
            except Exception:
                pass  # column already exists
    conn.close()
