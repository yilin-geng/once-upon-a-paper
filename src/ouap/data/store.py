"""SQLite storage layer for paper metadata."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from platformdirs import user_cache_dir

APP_NAME = "ouap"

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    bib_key     TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    abstract    TEXT,
    year        INTEGER,
    venue       TEXT NOT NULL,
    authors     TEXT,
    bibtex      TEXT NOT NULL,
    source      TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_venue_year ON papers(venue, year);
"""


def get_db_path() -> Path:
    """Return the default database path (~/.cache/ouap/ouap.db)."""
    cache = Path(user_cache_dir(APP_NAME))
    cache.mkdir(parents=True, exist_ok=True)
    return cache / "ouap.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the database and return a connection."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def upsert_papers(conn: sqlite3.Connection, papers: list[dict]) -> int:
    """Insert or replace papers. Returns count upserted."""
    if not papers:
        return 0
    conn.executemany(
        """INSERT OR REPLACE INTO papers
           (bib_key, title, abstract, year, venue, authors, bibtex, source, fetched_at)
           VALUES (:bib_key, :title, :abstract, :year, :venue, :authors, :bibtex, :source, :fetched_at)""",
        papers,
    )
    conn.commit()
    return len(papers)


def query_papers(
    conn: sqlite3.Connection,
    venues: list[str] | None = None,
    year_min: int = 2016,
    year_max: int = 2026,
) -> list[dict]:
    """Return papers matching venue/year filters that have abstracts."""
    sql = "SELECT * FROM papers WHERE abstract IS NOT NULL AND abstract != '' AND year >= ? AND year <= ?"
    params: list = [year_min, year_max]
    if venues:
        placeholders = ",".join("?" for _ in venues)
        sql += f" AND venue IN ({placeholders})"
        params.extend(venues)
    sql += " ORDER BY venue, year DESC"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_papers_by_keys(conn: sqlite3.Connection, bib_keys: list[str]) -> list[dict]:
    """Retrieve papers by bib_key list (handles SQLite parameter limit)."""
    if not bib_keys:
        return []
    results = []
    # SQLite has a default SQLITE_MAX_VARIABLE_NUMBER of 999
    chunk_size = 900
    for i in range(0, len(bib_keys), chunk_size):
        chunk = bib_keys[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT * FROM papers WHERE bib_key IN ({placeholders})", chunk
        ).fetchall()
        results.extend(dict(r) for r in rows)
    return results


def get_paper_count(
    conn: sqlite3.Connection,
    venues: list[str] | None = None,
    year_min: int = 2016,
    year_max: int = 2026,
) -> dict[str, int]:
    """Count papers per venue matching filters. Returns {venue: count}."""
    sql = """SELECT venue, COUNT(*) as cnt FROM papers
             WHERE abstract IS NOT NULL AND abstract != '' AND year >= ? AND year <= ?"""
    params: list = [year_min, year_max]
    if venues:
        placeholders = ",".join("?" for _ in venues)
        sql += f" AND venue IN ({placeholders})"
        params.extend(venues)
    sql += " GROUP BY venue"
    rows = conn.execute(sql, params).fetchall()
    return {r["venue"]: r["cnt"] for r in rows}


def get_venue_year_range(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """Return {venue: (min_year, max_year)} for cached data."""
    rows = conn.execute(
        "SELECT venue, MIN(year) as ymin, MAX(year) as ymax FROM papers GROUP BY venue"
    ).fetchall()
    return {r["venue"]: (r["ymin"], r["ymax"]) for r in rows}
