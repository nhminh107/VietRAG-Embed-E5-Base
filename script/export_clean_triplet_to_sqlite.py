#!/usr/bin/env python3
"""Create a clean, train-ready SQLite export from PostgreSQL ``triplet``.

The PostgreSQL table is read-only.  Cleaning happens only in the new SQLite
artifact, which is written atomically after integrity checks succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "database" / "triplet_clean.db"
SOURCE_TABLE = "triplet"
BATCH_SIZE = 5_000
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
WHITESPACE_PATTERN = re.compile(r"\s+")
BAD_ENCODING_MARKERS = ("\ufffd", "Ã")
COLUMNS = (
    "data_id",
    "source",
    "title",
    "topic",
    "domain",
    "anchor",
    "positive",
    "hard_negative",
)


def normalize_text(value: Any) -> str:
    """Apply lossless normalization appropriate for retrieval training text."""
    if not isinstance(value, str):
        return ""
    normalized = html.unescape(value)
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = normalized.replace("\u200b", " ").replace("\ufeff", " ")
    normalized = CONTROL_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def clean_row(row: dict[str, Any]) -> tuple[dict[str, str | None] | None, str | None]:
    """Normalize a triplet or return its explicit rejection reason."""
    anchor = normalize_text(row["anchor"])
    positive = normalize_text(row["positive"])
    hard_negative = normalize_text(row["hard_negative"])
    if not anchor or not positive or not hard_negative:
        return None, "blank_field"
    if not 8 <= len(anchor) <= 384:
        return None, "anchor_length"
    if not 80 <= len(positive) <= 2_000 or not 80 <= len(hard_negative) <= 2_000:
        return None, "passage_length"
    if any(marker in value for value in (anchor, positive, hard_negative) for marker in BAD_ENCODING_MARKERS):
        return None, "encoding_artifact"
    folded = (anchor.casefold(), positive.casefold(), hard_negative.casefold())
    if len(set(folded)) != 3:
        return None, "degenerate_triplet"
    return {
        "data_id": str(row["data_id"]),
        "source": normalize_text(row["source"]) or None,
        "title": normalize_text(row["title"]) or None,
        "topic": normalize_text(row["topic"]) or None,
        "domain": normalize_text(row["domain"]) or None,
        "anchor": anchor,
        "positive": positive,
        "hard_negative": hard_negative,
    }, None


def create_schema(connection: sqlite3.Connection) -> None:
    """Create a compact SQLite schema compatible with triplet training code."""
    connection.execute(
        """
        CREATE TABLE triplet (
            data_id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            topic TEXT,
            domain TEXT,
            anchor TEXT NOT NULL,
            positive TEXT NOT NULL,
            hard_negative TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX idx_triplet_source ON triplet(source)")
    connection.execute(
        "CREATE TABLE export_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def stream_source_rows(engine: Any) -> Iterator[dict[str, Any]]:
    """Read source rows without modifying PostgreSQL."""
    statement = text(
        f"""
        SELECT {", ".join(COLUMNS)}
        FROM {SOURCE_TABLE}
        WHERE anchor IS NOT NULL AND positive IS NOT NULL AND hard_negative IS NOT NULL
        ORDER BY source, data_id
        """
    )
    with engine.connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        result = connection.execution_options(stream_results=True).execute(statement)
        for partition in result.mappings().partitions(BATCH_SIZE):
            yield from partition


def validate_source_schema(engine: Any) -> None:
    """Fail early if the PostgreSQL source no longer has the expected schema."""
    with engine.connect() as connection:
        columns = {
            row[0]
            for row in connection.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema() AND table_name = :table_name
                    """
                ),
                {"table_name": SOURCE_TABLE},
            )
        }
    missing = set(COLUMNS).difference(columns)
    if missing:
        raise ValueError(f"Source table is missing columns: {sorted(missing)}")


def export_database(output_path: Path, overwrite: bool) -> Counter[str]:
    """Clean, deduplicate, validate, and atomically publish a SQLite export."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(PROJECT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set.")

    engine = create_engine(database_url, pool_pre_ping=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".tmp.db", dir=output_path.parent
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)
    sqlite_connection: sqlite3.Connection | None = None
    stats: Counter[str] = Counter()
    seen_hashes: set[bytes] = set()
    try:
        validate_source_schema(engine)
        sqlite_connection = sqlite3.connect(temporary_path)
        sqlite_connection.execute("PRAGMA journal_mode = WAL")
        sqlite_connection.execute("PRAGMA synchronous = NORMAL")
        sqlite_connection.execute("PRAGMA temp_store = MEMORY")
        create_schema(sqlite_connection)
        insert_sql = f"INSERT INTO triplet ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})"
        pending: list[tuple[str | None, ...]] = []

        for source_row in stream_source_rows(engine):
            stats["examined"] += 1
            cleaned, reason = clean_row(dict(source_row))
            if cleaned is None:
                stats[f"rejected:{reason}"] += 1
                continue
            payload = "\x1f".join(
                (cleaned["anchor"] or "", cleaned["positive"] or "", cleaned["hard_negative"] or "")
            ).casefold()
            digest = hashlib.sha256(payload.encode("utf-8")).digest()
            if digest in seen_hashes:
                stats["rejected:duplicate_triplet"] += 1
                continue
            seen_hashes.add(digest)
            pending.append(tuple(cleaned[column] for column in COLUMNS))
            stats["exported"] += 1
            if len(pending) >= BATCH_SIZE:
                sqlite_connection.executemany(insert_sql, pending)
                sqlite_connection.commit()
                pending.clear()
            if stats["examined"] % 50_000 == 0:
                print(
                    f"Examined {stats['examined']:,}; exported {stats['exported']:,}",
                    flush=True,
                )
        if pending:
            sqlite_connection.executemany(insert_sql, pending)
            sqlite_connection.commit()

        for key, value in sorted(stats.items()):
            sqlite_connection.execute(
                "INSERT INTO export_metadata (key, value) VALUES (?, ?)", (key, str(value))
            )
        sqlite_connection.commit()
        validate_output(sqlite_connection, stats["exported"])
        sqlite_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        sqlite_connection.execute("PRAGMA journal_mode = DELETE")
        sqlite_connection.close()
        sqlite_connection = None
        os.replace(temporary_path, output_path)
        return stats
    finally:
        if sqlite_connection is not None:
            sqlite_connection.close()
        engine.dispose()
        if temporary_path.exists():
            temporary_path.unlink()


def validate_output(connection: sqlite3.Connection, expected_rows: int) -> None:
    """Verify the exported SQLite database is complete and train-ready."""
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    summary = connection.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT data_id),
               SUM(anchor = '' OR positive = '' OR hard_negative = ''),
               SUM(anchor = positive OR anchor = hard_negative OR positive = hard_negative),
               SUM(instr(anchor || positive || hard_negative, '�') > 0
                   OR instr(anchor || positive || hard_negative, 'Ã') > 0)
        FROM triplet
        """
    ).fetchone()
    if integrity != "ok" or summary != (expected_rows, expected_rows, 0, 0, 0):
        raise RuntimeError(
            f"SQLite validation failed: integrity={integrity}, summary={summary}, "
            f"expected_rows={expected_rows}."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    stats = export_database(args.output.resolve(), args.overwrite)
    print(f"Output: {args.output.resolve()}")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
