"""Export the highest-quality, diversity-preserving general triplets to SQLite.

The source PostgreSQL table is never modified. Selection prioritizes one clean,
early chunk per anchor before filling the remaining quota with clean early
chunks. Exact duplicate triplets are removed deterministically.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "database" / "general_triplet_top500k.db"
DEFAULT_LIMIT = 500_000
DEFAULT_BATCH_SIZE = 5_000
SOURCE_TABLE = "general_triplet"
COLUMNS = (
    "data_id",
    "source",
    "title",
    "topic",
    "anchor",
    "positive",
    "hard_negative",
)

FORMULA_PATTERN = re.compile(r"formula_\d+", flags=re.IGNORECASE)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


SELECTION_QUERY = text(
    f"""
    WITH scored AS (
        SELECT
            {", ".join(COLUMNS)},
            CASE
                WHEN data_id ~ '^data_clean_[0-9]{{7}}_[0-9]{{4}}$'
                THEN right(data_id, 4)::integer
                ELSE 1000000
            END AS chunk_index,
            (
                CASE WHEN position('�' in anchor || positive || hard_negative) > 0
                    THEN 1 ELSE 0 END
                + CASE WHEN position('[[' in anchor || positive || hard_negative) > 0
                    THEN 1 ELSE 0 END
                + CASE WHEN position('{{{{' in anchor || positive || hard_negative) > 0
                    THEN 1 ELSE 0 END
                + CASE WHEN (anchor || positive || hard_negative)
                    ~ '<[^>]+>|formula_[0-9]'
                    THEN 1 ELSE 0 END
                + CASE WHEN length(btrim(positive)) NOT BETWEEN 80 AND 384
                    THEN 1 ELSE 0 END
            ) AS quality_issues,
            CASE
                WHEN title IS NOT NULL
                    AND btrim(title) <> ''
                    AND position(lower(btrim(title)) in lower(positive)) > 0
                THEN 1 ELSE 0
            END AS mentions_title,
            CASE WHEN btrim(positive) ~ '[.!?…]$' THEN 1 ELSE 0 END
                AS ends_sentence,
            md5(anchor || chr(30) || positive || chr(30) || hard_negative)
                AS triplet_hash
        FROM {SOURCE_TABLE}
        WHERE anchor IS NOT NULL AND btrim(anchor) <> ''
          AND positive IS NOT NULL AND btrim(positive) <> ''
          AND hard_negative IS NOT NULL AND btrim(hard_negative) <> ''
          AND anchor <> positive
          AND anchor <> hard_negative
          AND positive <> hard_negative
    ),
    deduplicated AS (
        SELECT *, row_number() OVER (
            PARTITION BY triplet_hash
            ORDER BY quality_issues, chunk_index, data_id
        ) AS duplicate_rank
        FROM scored
    ),
    per_anchor AS (
        SELECT *, row_number() OVER (
            PARTITION BY anchor
            ORDER BY
                quality_issues,
                chunk_index,
                mentions_title DESC,
                ends_sentence DESC,
                md5(data_id)
        ) AS anchor_rank
        FROM deduplicated
        WHERE duplicate_rank = 1
    )
    SELECT {", ".join(COLUMNS)}
    FROM per_anchor
    ORDER BY
        CASE WHEN anchor_rank = 1 THEN 0 ELSE 1 END,
        CASE WHEN anchor_rank = 1 THEN 0 ELSE quality_issues END,
        anchor_rank,
        chunk_index,
        mentions_title DESC,
        ends_sentence DESC,
        md5(data_id)
    LIMIT :row_limit
    """
)


def validate_source(connection: Any, row_limit: int) -> int:
    """Validate the source schema and return its distinct anchor count."""
    columns = {
        row[0]
        for row in connection.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = :table_name
                """
            ),
            {"table_name": SOURCE_TABLE},
        )
    }
    missing_columns = set(COLUMNS).difference(columns)
    if missing_columns:
        raise ValueError(
            f"PostgreSQL table {SOURCE_TABLE!r} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    stats = connection.execute(
        text(
            f"""
            SELECT COUNT(*) AS rows, COUNT(DISTINCT anchor) AS anchors
            FROM {SOURCE_TABLE}
            WHERE anchor IS NOT NULL AND btrim(anchor) <> ''
              AND positive IS NOT NULL AND btrim(positive) <> ''
              AND hard_negative IS NOT NULL AND btrim(hard_negative) <> ''
              AND anchor <> positive
              AND anchor <> hard_negative
              AND positive <> hard_negative
            """
        )
    ).one()
    eligible_rows = int(stats.rows)
    distinct_anchors = int(stats.anchors)
    if eligible_rows < row_limit:
        raise ValueError(
            f"Only {eligible_rows:,} eligible rows are available; "
            f"cannot export {row_limit:,}."
        )
    return min(distinct_anchors, row_limit)


def create_sqlite_table(connection: sqlite3.Connection) -> None:
    """Create the notebook-compatible output table."""
    connection.execute(
        """
        CREATE TABLE general_triplet (
            data_id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            topic TEXT,
            anchor TEXT NOT NULL,
            positive TEXT NOT NULL,
            hard_negative TEXT NOT NULL
        )
        """
    )


def clean_residual_artifacts(value: str) -> str:
    """Remove rare broken wiki markers without rewriting normal text."""
    if not (
        "�" in value
        or "[[" in value
        or "]]" in value
        or "{{" in value
        or "}}" in value
        or FORMULA_PATTERN.search(value)
        or CONTROL_PATTERN.search(value)
    ):
        return value

    cleaned = value.replace("�", " ")
    for marker in ("[[", "]]", "{{", "}}"):
        cleaned = cleaned.replace(marker, " ")
    cleaned = FORMULA_PATTERN.sub(" ", cleaned)
    cleaned = CONTROL_PATTERN.sub(" ", cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("Artifact cleanup produced an empty training field.")
    return cleaned


def validate_output(
    connection: sqlite3.Connection,
    expected_rows: int,
    expected_anchors: int,
) -> None:
    """Validate row count, coverage, required fields, and file integrity."""
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    stats = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT data_id) AS ids,
            COUNT(DISTINCT anchor) AS anchors,
            SUM(
                length(trim(anchor)) = 0
                OR length(trim(positive)) = 0
                OR length(trim(hard_negative)) = 0
            ) AS empty_fields,
            SUM(
                anchor = positive
                OR anchor = hard_negative
                OR positive = hard_negative
            ) AS equal_fields,
            MAX(length(positive)) AS max_positive_chars
        FROM general_triplet
        """
    ).fetchone()

    checks = {
        "integrity": integrity == "ok",
        "row_count": stats[0] == expected_rows,
        "unique_ids": stats[1] == expected_rows,
        "all_anchors_preserved": stats[2] == expected_anchors,
        "non_empty_fields": stats[3] == 0,
        "distinct_triplet_fields": stats[4] == 0,
        "positive_within_384_chars": stats[5] <= 384,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"SQLite validation failed: {failed}")

    duplicate_triplets = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT 1
            FROM general_triplet
            GROUP BY anchor, positive, hard_negative
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicate_triplets:
        raise RuntimeError(
            f"SQLite contains {duplicate_triplets:,} duplicate triplets."
        )


def export_database(
    source_url: str,
    output_path: Path,
    row_limit: int,
    batch_size: int,
    overwrite: bool,
) -> tuple[int, int]:
    """Select and stream PostgreSQL rows into an atomic SQLite output."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp.db",
        dir=output_path.parent,
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)

    engine = create_engine(source_url, connect_args={"connect_timeout": 15})
    sqlite_connection: sqlite3.Connection | None = None
    try:
        with engine.connect() as source:
            source.execute(text("SET statement_timeout = 0"))
            expected_anchors = validate_source(source, row_limit)
            print(f"Distinct eligible anchors: {expected_anchors:,}", flush=True)

            sqlite_connection = sqlite3.connect(temporary_path)
            sqlite_connection.execute("PRAGMA journal_mode = WAL")
            sqlite_connection.execute("PRAGMA synchronous = NORMAL")
            sqlite_connection.execute("PRAGMA temp_store = MEMORY")
            create_sqlite_table(sqlite_connection)

            insert_sql = f"""
                INSERT INTO general_triplet ({", ".join(COLUMNS)})
                VALUES ({", ".join("?" for _ in COLUMNS)})
            """
            result = source.execution_options(stream_results=True).execute(
                SELECTION_QUERY,
                {"row_limit": row_limit},
            )
            exported_rows = 0
            for partition in result.mappings().partitions(batch_size):
                values = []
                for row in partition:
                    values.append(
                        tuple(
                            clean_residual_artifacts(str(row[column]))
                            if column in {"anchor", "positive", "hard_negative"}
                            else row[column]
                            for column in COLUMNS
                        )
                    )
                sqlite_connection.executemany(insert_sql, values)
                sqlite_connection.commit()
                exported_rows += len(values)
                if exported_rows % 50_000 == 0 or exported_rows == row_limit:
                    print(f"Exported {exported_rows:,} rows", flush=True)

            validate_output(
                sqlite_connection,
                expected_rows=row_limit,
                expected_anchors=expected_anchors,
            )
            sqlite_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            sqlite_connection.execute("PRAGMA journal_mode = DELETE")
            sqlite_connection.close()
            sqlite_connection = None

        os.replace(temporary_path, output_path)
        return exported_rows, expected_anchors
    finally:
        if sqlite_connection is not None:
            sqlite_connection.close()
        engine.dispose()
        if temporary_path.exists():
            temporary_path.unlink()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the quality-preserving PostgreSQL-to-SQLite export."""
    arguments = parse_arguments()
    if arguments.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    load_dotenv(PROJECT_DIR / ".env")
    source_url = os.getenv("DATABASE_URL")
    if not source_url:
        raise ValueError("DATABASE_URL is not set.")

    output_path = arguments.output.resolve()
    rows, anchors = export_database(
        source_url=source_url,
        output_path=output_path,
        row_limit=arguments.limit,
        batch_size=arguments.batch_size,
        overwrite=arguments.overwrite,
    )
    print(f"Completed: {rows:,} rows", flush=True)
    print(f"Unique anchors: {anchors:,}", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
