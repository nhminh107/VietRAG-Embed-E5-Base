"""Export non-Wikipedia rows from PostgreSQL ``general`` to SQLite.

Default output:

    database/general_export.db

Run from the project root:

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate DL_Env
    python script/export_general_non_wikipedia_to_sqlite.py --overwrite
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import text


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from database.sql_manager import SQL_Manager  # noqa: E402


SOURCE_TABLE = "general"
EXCLUDED_SOURCE = "Wikipedia vi"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "database" / "general_export.db"
DEFAULT_BATCH_SIZE = 5_000
COLUMNS = (
    "data_id",
    "source",
    "title",
    "topic",
    "anchor",
    "positive",
    "hard_negative",
)


def validate_postgres_table(connection: Any) -> None:
    """Check that PostgreSQL has the expected source table and columns."""
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
        ).fetchall()
    }
    missing_columns = set(COLUMNS).difference(columns)
    if missing_columns:
        raise ValueError(
            f"PostgreSQL table {SOURCE_TABLE!r} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def get_source_count(connection: Any) -> int:
    """Return the number of rows that will be exported."""
    return int(
        connection.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {SOURCE_TABLE}
                WHERE source <> :excluded_source
                """
            ),
            {"excluded_source": EXCLUDED_SOURCE},
        ).scalar_one()
    )


def fetch_batch(
    connection: Any,
    last_data_id: str | None,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Read one deterministic PostgreSQL batch using keyset pagination."""
    if last_data_id is None:
        query = text(
            f"""
            SELECT {', '.join(COLUMNS)}
            FROM {SOURCE_TABLE}
            WHERE source <> :excluded_source
            ORDER BY data_id
            LIMIT :batch_size
            """
        )
        parameters = {
            "excluded_source": EXCLUDED_SOURCE,
            "batch_size": batch_size,
        }
    else:
        query = text(
            f"""
            SELECT {', '.join(COLUMNS)}
            FROM {SOURCE_TABLE}
            WHERE source <> :excluded_source
              AND data_id > :last_data_id
            ORDER BY data_id
            LIMIT :batch_size
            """
        )
        parameters = {
            "excluded_source": EXCLUDED_SOURCE,
            "last_data_id": last_data_id,
            "batch_size": batch_size,
        }

    return [dict(row) for row in connection.execute(query, parameters).mappings()]


def create_sqlite_table(connection: sqlite3.Connection) -> None:
    """Create the SQLite table with the same columns as PostgreSQL general."""
    connection.execute(
        """
        CREATE TABLE general (
            data_id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            topic TEXT,
            anchor TEXT NOT NULL,
            positive TEXT,
            hard_negative TEXT
        )
        """
    )


def insert_batch(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    """Insert one batch into SQLite."""
    values = [tuple(row[column] for column in COLUMNS) for row in rows]
    connection.executemany(
        """
        INSERT INTO general
            (data_id, source, title, topic, anchor, positive, hard_negative)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )


def validate_sqlite_output(
    connection: sqlite3.Connection,
    expected_count: int,
) -> dict[str, Any]:
    """Validate count, schema, source filter, and required text fields."""
    count = int(connection.execute("SELECT COUNT(*) FROM general").fetchone()[0])
    wiki_rows = int(
        connection.execute(
            "SELECT COUNT(*) FROM general WHERE source = ?",
            (EXCLUDED_SOURCE,),
        ).fetchone()[0]
    )
    invalid_rows = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM general
            WHERE anchor IS NULL OR trim(anchor) = ''
               OR positive IS NULL OR trim(positive) = ''
            """
        ).fetchone()[0]
    )
    if count != expected_count:
        raise RuntimeError(
            f"SQLite has {count:,} rows, expected {expected_count:,}."
        )
    if wiki_rows:
        raise RuntimeError(f"SQLite contains {wiki_rows:,} Wikipedia rows.")
    if invalid_rows:
        raise RuntimeError(f"SQLite contains {invalid_rows:,} invalid rows.")

    sources = connection.execute(
        """
        SELECT source, COUNT(*)
        FROM general
        GROUP BY source
        ORDER BY COUNT(*) DESC
        """
    ).fetchall()
    return {"rows": count, "sources": sources}


def export_database(output_path: Path, batch_size: int, overwrite: bool) -> None:
    """Export PostgreSQL rows to a temporary SQLite file, then replace output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp.db",
        dir=output_path.parent,
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)
    postgres_manager = SQL_Manager()
    sqlite_connection: sqlite3.Connection | None = None

    try:
        with postgres_manager.engine.connect() as postgres_connection:
            validate_postgres_table(postgres_connection)
            expected_count = get_source_count(postgres_connection)
            print(f"PostgreSQL rows to export: {expected_count:,}")

            sqlite_connection = sqlite3.connect(temporary_path)
            sqlite_connection.execute("PRAGMA journal_mode = DELETE")
            sqlite_connection.execute("PRAGMA synchronous = NORMAL")
            create_sqlite_table(sqlite_connection)

            processed = 0
            last_data_id: str | None = None
            while True:
                rows = fetch_batch(postgres_connection, last_data_id, batch_size)
                if not rows:
                    break

                insert_batch(sqlite_connection, rows)
                sqlite_connection.commit()
                processed += len(rows)
                last_data_id = str(rows[-1]["data_id"])
                print(f"Processed: {processed:,}/{expected_count:,}", flush=True)

            if processed != expected_count:
                raise RuntimeError(
                    f"Processed {processed:,} rows, expected {expected_count:,}."
                )

            validation = validate_sqlite_output(sqlite_connection, expected_count)
            print("SQLite validation:", validation)
            sqlite_connection.close()
            sqlite_connection = None

            os.replace(temporary_path, output_path)
            print(f"Exported: {output_path}")
    finally:
        if sqlite_connection is not None:
            sqlite_connection.close()
        if temporary_path.exists():
            temporary_path.unlink()
        postgres_manager.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"SQLite output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows exported per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file after successful validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the PostgreSQL-to-SQLite export."""
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    export_database(args.output, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
