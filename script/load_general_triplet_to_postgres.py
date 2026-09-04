"""Load a SQLite general_triplet database into PostgreSQL."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

from database.models import GeneralTriplet
from database.sql_manager import SQL_Manager


# Change only this filename when loading another SQLite export.
SQLITE_FILENAME = "general_triplet.db"
SQLITE_PATH = PROJECT_DIR / "database" / SQLITE_FILENAME

SOURCE_TABLE = "general_triplet"
INSERT_BATCH_SIZE = 1_000
PROGRESS_EVERY = 10_000


def validate_source_table(connection: sqlite3.Connection) -> None:
    """Check that the SQLite file contains the expected source table and columns."""
    columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({SOURCE_TABLE})")
    }
    required_columns = {
        "data_id",
        "source",
        "title",
        "topic",
        "anchor",
        "positive",
        "hard_negative",
    }
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        raise ValueError(
            f"SQLite table {SOURCE_TABLE!r} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def get_source_count(connection: sqlite3.Connection) -> int:
    """Return the number of records in the SQLite source table."""
    return int(
        connection.execute(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]
    )


def fetch_source_batch(
    connection: sqlite3.Connection,
    last_data_id: str | None,
) -> list[dict]:
    """Read one deterministic batch without loading the whole SQLite file."""
    query = f"""
        SELECT data_id, source, title, topic, anchor, positive, hard_negative
        FROM {SOURCE_TABLE}
    """
    parameters: list[object] = []
    if last_data_id is not None:
        query += " WHERE data_id > ?"
        parameters.append(last_data_id)
    query += " ORDER BY data_id LIMIT ?"
    parameters.append(INSERT_BATCH_SIZE)

    rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def insert_batch(sql_manager: SQL_Manager, rows: list[dict]) -> int:
    """Insert one batch and leave existing data_id values unchanged."""
    if not rows:
        return 0

    statement = postgresql_insert(GeneralTriplet).values(rows)
    statement = statement.on_conflict_do_nothing(
        index_elements=["data_id"]
    ).returning(GeneralTriplet.data_id)
    inserted_ids = sql_manager.con.scalars(statement).all()
    return len(inserted_ids)


def main() -> None:
    if not SQLITE_PATH.is_file():
        raise FileNotFoundError(f"SQLite file does not exist: {SQLITE_PATH}")

    source_connection = sqlite3.connect(
        f"file:{SQLITE_PATH}?mode=ro",
        uri=True,
    )
    source_connection.row_factory = sqlite3.Row
    sql_manager = SQL_Manager()

    try:
        validate_source_table(source_connection)
        source_count = get_source_count(source_connection)
        sql_manager.create_general_triplet()

        existing_count = sql_manager.con.scalar(
            select(func.count()).select_from(GeneralTriplet)
        )
        print(f"SQLite source: {SQLITE_PATH}")
        print(f"Source rows: {source_count:,}")
        print(f"Existing PostgreSQL rows: {existing_count:,}")

        processed = 0
        inserted = 0
        last_data_id = None

        while True:
            rows = fetch_source_batch(source_connection, last_data_id)
            if not rows:
                break

            inserted += insert_batch(sql_manager, rows)
            sql_manager.con.commit()
            processed += len(rows)
            last_data_id = rows[-1]["data_id"]

            if processed % PROGRESS_EVERY < len(rows) or processed == source_count:
                print(
                    f"Processed: {processed:,}/{source_count:,}; "
                    f"inserted: {inserted:,}; skipped: {processed - inserted:,}",
                    flush=True,
                )

        if processed != source_count:
            raise RuntimeError(
                f"Processed {processed:,} rows, but SQLite reports {source_count:,}."
            )

        target_count = sql_manager.con.scalar(
            select(func.count()).select_from(GeneralTriplet)
        )
        print(
            f"Finished. Processed: {processed:,}; inserted: {inserted:,}; "
            f"target rows: {target_count:,}"
        )
    except Exception:
        sql_manager.con.rollback()
        raise
    finally:
        sql_manager.close()
        source_connection.close()


if __name__ == "__main__":
    main()
