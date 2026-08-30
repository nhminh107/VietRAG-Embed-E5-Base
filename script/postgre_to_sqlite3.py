import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


BATCH_SIZE = 10_000
DESTINATION = Path("database/general_export.db")
TEMPORARY_DESTINATION = DESTINATION.with_suffix(".db.tmp")


def write_batch(connection: sqlite3.Connection, rows: list[tuple[object, ...]]) -> None:
    connection.executemany(
        """
        INSERT INTO general (
            data_id, source, title, topic, anchor, positive, hard_negative
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()


def main() -> None:
    load_dotenv(Path.cwd() / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set.")
    if DESTINATION.exists() or TEMPORARY_DESTINATION.exists():
        raise FileExistsError(
            f"Refusing to overwrite {DESTINATION} or {TEMPORARY_DESTINATION}."
        )

    source_engine = create_engine(database_url)
    destination_connection = sqlite3.connect(TEMPORARY_DESTINATION)
    destination_connection.execute("PRAGMA journal_mode = WAL")
    destination_connection.execute("PRAGMA synchronous = NORMAL")
    destination_connection.execute(
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

    exported_rows = 0
    try:
        with source_engine.connect() as source_connection:
            result = source_connection.execution_options(stream_results=True).execute(
                text(
                    """
                    SELECT data_id, source, title, topic, anchor, positive, hard_negative
                    FROM general
                    """
                )
            )
            batch: list[tuple[object, ...]] = []
            for row in result.mappings():
                batch.append(
                    (
                        row["data_id"],
                        row["source"],
                        row["title"],
                        row["topic"],
                        row["anchor"],
                        row["positive"],
                        row["hard_negative"],
                    )
                )
                if len(batch) == BATCH_SIZE:
                    write_batch(destination_connection, batch)
                    exported_rows += len(batch)
                    batch.clear()
                    print(f"Exported: {exported_rows:,}", flush=True)

            if batch:
                write_batch(destination_connection, batch)
                exported_rows += len(batch)

        destination_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        destination_connection.execute("PRAGMA journal_mode = DELETE")
        destination_connection.close()

        verification_connection = sqlite3.connect(TEMPORARY_DESTINATION)
        exported_count = verification_connection.execute(
            "SELECT COUNT(*) FROM general"
        ).fetchone()[0]
        integrity = verification_connection.execute("PRAGMA integrity_check").fetchone()[0]
        verification_connection.close()
        if exported_count != exported_rows or integrity != "ok":
            raise RuntimeError(
                f"Export verification failed: rows={exported_count}, integrity={integrity}."
            )

        TEMPORARY_DESTINATION.rename(DESTINATION)
        print(f"Export complete: {DESTINATION.resolve()}")
        print(f"Rows: {exported_count:,}; integrity: {integrity}")
    finally:
        source_engine.dispose()
        if destination_connection:
            destination_connection.close()


if __name__ == "__main__":
    main()