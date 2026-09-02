"""Export one PostgreSQL table to a standalone SQLite database file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, func, select


def export_table(
    source_url: str,
    table_name: str,
    output_path: Path,
    batch_size: int,
) -> int:
    """Copy a PostgreSQL table to SQLite in bounded batches and return its row count."""
    source_engine = create_engine(source_url)
    destination_engine = create_engine(f"sqlite:///{output_path}")

    try:
        source_metadata = MetaData()
        source_table = Table(table_name, source_metadata, autoload_with=source_engine)
        destination_metadata = MetaData()
        destination_table = source_table.to_metadata(destination_metadata)
        destination_metadata.create_all(destination_engine)

        exported_rows = 0
        with source_engine.connect().execution_options(stream_results=True) as source_connection:
            result = source_connection.execute(select(source_table))
            for partition in result.mappings().partitions(batch_size):
                rows = list(partition)
                with destination_engine.begin() as destination_connection:
                    destination_connection.execute(destination_table.insert(), rows)
                exported_rows += len(rows)
                print(f"Exported {exported_rows:,} rows", flush=True)

        with destination_engine.connect() as destination_connection:
            destination_count = destination_connection.execute(
                select(func.count()).select_from(destination_table)
            ).scalar_one()
            integrity_check = destination_connection.exec_driver_sql(
                "PRAGMA integrity_check"
            ).scalar_one()

        if destination_count != exported_rows:
            raise RuntimeError(
                f"Row-count mismatch: exported {exported_rows}, found {destination_count}."
            )
        if integrity_check != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity_check}")

        return exported_rows
    finally:
        source_engine.dispose()
        destination_engine.dispose()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PostgreSQL table to a new SQLite .db file."
    )
    parser.add_argument("--table", default="legal", help="PostgreSQL table to export.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("legal.db"),
        help="Destination SQLite file (must not already exist).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Rows copied per transaction.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    load_dotenv()
    source_url = os.getenv("DATABASE_URL")
    if not source_url:
        raise ValueError("DATABASE_URL is not set.")

    output_path = arguments.output.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Destination already exists: {output_path}. Choose another --output path."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = export_table(
        source_url=source_url,
        table_name=arguments.table,
        output_path=output_path,
        batch_size=arguments.batch_size,
    )
    print(f"Completed: {row_count:,} rows written to {output_path}")


if __name__ == "__main__":
    main()
