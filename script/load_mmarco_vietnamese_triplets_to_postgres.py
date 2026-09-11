#!/usr/bin/env python3
"""Load the filtered Vietnamese mMARCO RAG/QA triplets into PostgreSQL.

The source Parquet is streamed in batches and copied to a temporary staging
table.  The final insert is idempotent: existing primary keys are preserved.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
from dotenv import load_dotenv
from sqlalchemy import create_engine, func, select, text


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from database.models import TripletModel  # noqa: E402


SOURCE_FILE = PROJECT_DIR / "data/mmarco-vietnamese-rag-qa/mmarco_vi_rag_qa_600k.parquet"
SOURCE_NAME = "mMARCO Vietnamese filtered 600k"
SOURCE_TOPIC = "mMARCO triples/train"
SOURCE_DOMAIN = "web_passage_qa"
BATCH_SIZE = 5_000


def iter_rows(source_file: Path) -> Iterator[dict[str, str | None]]:
    """Stream valid Parquet rows and attach stable database metadata."""
    parquet_file = pq.ParquetFile(source_file)
    expected_columns = {"query", "positive", "negative"}
    if set(parquet_file.schema_arrow.names) != expected_columns:
        raise ValueError(
            f"Unexpected source columns: {parquet_file.schema_arrow.names}; "
            f"expected {sorted(expected_columns)}."
        )

    row_index = 0
    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["query", "positive", "negative"],
    ):
        for raw_row in batch.to_pylist():
            anchor = raw_row["query"]
            positive = raw_row["positive"]
            hard_negative = raw_row["negative"]
            if not all(
                isinstance(value, str) and value.strip()
                for value in (anchor, positive, hard_negative)
            ):
                raise ValueError(f"Invalid blank field at Parquet row {row_index}.")
            if positive == hard_negative or anchor == positive:
                raise ValueError(f"Degenerate triplet at Parquet row {row_index}.")

            yield {
                "data_id": f"mmarco_vi_{row_index:06d}",
                "source": SOURCE_NAME,
                "title": None,
                "topic": SOURCE_TOPIC,
                "domain": SOURCE_DOMAIN,
                "anchor": anchor,
                "positive": positive,
                "hard_negative": hard_negative,
            }
            row_index += 1


def validate_source(source_file: Path) -> int:
    """Validate the complete source artifact before any database mutation."""
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing source Parquet: {source_file}")
    retained = sum(1 for _ in iter_rows(source_file))
    if retained == 0:
        raise RuntimeError("Source Parquet contains no usable triplets.")
    print(f"Validated source rows: {retained:,}", flush=True)
    return retained


def load_rows(engine: Any, source_file: Path) -> int:
    """Stage and insert all rows, returning the number newly inserted."""
    TripletModel.__table__.create(engine, checkfirst=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TEMPORARY TABLE mmarco_triplet_stage (
                    data_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT,
                    topic TEXT,
                    domain TEXT,
                    anchor TEXT NOT NULL,
                    positive TEXT NOT NULL,
                    hard_negative TEXT NOT NULL
                ) ON COMMIT DROP
                """
            )
        )
        raw_connection = connection.connection.driver_connection
        copy_sql = (
            "COPY mmarco_triplet_stage "
            "(data_id, source, title, topic, domain, anchor, positive, hard_negative) "
            "FROM STDIN"
        )
        with raw_connection.cursor() as cursor:
            with cursor.copy(copy_sql) as copy:
                for row in iter_rows(source_file):
                    copy.write_row(
                        (
                            row["data_id"],
                            row["source"],
                            row["title"],
                            row["topic"],
                            row["domain"],
                            row["anchor"],
                            row["positive"],
                            row["hard_negative"],
                        )
                    )
            cursor.execute(
                """
                INSERT INTO triplet (
                    data_id, source, title, topic, domain, anchor, positive, hard_negative
                )
                SELECT
                    data_id, source, title, topic, domain, anchor, positive, hard_negative
                FROM mmarco_triplet_stage
                ON CONFLICT (data_id) DO NOTHING
                """
            )
            return cursor.rowcount


def validate_database(engine: Any, expected_rows: int) -> None:
    """Confirm complete source cardinality and required fields after loading."""
    statement = select(
        func.count().label("rows"),
        func.count(TripletModel.hard_negative).label("hard_negatives"),
        func.count()
        .filter(
            (func.length(func.trim(TripletModel.anchor)) == 0)
            | (func.length(func.trim(TripletModel.positive)) == 0)
            | (func.length(func.trim(TripletModel.hard_negative)) == 0)
            | (TripletModel.positive == TripletModel.hard_negative)
        )
        .label("invalid"),
    ).where(TripletModel.source == SOURCE_NAME)
    with engine.connect() as connection:
        result = connection.execute(statement).one()
    if result.rows != expected_rows:
        raise RuntimeError(
            f"PostgreSQL has {result.rows:,} mMARCO rows; expected {expected_rows:,}."
        )
    if result.hard_negatives != expected_rows or result.invalid:
        raise RuntimeError(
            "PostgreSQL validation failed: "
            f"hard_negatives={result.hard_negatives:,}, invalid={result.invalid:,}."
        )
    print(
        f"Validated PostgreSQL: rows={result.rows:,}; "
        f"hard_negatives={result.hard_negatives:,}; invalid={result.invalid:,}",
        flush=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, default=SOURCE_FILE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the Parquet only; do not write to PostgreSQL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    expected_rows = validate_source(args.source_file)
    if args.dry_run:
        return

    load_dotenv(PROJECT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set.")
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        inserted_rows = load_rows(engine, args.source_file)
        print(f"Newly inserted rows: {inserted_rows:,}", flush=True)
        validate_database(engine, expected_rows)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
