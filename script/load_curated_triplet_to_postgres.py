"""Create and populate the PostgreSQL ``triplet`` table.

The table combines the existing UIT ViQuAD2 rows from ``general`` with the
``nhminh107/VietEmbed-RAG-Science`` Hugging Face training split. Existing
tables are read-only and existing ``triplet`` rows are preserved on reruns.

The UIT source does not provide hard negatives in ``general``; those rows are
therefore inserted with ``hard_negative`` set to NULL. Science rows contain a
complete triplet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from dotenv import load_dotenv
from sqlalchemy import Integer, cast, func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from database.models import GeneralModel, TripletModel  # noqa: E402
from database.sql_manager import SQL_Manager  # noqa: E402


HF_DATASET = "nhminh107/VietEmbed-RAG-Science"
HF_SOURCE = "VietEmbed-RAG-Science"
UIT_SOURCE = "UIT ViQuAD2"
BATCH_SIZE = 1_000
REQUIRED_HF_COLUMNS = (
    "title",
    "topic",
    "anchor",
    "positive",
    "hard_negative",
    "domain",
)


def non_empty(value: Any, field_name: str, row_number: int) -> str:
    """Validate and return a required text field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"HF row {row_number} has an empty/non-string {field_name!r} field."
        )
    return value


def chunks(items: Iterable[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield bounded batches from an iterable."""
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def insert_rows(sql_manager: SQL_Manager, rows: list[dict[str, Any]]) -> int:
    """Insert rows idempotently and return the number accepted by PostgreSQL."""
    if not rows:
        return 0
    statement = postgresql_insert(TripletModel).values(rows)
    statement = statement.on_conflict_do_nothing(
        index_elements=[TripletModel.data_id]
    ).returning(TripletModel.data_id)
    inserted_ids = sql_manager.con.scalars(statement).all()
    sql_manager.con.commit()
    return len(inserted_ids)


def read_uit_rows(sql_manager: SQL_Manager) -> Iterable[dict[str, Any]]:
    """Read answerable UIT ViQuAD2 pairs from the existing General table."""
    statement = (
        select(
            GeneralModel.data_id,
            GeneralModel.title,
            GeneralModel.topic,
            GeneralModel.anchor,
            GeneralModel.positive,
        )
        .where(
            GeneralModel.source == UIT_SOURCE,
            GeneralModel.anchor.is_not(None),
            GeneralModel.positive.is_not(None),
            func.length(func.trim(GeneralModel.anchor)) > 0,
            func.length(func.trim(GeneralModel.positive)) > 0,
        )
        .order_by(GeneralModel.data_id)
    )
    for row in sql_manager.con.execute(statement):
        yield {
            "data_id": f"uit_viquad2_{row.data_id}",
            "source": UIT_SOURCE,
            "title": row.title,
            "topic": row.topic,
            "domain": None,
            "anchor": row.anchor,
            "positive": row.positive,
            "hard_negative": None,
        }


def read_hf_rows(cache_dir: Path) -> Iterable[dict[str, Any]]:
    """Stream and validate the Hugging Face Science split."""
    dataset = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        cache_dir=str(cache_dir),
    )
    missing_columns = set(REQUIRED_HF_COLUMNS).difference(dataset.column_names)
    if missing_columns:
        raise ValueError(f"HF dataset is missing columns: {sorted(missing_columns)}")

    for row_number, row in enumerate(dataset, start=1):
        yield {
            "data_id": f"science_{row_number - 1:08d}",
            "source": HF_SOURCE,
            "title": non_empty(row["title"], "title", row_number),
            "topic": non_empty(row["topic"], "topic", row_number),
            "domain": non_empty(row["domain"], "domain", row_number),
            "anchor": non_empty(row["anchor"], "anchor", row_number),
            "positive": non_empty(row["positive"], "positive", row_number),
            "hard_negative": non_empty(
                row["hard_negative"], "hard_negative", row_number
            ),
        }


def load_source(
    sql_manager: SQL_Manager,
    rows: Iterable[dict[str, Any]],
    source_name: str,
) -> tuple[int, int]:
    """Insert one source and return (processed, inserted)."""
    processed = 0
    inserted = 0
    for batch in chunks(rows, BATCH_SIZE):
        processed += len(batch)
        inserted += insert_rows(sql_manager, batch)
        if processed % (BATCH_SIZE * 10) == 0 or len(batch) < BATCH_SIZE:
            print(
                f"{source_name}: processed {processed:,}; inserted {inserted:,}",
                flush=True,
            )
    return processed, inserted


def validate_output(sql_manager: SQL_Manager, expected_uit: int, expected_hf: int) -> None:
    """Validate source counts and required fields after loading."""
    rows = sql_manager.con.execute(
        select(
            TripletModel.source,
            func.count().label("rows"),
            func.count(TripletModel.hard_negative).label("with_hard_negative"),
            func.sum(
                cast(
                    (TripletModel.anchor.is_(None))
                    | (func.length(func.trim(TripletModel.anchor)) == 0),
                    Integer,
                )
            ).label("empty_anchor"),
            func.sum(
                cast(
                    (TripletModel.positive.is_(None))
                    | (func.length(func.trim(TripletModel.positive)) == 0),
                    Integer,
                )
            ).label("empty_positive"),
        )
        .group_by(TripletModel.source)
        .order_by(TripletModel.source)
    ).all()
    stats = {row.source: row for row in rows}
    expected = {
        UIT_SOURCE: (expected_uit, 0),
        HF_SOURCE: (expected_hf, expected_hf),
    }
    for source, (expected_rows, expected_hard_negatives) in expected.items():
        row = stats.get(source)
        if row is None or row.rows != expected_rows:
            raise RuntimeError(
                f"{source}: found {row.rows if row else 0:,} rows; "
                f"expected {expected_rows:,}."
            )
        if row.with_hard_negative != expected_hard_negatives:
            raise RuntimeError(
                f"{source}: found {row.with_hard_negative:,} hard negatives; "
                f"expected {expected_hard_negatives:,}."
            )
        if row.empty_anchor or row.empty_positive:
            raise RuntimeError(f"{source}: blank required fields detected.")

    total = sql_manager.con.scalar(select(func.count()).select_from(TripletModel))
    expected_total = expected_uit + expected_hf
    if total != expected_total:
        raise RuntimeError(f"Total rows: {total:,}; expected {expected_total:,}.")
    print(f"Validated triplet rows: {total:,}")
    for source in sorted(expected):
        row = stats[source]
        print(
            f"{source}: rows={row.rows:,}; "
            f"with_hard_negative={row.with_hard_negative:,}"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/vietembed-rag-hf-cache"),
        help="Temporary Hugging Face cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the table, load both sources, and validate the result."""
    arguments = parse_arguments()
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("DATABASE_URL"):
        raise ValueError("DATABASE_URL is not set.")

    sql_manager = SQL_Manager()
    try:
        sql_manager.create_triplet()
        uit_rows, uit_inserted = load_source(
            sql_manager, read_uit_rows(sql_manager), UIT_SOURCE
        )
        hf_rows, hf_inserted = load_source(
            sql_manager, read_hf_rows(arguments.cache_dir), HF_SOURCE
        )
        print(
            f"Load complete: UIT processed={uit_rows:,}, inserted={uit_inserted:,}; "
            f"Science processed={hf_rows:,}, inserted={hf_inserted:,}",
            flush=True,
        )
        validate_output(sql_manager, expected_uit=uit_rows, expected_hf=hf_rows)
    finally:
        sql_manager.close()


if __name__ == "__main__":
    main()
