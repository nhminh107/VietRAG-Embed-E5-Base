#!/usr/bin/env python3
"""Load verified English QA/RAG retrieval triplets into PostgreSQL.

The loader is deliberately restricted to train-ready datasets whose records
contain a query, an evidence passage, and a hard-negative passage.  It never
uses evaluation splits and keeps every source revision in a separate table so
the existing Vietnamese ``triplet`` table remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert


PROJECT_DIR = Path(__file__).resolve().parents[1]
TABLE_NAME = "english_rag_qa_triplet"
DEFAULT_SOURCE_ROOT = Path("/tmp/english-rag-qa-sources")
BATCH_SIZE = 2_000


@dataclass(frozen=True)
class SourceSpec:
    """Immutable provenance and parsing details for one HF train source."""

    key: str
    dataset_id: str
    revision: str
    relative_path: Path
    domain: str
    expected_rows: int
    fields: tuple[str, str, str] | None = None


SOURCES = (
    SourceSpec(
        key="msmarco_500k",
        dataset_id="bclavie/msmarco-500k-triplets",
        revision="cb1a85c1261fa7c65f4ea43f94e50f8b467c372f",
        relative_path=Path("msmarco/data/train-00000-of-00001.parquet"),
        domain="web_passage_qa",
        expected_rows=500_000,
        fields=("query", "positive", "negative"),
    ),
    SourceSpec(
        key="news_as2",
        dataset_id="sirCamp/news_as2_pairs_and_triplets",
        revision="ce9fc30c479f40e62adaa79451ab24b97d16a7f3",
        relative_path=Path("news_as2/data/triplets.parquet"),
        domain="news_qa",
        expected_rows=254_649,
    ),
    SourceSpec(
        key="trivia_qa",
        dataset_id="sentence-transformers/trivia-qa-triplet",
        revision="bfe94607eb149a89fe8107e8c8d187977e587a7d",
        relative_path=Path("trivia_qa/triplet/train-00000-of-00001.parquet"),
        domain="open_domain_trivia",
        expected_rows=60_315,
        fields=("anchor", "positive", "negative"),
    ),
)


def build_table(metadata: MetaData) -> Table:
    """Return the isolated, provenance-preserving English training table."""
    return Table(
        TABLE_NAME,
        metadata,
        Column("data_id", Text, primary_key=True),
        Column("source", Text, nullable=False),
        Column("source_revision", Text, nullable=False),
        Column("source_split", Text, nullable=False),
        Column("source_row_index", Integer, nullable=False),
        Column("domain", Text, nullable=False),
        Column("task_type", Text, nullable=False),
        Column("anchor", Text, nullable=False),
        Column("positive", Text, nullable=False),
        Column("hard_negative", Text, nullable=False),
    )


def normalize_text(value: Any) -> str:
    """Conservatively normalize a text value without changing its content."""
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFC", value).strip()


def is_valid_triplet(anchor: str, positive: str, hard_negative: str) -> bool:
    """Reject incomplete or degenerate retrieval supervision."""
    if not (3 <= len(anchor) <= 1_000):
        return False
    if not (20 <= len(positive) <= 12_000 and 20 <= len(hard_negative) <= 12_000):
        return False
    folded = (anchor.casefold(), positive.casefold(), hard_negative.casefold())
    return folded[0] != folded[1] and folded[1] != folded[2]


def extract_triplet(source: SourceSpec, raw_row: dict[str, Any]) -> tuple[str, str, str]:
    """Extract one fully specified retrieval triplet from a source row."""
    if source.fields is not None:
        values = tuple(normalize_text(raw_row.get(field)) for field in source.fields)
    else:
        values_raw = raw_row.get("texts")
        if not isinstance(values_raw, list) or len(values_raw) != 3:
            raise ValueError(f"{source.key}: expected a three-text triplet.")
        values = tuple(normalize_text(value) for value in values_raw)
    anchor, positive, hard_negative = values
    if not is_valid_triplet(anchor, positive, hard_negative):
        raise ValueError(f"{source.key}: incomplete or degenerate triplet.")
    return anchor, positive, hard_negative


def make_row(
    source: SourceSpec, source_row_index: int, raw_row: dict[str, Any]
) -> dict[str, Any]:
    """Attach complete provenance to one source triplet."""
    anchor, positive, hard_negative = extract_triplet(source, raw_row)
    identity = f"{source.dataset_id}\x1f{source.revision}\x1f{source_row_index}"
    return {
        "data_id": f"en_rag_{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
        "source": source.dataset_id,
        "source_revision": source.revision,
        "source_split": "train",
        "source_row_index": source_row_index,
        "domain": source.domain,
        "task_type": "rag_qa_retrieval_triplet",
        "anchor": anchor,
        "positive": positive,
        "hard_negative": hard_negative,
    }


def iter_raw_source_rows(
    source: SourceSpec, source_root: Path
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream raw rows and verify the immutable source-shard cardinality."""
    source_path = source_root / source.relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source shard: {source_path}")
    row_index = 0
    for batch in pq.ParquetFile(source_path).iter_batches(batch_size=BATCH_SIZE):
        for raw_row in batch.to_pylist():
            yield row_index, raw_row
            row_index += 1
    if row_index != source.expected_rows:
        raise RuntimeError(
            f"{source.key}: read {row_index:,} rows; expected {source.expected_rows:,}."
        )


def iter_source_rows(source: SourceSpec, source_root: Path) -> Iterator[dict[str, Any]]:
    """Stream unique, non-degenerate retrieval triplets from a source shard."""
    triplet_hashes: set[bytes] = set()
    for row_index, raw_row in iter_raw_source_rows(source, source_root):
        try:
            row = make_row(source, row_index, raw_row)
        except ValueError:
            continue
        payload = "\x1f".join(
            (row["anchor"], row["positive"], row["hard_negative"])
        ).casefold()
        digest = hashlib.sha256(payload.encode()).digest()
        if digest in triplet_hashes:
            continue
        triplet_hashes.add(digest)
        yield row


def validate_source(source: SourceSpec, source_root: Path) -> int:
    """Count the unique records retained after conservative QA/RAG quality checks."""
    accepted = sum(1 for _ in iter_source_rows(source, source_root))
    if accepted == 0:
        raise RuntimeError(f"{source.key}: no valid retrieval triplets were retained.")
    print(
        f"{source.key}: retained={accepted:,}; "
        f"quality_rejected={source.expected_rows - accepted:,}",
        flush=True,
    )
    return accepted


def batched(rows: Iterator[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    """Group a row stream into bounded database inserts."""
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = []
    if batch:
        yield batch


def load_source(engine: Any, table: Table, source: SourceSpec, source_root: Path) -> tuple[int, int]:
    """Insert a source idempotently and return processed/inserted counts."""
    processed = 0
    inserted = 0
    for batch in batched(iter_source_rows(source, source_root)):
        statement = postgresql_insert(table).values(batch).on_conflict_do_nothing(
            index_elements=[table.c.data_id]
        ).returning(table.c.data_id)
        with engine.begin() as connection:
            inserted += len(connection.execute(statement).scalars().all())
        processed += len(batch)
        if processed % 20_000 == 0 or len(batch) < BATCH_SIZE:
            print(f"{source.key}: processed={processed:,}; inserted={inserted:,}", flush=True)
    return processed, inserted


def validate_database(engine: Any, table: Table, expected: dict[str, int]) -> None:
    """Verify source counts and required fields after the load."""
    with engine.connect() as connection:
        rows = connection.execute(
            select(table.c.source, func.count().label("rows"))
            .where(table.c.source.in_(expected))
            .group_by(table.c.source)
        ).all()
        actual = {row.source: row.rows for row in rows}
        if actual != expected:
            raise RuntimeError(f"Unexpected source counts: {actual}; expected {expected}.")
        invalid = connection.scalar(
            select(func.count())
            .select_from(table)
            .where(
                table.c.source.in_(expected),
                (table.c.anchor == "")
                | (table.c.positive == "")
                | (table.c.hard_negative == "")
                | (table.c.positive == table.c.hard_negative),
            )
        )
    if invalid:
        raise RuntimeError(f"Database contains {invalid:,} invalid retrieval triplets.")


def parse_arguments() -> argparse.Namespace:
    """Parse loader options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not change PostgreSQL.")
    return parser.parse_args()


def main() -> None:
    """Validate source artifacts, then create and load the dedicated table."""
    args = parse_arguments()
    validated = {source.key: validate_source(source, args.source_root) for source in SOURCES}
    total = sum(validated.values())
    print(f"Validated {total:,} complete train triplets: {validated}", flush=True)
    if args.dry_run:
        return

    load_dotenv(PROJECT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set.")
    metadata = MetaData()
    table = build_table(metadata)
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        metadata.create_all(engine, tables=[table], checkfirst=True)
        for source in SOURCES:
            processed, inserted = load_source(engine, table, source, args.source_root)
            if processed != validated[source.key]:
                raise RuntimeError(f"{source.key}: incomplete load.")
            print(f"{source.key}: final inserted={inserted:,}", flush=True)
        validate_database(
            engine,
            table,
            {source.dataset_id: validated[source.key] for source in SOURCES},
        )
        print(f"Validated PostgreSQL table {TABLE_NAME!r} with {total:,} rows.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
