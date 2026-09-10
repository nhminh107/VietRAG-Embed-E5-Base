#!/usr/bin/env python3
"""Create a cleaned, deduplicated English QA/RAG triplet table.

The original ``english_rag_qa_triplet`` table is immutable. This script repairs
text encoding, removes malformed retrieval supervision, and retains the most
lexically challenging negative for each source-local query/positive pair.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from ftfy import fix_text
from sqlalchemy import Column, Float, MetaData, Table, Text, create_engine, func, select, text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_TABLE = "english_rag_qa_triplet"
TARGET_TABLE = "english_rag_qa_triplet_clean"
CLEANING_VERSION = "v1_ftfy_pair_dedup"
BATCH_SIZE = 2_000
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def build_table(metadata: MetaData) -> Table:
    """Define the cleaned table and its source-local pair uniqueness constraint."""
    return Table(
        TARGET_TABLE,
        metadata,
        Column("data_id", Text, primary_key=True),
        Column("pair_hash", Text, nullable=False, unique=True),
        Column("source_data_id", Text, nullable=False),
        Column("source", Text, nullable=False),
        Column("source_revision", Text, nullable=False),
        Column("source_split", Text, nullable=False),
        Column("source_row_index", Text, nullable=False),
        Column("domain", Text, nullable=False),
        Column("task_type", Text, nullable=False),
        Column("cleaning_version", Text, nullable=False),
        Column("hard_negative_score", Float, nullable=False),
        Column("anchor", Text, nullable=False),
        Column("positive", Text, nullable=False),
        Column("hard_negative", Text, nullable=False),
    )


def clean_text(value: Any) -> str:
    """Repair common encoding damage and normalize whitespace conservatively."""
    if not isinstance(value, str):
        return ""
    repaired = fix_text(value, normalization="NFC")
    return WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", repaired)).strip()


def lexical_overlap(query: str, passage: str) -> float:
    """Score how much a candidate negative overlaps the retrieval query."""
    query_tokens = {token for token in TOKEN_PATTERN.findall(query.casefold()) if len(token) > 1}
    if not query_tokens:
        return 0.0
    passage_tokens = set(TOKEN_PATTERN.findall(passage.casefold()))
    return len(query_tokens.intersection(passage_tokens)) / len(query_tokens)


def invalid_reason(anchor: str, positive: str, hard_negative: str) -> str | None:
    """Return an explicit reason when a triplet is unsuitable for RAG training."""
    if not 5 <= len(anchor) <= 700:
        return "anchor_length"
    if not 60 <= len(positive) <= 9_000:
        return "positive_length"
    if not 60 <= len(hard_negative) <= 9_000:
        return "negative_length"
    if not re.search(r"[A-Za-z]", anchor):
        return "anchor_not_english"
    if CONTROL_PATTERN.search(anchor + positive + hard_negative):
        return "control_character"
    folded = (anchor.casefold(), positive.casefold(), hard_negative.casefold())
    if folded[0] == folded[1] or folded[1] == folded[2]:
        return "degenerate_triplet"
    return None


def make_row(source_row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Clean, validate, and attach pair-level provenance to one source record."""
    anchor = clean_text(source_row["anchor"])
    positive = clean_text(source_row["positive"])
    hard_negative = clean_text(source_row["hard_negative"])
    changed = (anchor, positive, hard_negative) != (
        source_row["anchor"],
        source_row["positive"],
        source_row["hard_negative"],
    )
    reason = invalid_reason(anchor, positive, hard_negative)
    if reason:
        return None, reason, changed

    source = str(source_row["source"])
    pair_identity = f"{source}\x1f{anchor.casefold()}\x1f{positive.casefold()}"
    pair_hash = hashlib.sha256(pair_identity.encode()).hexdigest()
    return (
        {
            "data_id": f"en_clean_{pair_hash[:24]}",
            "pair_hash": pair_hash,
            "source_data_id": str(source_row["data_id"]),
            "source": source,
            "source_revision": str(source_row["source_revision"]),
            "source_split": str(source_row["source_split"]),
            "source_row_index": str(source_row["source_row_index"]),
            "domain": str(source_row["domain"]),
            "task_type": str(source_row["task_type"]),
            "cleaning_version": CLEANING_VERSION,
            "hard_negative_score": lexical_overlap(anchor, hard_negative),
            "anchor": anchor,
            "positive": positive,
            "hard_negative": hard_negative,
        },
        None,
        changed,
    )


def iter_source_rows(engine: Any) -> Iterator[dict[str, Any]]:
    """Read source rows in deterministic order without loading the corpus in RAM."""
    statement = text(
        f"""
        SELECT data_id, source, source_revision, source_split, source_row_index,
               domain, task_type, anchor, positive, hard_negative
        FROM {SOURCE_TABLE}
        ORDER BY source, source_row_index
        """
    )
    with engine.connect().execution_options(stream_results=True) as connection:
        yield from connection.execute(statement).mappings()


def batches(rows: Iterator[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    """Group transformed records into bounded PostgreSQL upserts."""
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = []
    if batch:
        yield batch


def upsert_batch(engine: Any, table: Table, rows: list[dict[str, Any]]) -> None:
    """Keep the highest-overlap negative for a unique source query/positive pair."""
    best_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        previous = best_rows.get(str(row["pair_hash"]))
        if previous is None or row["hard_negative_score"] > previous["hard_negative_score"]:
            best_rows[str(row["pair_hash"])] = row

    statement = postgresql_insert(table).values(list(best_rows.values()))
    statement = statement.on_conflict_do_update(
        index_elements=[table.c.pair_hash],
        set_={
            "source_data_id": statement.excluded.source_data_id,
            "source_row_index": statement.excluded.source_row_index,
            "hard_negative_score": statement.excluded.hard_negative_score,
            "hard_negative": statement.excluded.hard_negative,
        },
        where=statement.excluded.hard_negative_score > table.c.hard_negative_score,
    )
    with engine.begin() as connection:
        connection.execute(statement)


def clean_table(engine: Any, table: Table) -> Counter[str]:
    """Stream all source rows, clean them, and populate the target table."""
    stats: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    for source_row in iter_source_rows(engine):
        stats["source_rows"] += 1
        row, reason, changed = make_row(dict(source_row))
        if changed:
            stats["encoding_or_whitespace_repaired"] += 1
        if reason:
            stats[f"rejected_{reason}"] += 1
            continue
        accepted.append(row)
        if len(accepted) == BATCH_SIZE:
            upsert_batch(engine, table, accepted)
            stats["accepted_candidates"] += len(accepted)
            accepted = []
            if stats["source_rows"] % 20_000 == 0:
                print(dict(stats), flush=True)
    if accepted:
        upsert_batch(engine, table, accepted)
        stats["accepted_candidates"] += len(accepted)
    return stats


def validate_output(engine: Any, table: Table) -> list[dict[str, Any]]:
    """Verify all retained rows satisfy the same cleaning constraints."""
    with engine.connect() as connection:
        rows = connection.execute(
            select(table.c.source, func.count().label("rows"))
            .group_by(table.c.source)
            .order_by(table.c.source)
        ).all()
        invalid = connection.scalar(
            select(func.count())
            .select_from(table)
            .where(
                (func.length(table.c.anchor) < 5)
                | (func.length(table.c.anchor) > 700)
                | (func.length(table.c.positive) < 60)
                | (func.length(table.c.hard_negative) < 60)
                | (table.c.positive == table.c.hard_negative)
            )
        )
    if invalid:
        raise RuntimeError(f"Clean table has {invalid:,} invalid rows.")
    return [dict(row._mapping) for row in rows]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    load_dotenv(PROJECT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set.")
    metadata = MetaData()
    table = build_table(metadata)
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        if args.dry_run:
            stats = Counter()
            for source_row in iter_source_rows(engine):
                stats["source_rows"] += 1
                _, reason, changed = make_row(dict(source_row))
                if changed:
                    stats["encoding_or_whitespace_repaired"] += 1
                if reason:
                    stats[f"rejected_{reason}"] += 1
                else:
                    stats["accepted_candidates"] += 1
            print(dict(stats))
            return
        metadata.create_all(engine, tables=[table], checkfirst=True)
        stats = clean_table(engine, table)
        print(dict(stats), flush=True)
        print(validate_output(engine, table), flush=True)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
