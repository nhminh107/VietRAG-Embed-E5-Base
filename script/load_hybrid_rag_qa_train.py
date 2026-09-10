#!/usr/bin/env python3
"""Create a provenance-safe 300k hybrid training table for Vietnamese RAG/QA.

This loader reads only established training sources already stored in PostgreSQL.
VN-MTEB is deliberately excluded: its ``dev`` and ``test`` splits are benchmark
data, not training data.  QA, synthetic QA, and semantic-paraphrase examples
are stored under distinct task types so downstream training can weight or
exclude each category explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, Integer, MetaData, Table, Text, create_engine, func, select, text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert


PROJECT_DIR = Path(__file__).resolve().parents[1]
TABLE_NAME = "rag_qa_hybrid_train"
BATCH_SIZE = 1_000
WHITESPACE_PATTERN = re.compile(r"\s+")
ARTIFACT_MARKERS = ("\ufffd", "[[", "]]", "{{", "}}")


@dataclass(frozen=True)
class SourceSpec:
    """A source-specific quota and its honest training-task annotation."""

    source_table: str
    source: str
    task_type: str
    is_synthetic: bool
    quality_score: int
    quota: int


SPECS = (
    SourceSpec("legal", "Law Reading Comprehension QA", "rag_qa_gold", False, 100, 180_000),
    SourceSpec("legal", "Zalo AI Legal", "rag_qa_gold", False, 100, 1_000),
    SourceSpec("general", "UIT ViQuAD2", "rag_qa_gold", False, 100, 19_000),
    SourceSpec("general", "Wikipedia vi", "rag_qa_synthetic", True, 70, 50_000),
    SourceSpec("general", "ViNewsQA", "semantic_paraphrase", False, 85, 20_000),
    SourceSpec("general", "ViQuAD", "semantic_paraphrase", False, 85, 15_000),
    SourceSpec("general", "ViNLI", "semantic_paraphrase", False, 80, 10_000),
    SourceSpec("general", "ALQAC", "semantic_paraphrase", False, 80, 5_000),
)


def build_table(metadata: MetaData) -> Table:
    """Define the train-only table with source-level provenance."""
    return Table(
        TABLE_NAME,
        metadata,
        Column("data_id", Text, primary_key=True),
        Column("split", Text, nullable=False),
        Column("task_type", Text, nullable=False),
        Column("is_synthetic", Boolean, nullable=False),
        Column("quality_score", Integer, nullable=False),
        Column("source_table", Text, nullable=False),
        Column("source", Text, nullable=False),
        Column("source_data_id", Text, nullable=False),
        Column("title", Text, nullable=True),
        Column("anchor", Text, nullable=False),
        Column("positive", Text, nullable=False),
        Column("hard_negative", Text, nullable=True),
    )


def normalize_text(value: Any) -> str:
    """Normalize Unicode and whitespace without rewriting source meaning."""
    if not isinstance(value, str):
        return ""
    return WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", value)).strip()


def is_clean_pair(anchor: str, positive: str) -> bool:
    """Apply conservative text hygiene shared by every selected source."""
    if not 15 <= len(anchor) <= 420 or not 80 <= len(positive) <= 1_400:
        return False
    if anchor.casefold() == positive.casefold():
        return False
    combined = f"{anchor}\n{positive}"
    return not any(marker in combined for marker in ARTIFACT_MARKERS)


def stable_id(spec: SourceSpec, source_data_id: str) -> str:
    """Create a reproducible destination ID without reusing a source key."""
    payload = f"{spec.source_table}\x1f{spec.source}\x1f{source_data_id}"
    return f"hybrid_{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


LEGAL_SELECTION = text(
    """
    WITH candidates AS (
        SELECT data_id, source, title, anchor, positive,
               CASE WHEN position(lower(btrim(title)) IN lower(positive)) > 0
                    THEN 1 ELSE 0 END AS title_in_positive,
               CASE WHEN btrim(positive) ~ '[.!…][[:space:]]*$' THEN 1 ELSE 0 END
                    AS ends_sentence,
               md5(data_id) AS stable_order
        FROM public.legal
        WHERE source = :source
          AND title IS NOT NULL AND btrim(title) <> ''
          AND anchor IS NOT NULL AND btrim(anchor) <> ''
          AND positive IS NOT NULL AND btrim(positive) <> ''
          AND length(btrim(regexp_replace(anchor, '[[:space:]]+', ' ', 'g')))
              BETWEEN 20 AND 420
          AND length(btrim(regexp_replace(positive, '[[:space:]]+', ' ', 'g')))
              BETWEEN 120 AND 900
          AND anchor <> positive
          AND position('�' IN anchor || positive) = 0
          AND position('[[' IN anchor || positive) = 0
          AND position('{{' IN anchor || positive) = 0
    ),
    pair_deduplicated AS (
        SELECT *, row_number() OVER (
            PARTITION BY anchor, positive
            ORDER BY title_in_positive DESC, ends_sentence DESC, stable_order
        ) AS pair_rank
        FROM candidates
    ),
    title_deduplicated AS (
        SELECT *, row_number() OVER (
            PARTITION BY lower(btrim(regexp_replace(title, '[[:space:]]+', ' ', 'g')))
            ORDER BY title_in_positive DESC, ends_sentence DESC,
                     abs(length(positive) - 450), stable_order
        ) AS title_rank
        FROM pair_deduplicated
        WHERE pair_rank = 1
    )
    SELECT data_id, source, title, anchor, positive
    FROM title_deduplicated
    WHERE title_rank = 1
    ORDER BY title_in_positive DESC, ends_sentence DESC,
             abs(length(positive) - 450), stable_order
    LIMIT :row_limit
    """
)


GENERAL_SELECTION = text(
    """
    WITH candidates AS (
        SELECT data_id, source, title, anchor, positive, md5(data_id) AS stable_order
        FROM public.general
        WHERE source = :source
          AND anchor IS NOT NULL AND btrim(anchor) <> ''
          AND positive IS NOT NULL AND btrim(positive) <> ''
          AND length(btrim(regexp_replace(anchor, '[[:space:]]+', ' ', 'g')))
              BETWEEN 15 AND 420
          AND length(btrim(regexp_replace(positive, '[[:space:]]+', ' ', 'g')))
              BETWEEN 80 AND 1400
          AND anchor <> positive
          AND position('�' IN anchor || positive) = 0
          AND position('[[' IN anchor || positive) = 0
          AND position('{{' IN anchor || positive) = 0
          AND (
              :source <> 'Wikipedia vi'
              OR (title IS NOT NULL AND btrim(title) <> '' AND anchor LIKE '% là gì?')
          )
          AND (
              :source <> 'UIT ViQuAD2' OR anchor LIKE '%?'
          )
    ),
    deduplicated AS (
        SELECT *, row_number() OVER (
            PARTITION BY anchor, positive ORDER BY stable_order
        ) AS pair_rank
        FROM candidates
    )
    SELECT data_id, source, title, anchor, positive
    FROM deduplicated
    WHERE pair_rank = 1
    ORDER BY stable_order
    LIMIT :row_limit
    """
)


def selected_rows(connection: Any, spec: SourceSpec) -> Iterator[dict[str, Any]]:
    """Yield deterministic, source-filtered candidates for one quota."""
    statement = LEGAL_SELECTION if spec.source_table == "legal" else GENERAL_SELECTION
    result = connection.execution_options(stream_results=True).execute(
        statement,
        {"source": spec.source, "row_limit": spec.quota},
    )
    yield from result.mappings()


def make_row(spec: SourceSpec, source_row: dict[str, Any]) -> dict[str, Any]:
    """Attach train split, task type, and provenance to a source record."""
    source_data_id = normalize_text(source_row["data_id"])
    anchor = normalize_text(source_row["anchor"])
    positive = normalize_text(source_row["positive"])
    if not source_data_id or not is_clean_pair(anchor, positive):
        raise ValueError(f"Invalid selected row from {spec.source}: {source_data_id!r}")
    title = normalize_text(source_row["title"])
    return {
        "data_id": stable_id(spec, source_data_id),
        "split": "train",
        "task_type": spec.task_type,
        "is_synthetic": spec.is_synthetic,
        "quality_score": spec.quality_score,
        "source_table": spec.source_table,
        "source": spec.source,
        "source_data_id": source_data_id,
        "title": title or None,
        "anchor": anchor,
        "positive": positive,
        "hard_negative": None,
    }


def select_training_rows(connection: Any, specs: Iterable[SourceSpec]) -> list[dict[str, Any]]:
    """Select exact quotas, preventing duplicate IDs and training pairs."""
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for spec in specs:
        source_rows = [make_row(spec, dict(row)) for row in selected_rows(connection, spec)]
        if len(source_rows) != spec.quota:
            raise RuntimeError(
                f"{spec.source}: selected {len(source_rows):,} rows; expected {spec.quota:,}."
            )
        unique_source_rows = [
            row for row in source_rows
            if row["data_id"] not in ids
            and (row["anchor"].casefold(), row["positive"].casefold()) not in pairs
        ]
        if len(unique_source_rows) != spec.quota:
            raise RuntimeError(f"{spec.source}: duplicate pairs cross the hybrid selection.")
        rows.extend(unique_source_rows)
        ids.update(str(row["data_id"]) for row in unique_source_rows)
        pairs.update(
            (str(row["anchor"]).casefold(), str(row["positive"]).casefold())
            for row in unique_source_rows
        )
    return rows


def chunks(rows: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield bounded insert batches."""
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def insert_batch(connection: Any, table: Table, rows: list[dict[str, Any]]) -> int:
    """Insert one batch and surface a duplicate rather than silently masking it."""
    statement = postgresql_insert(table).values(rows).returning(table.c.data_id)
    return len(connection.execute(statement).scalars().all())


def validate_selection(rows: list[dict[str, Any]]) -> None:
    """Verify exact quota, task labeling, and strict benchmark exclusion."""
    expected = sum(spec.quota for spec in SPECS)
    if len(rows) != expected:
        raise RuntimeError(f"Selected {len(rows):,} rows; expected {expected:,}.")
    if len({row["data_id"] for row in rows}) != expected:
        raise RuntimeError("Selected rows contain duplicate IDs.")
    if any(row["split"] != "train" for row in rows):
        raise RuntimeError("A non-train row was selected.")
    if any("GreenNode/" in str(row["source"]) for row in rows):
        raise RuntimeError("VN-MTEB benchmark data must not enter training.")
    if any(not is_clean_pair(str(row["anchor"]), str(row["positive"])) for row in rows):
        raise RuntimeError("A selected row failed the text-quality filter.")
    actual = {(row["source"], row["task_type"]): 0 for row in rows}
    for row in rows:
        actual[(row["source"], row["task_type"])] += 1
    expected_quotas = {(spec.source, spec.task_type): spec.quota for spec in SPECS}
    if actual != expected_quotas:
        raise RuntimeError(f"Unexpected source/task quotas: {actual}")


def validate_database(connection: Any, table: Table) -> None:
    """Check final count, required text, and task distribution in PostgreSQL."""
    expected = sum(spec.quota for spec in SPECS)
    stats = connection.execute(
        select(
            func.count().label("rows"),
            func.count(func.distinct(table.c.data_id)).label("ids"),
            func.count().filter((func.btrim(table.c.anchor) == "") | (func.btrim(table.c.positive) == "")).label("blank"),
            func.count().filter(table.c.split != "train").label("non_train"),
        ).select_from(table)
    ).one()
    if stats.rows != expected or stats.ids != expected or stats.blank or stats.non_train:
        raise RuntimeError(f"Database validation failed: {dict(stats._mapping)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate selection without writing.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    load_dotenv(PROJECT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set.")

    engine = create_engine(database_url, pool_pre_ping=True)
    metadata = MetaData()
    table = build_table(metadata)
    try:
        with engine.connect() as connection:
            rows = select_training_rows(connection, SPECS)
        validate_selection(rows)
        print(f"Selected {len(rows):,} train rows with benchmark data excluded.")
        for spec in SPECS:
            print(f"{spec.task_type}: {spec.source}: {spec.quota:,}")
        if args.dry_run:
            return 0

        metadata.create_all(engine, tables=[table], checkfirst=True)
        with engine.connect() as connection:
            existing = connection.scalar(select(func.count()).select_from(table))
        if existing:
            raise RuntimeError(f"{TABLE_NAME} already contains {existing:,} rows; refusing to mix runs.")

        inserted = 0
        with engine.begin() as connection:
            for index, batch in enumerate(chunks(rows, args.batch_size), start=1):
                inserted += insert_batch(connection, table, batch)
                if index % 20 == 0:
                    print(f"Inserted {inserted:,}/{len(rows):,} rows", flush=True)
        with engine.connect() as connection:
            validate_database(connection, table)
        print(f"Completed: {inserted:,} rows in {TABLE_NAME}.")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
