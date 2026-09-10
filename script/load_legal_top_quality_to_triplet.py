#!/usr/bin/env python3
"""Select 100k diverse legal QA pairs and load them into PostgreSQL ``triplet``.

The source ``legal`` table is read only.  Rows are taken from a reproducible,
stratified page sample, filtered for malformed text, deduplicated by exact
question/answer pair, and limited to one row per title so the resulting
training data covers many legal documents rather than repeatedly sampling the
same one.  Legal source data does not contain
verified hard negatives, so this loader leaves ``hard_negative`` NULL; training
can use in-batch negatives until a separate dense-mining pass is available.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from database.models import TripletModel  # noqa: E402


DEFAULT_LIMIT = 100_000
DEFAULT_BATCH_SIZE = 1_000
LEGAL_SOURCE = "Law Reading Comprehension QA"
ZALO_SOURCE = "Zalo AI Legal"
DESTINATION_SOURCE = "Legal top-100k v1"
TOPIC_PATTERN = re.compile(
    r"\b(Bộ luật|Pháp lệnh|Nghị định|Thông tư|Quyết định|Nghị quyết|Chỉ thị|Công văn|Luật)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class LegalCandidate:
    """One normalized legal QA pair selected for retrieval training."""

    data_id: str
    source: str
    title: str
    anchor: str
    positive: str


def normalize_text(value: str) -> str:
    """Normalize Unicode and spacing without discarding legal paragraph text."""
    normalized = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def infer_topic(title: str) -> str | None:
    """Return the legal document type when it can be read reliably from a title."""
    match = TOPIC_PATTERN.search(title)
    return match.group(1).capitalize() if match else None


SELECTION_QUERY = text(
    """
    WITH raw_candidates AS (
        SELECT data_id, source, title, anchor, positive
        FROM public.legal TABLESAMPLE SYSTEM (15) REPEATABLE (20260910)
        WHERE source = :legal_source
        UNION ALL
        SELECT data_id, source, title, anchor, positive
        FROM public.legal
        WHERE source = :zalo_source
    ),
    quality_filtered AS (
        SELECT
            data_id,
            source,
            title,
            anchor,
            positive,
            CASE
                WHEN position(lower(btrim(title)) IN lower(positive)) > 0 THEN 1
                ELSE 0
            END AS title_in_positive,
            CASE WHEN btrim(positive) ~ '[.!…][[:space:]]*$' THEN 1 ELSE 0 END
                AS ends_sentence,
            md5(data_id) AS stable_order
        FROM raw_candidates
        WHERE title IS NOT NULL AND btrim(title) <> ''
          AND anchor IS NOT NULL AND btrim(anchor) <> ''
          AND positive IS NOT NULL AND btrim(positive) <> ''
          AND length(btrim(regexp_replace(anchor, '[[:space:]]+', ' ', 'g')))
              BETWEEN 20 AND 420
          AND length(btrim(regexp_replace(positive, '[[:space:]]+', ' ', 'g')))
              BETWEEN 120 AND 900
          AND anchor <> positive
          AND position('�' IN anchor) = 0
          AND position('�' IN positive) = 0
          AND position('[[' IN anchor) = 0
          AND position('[[' IN positive) = 0
          AND position('{{' IN anchor) = 0
          AND position('{{' IN positive) = 0
    ),
    deduplicated_pairs AS (
        SELECT *, row_number() OVER (
            PARTITION BY anchor, positive
            ORDER BY title_in_positive DESC, ends_sentence DESC, stable_order
        ) AS pair_rank
        FROM quality_filtered
    ),
    one_per_title AS (
        SELECT *, row_number() OVER (
            PARTITION BY lower(btrim(regexp_replace(title, '[[:space:]]+', ' ', 'g')))
            ORDER BY
                CASE WHEN source = :zalo_source THEN 0 ELSE 1 END,
                title_in_positive DESC,
                ends_sentence DESC,
                abs(length(positive) - 450),
                stable_order
        ) AS title_rank
        FROM deduplicated_pairs
        WHERE pair_rank = 1
    ),
    zalo AS (
        SELECT data_id, source, title, anchor, positive
        FROM one_per_title
        WHERE source = :zalo_source AND title_rank = 1
        ORDER BY md5(title), stable_order
        LIMIT :row_limit
    ),
    law_reading AS (
        SELECT data_id, source, title, anchor, positive
        FROM one_per_title
        WHERE source = :legal_source AND title_rank = 1
        ORDER BY md5(title), stable_order
        LIMIT GREATEST(:row_limit - (SELECT count(*) FROM zalo), 0)
    )
    SELECT data_id, source, title, anchor, positive FROM zalo
    UNION ALL
    SELECT data_id, source, title, anchor, positive FROM law_reading
    ORDER BY source, data_id
    """
).bindparams()


def candidates(connection: Any, row_limit: int) -> Iterable[LegalCandidate]:
    """Stream deterministic, deduplicated legal pairs from the source table."""
    result = connection.execution_options(stream_results=True).execute(
        SELECTION_QUERY,
        {
            "legal_source": LEGAL_SOURCE,
            "zalo_source": ZALO_SOURCE,
            "row_limit": row_limit,
        },
    )
    for row in result.mappings():
        yield LegalCandidate(
            data_id=str(row["data_id"]),
            source=str(row["source"]),
            title=normalize_text(str(row["title"])),
            anchor=normalize_text(str(row["anchor"])),
            positive=normalize_text(str(row["positive"])),
        )


def to_triplet_row(candidate: LegalCandidate) -> dict[str, str | None]:
    """Map a legal candidate to the unified triplet schema."""
    return {
        "data_id": f"legal_top100k_v1_{candidate.data_id}",
        "source": DESTINATION_SOURCE,
        "title": candidate.title,
        "topic": infer_topic(candidate.title),
        "domain": "legal",
        "anchor": candidate.anchor,
        "positive": candidate.positive,
        "hard_negative": None,
    }


def chunks(items: Iterable[dict[str, str | None]], size: int) -> Iterable[list[dict[str, str | None]]]:
    """Yield bounded insert batches."""
    batch: list[dict[str, str | None]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def insert_batch(connection: Any, rows: list[dict[str, str | None]]) -> int:
    """Insert a batch without changing pre-existing destination rows."""
    statement = postgresql_insert(TripletModel).values(rows)
    statement = statement.on_conflict_do_nothing(
        index_elements=[TripletModel.data_id]
    ).returning(TripletModel.data_id)
    return len(connection.execute(statement).scalars().all())


def validate_selection(rows: list[dict[str, str | None]], row_limit: int) -> None:
    """Fail before database writes if the selected rows violate quality rules."""
    if len(rows) != row_limit:
        raise RuntimeError(f"Selected {len(rows):,} rows; expected {row_limit:,}.")
    identifiers = {str(row["data_id"]) for row in rows}
    titles = {str(row["title"]) for row in rows}
    pairs = {(str(row["anchor"]), str(row["positive"])) for row in rows}
    if len(identifiers) != len(rows) or len(pairs) != len(rows):
        raise RuntimeError("The selected legal rows contain duplicate IDs or QA pairs.")
    if any(len(str(row["anchor"])) < 20 or len(str(row["positive"])) < 120 for row in rows):
        raise RuntimeError("The selected legal rows contain text below the quality thresholds.")
    if len(titles) != len(rows):
        raise RuntimeError("The selection contains more than one row for at least one title.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Validate selection without writing to PostgreSQL.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.limit < 1 or args.batch_size < 1:
        raise ValueError("--limit and --batch-size must be positive.")
    load_dotenv(PROJECT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set.")

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        print("Selecting and validating legal candidates...", flush=True)
        with engine.connect() as read_connection:
            read_connection.execute(text("SET statement_timeout = 0"))
            selected = [to_triplet_row(item) for item in candidates(read_connection, args.limit)]
        validate_selection(selected, args.limit)
        print(f"Selected {len(selected):,} rows with {len({row['title'] for row in selected}):,} unique titles.")
        print(f"Destination source: {DESTINATION_SOURCE}")
        if args.dry_run:
            print("Dry run: PostgreSQL triplet table was not changed.")
            return 0

        inserted = 0
        with engine.begin() as write_connection:
            for batch_number, batch in enumerate(chunks(selected, args.batch_size), start=1):
                inserted += insert_batch(write_connection, batch)
                if batch_number % 10 == 0:
                    print(
                        f"Prepared {batch_number * args.batch_size:,}/{len(selected):,} rows "
                        f"({inserted:,} new)",
                        flush=True,
                    )
        print(f"Inserted {inserted:,} rows; {len(selected) - inserted:,} already existed.")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
