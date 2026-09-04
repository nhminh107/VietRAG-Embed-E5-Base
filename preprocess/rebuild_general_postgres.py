"""Rebuild PostgreSQL table ``general`` with document-level cleaning.

The current ``general`` table contains Wikipedia chunks and QA/paraphrase
pairs. Wikipedia chunks are stitched by ``source`` and ``title`` first, then
cleaned and split again at word boundaries. QA rows are copied as semantic
pairs because their positive text is not a document chunk.

The script builds ``general_rebuilt`` first. It only replaces ``general``
when ``--commit`` is passed, and keeps the old table as a backup.

Run from the project root:

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate DL_Env
    python preprocess/rebuild_general_postgres.py
    python preprocess/rebuild_general_postgres.py --commit
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from database.sql_manager import SQL_Manager  # noqa: E402


SOURCE_TABLE = "general"
STAGING_TABLE = "general_rebuilt"
WIKIPEDIA_SOURCE = "Wikipedia vi"
DEFAULT_MAX_CHARS = 384
DEFAULT_OVERLAP_CHARS = 48
DEFAULT_MIN_CHARS = 80
DEFAULT_INSERT_BATCH_SIZE = 5_000


def quote_identifier(value: str) -> str:
    """Quote a known PostgreSQL identifier after validating its characters."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError(f"Unsafe PostgreSQL identifier: {value!r}")
    return f'"{value}"'


def normalize_document_text(value: str) -> str:
    """Clean a complete document after its old chunks have been stitched."""
    cleaned = html.unescape(value)
    cleaned = unicodedata.normalize("NFC", cleaned)

    cleaned = cleaned.replace("\u200b", " ").replace("\ufeff", " ")
    cleaned = cleaned.replace("\ufffd", " ")
    cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(
        r"<ref\b[^>]*>.*?</ref\s*>",
        " ",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"<ref\b[^>]*/\s*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?[^>]+>", " ", cleaned)

    # Remove complete file/image/category links before normal wiki links.
    cleaned = re.sub(
        r"\[\[(?:file|image|hình|hinh|tập tin|category|thể loại|media):[^\]]*\]\]",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", cleaned)
    cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)

    for _ in range(10):
        next_value = re.sub(r"\{\{.*?\}\}", " ", cleaned, flags=re.DOTALL)
        if next_value == cleaned:
            break
        cleaned = next_value

    # Remove only known MediaWiki magic words. Do not remove arbitrary
    # ``__...__`` because Wikipedia taxonomy text can contain underscores.
    cleaned = re.sub(
        r"__(?:NOTOC|TOC|FORCETOC|NOEDITSECTION|HIDDENCAT|INDEX|NOINDEX|NOGALLERY|NEWSECTIONLINK|NONEWSECTIONLINK)__",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\|\s*(?:group|class|style)\s*=\s*[^|]+", " ", cleaned)
    cleaned = re.sub(r"\[\s*\d+\s*\]", " ", cleaned)
    cleaned = re.sub(r"(?i)(?:[A-Za-z]{0,3};)?br[&>]", " ", cleaned)
    cleaned = re.sub(r"(?i);br>", " ", cleaned)

    # Any remaining brackets are malformed links left by an old chunk
    # boundary. Removing only the markers preserves the visible text.
    cleaned = cleaned.replace("[[", " ").replace("]]", " ")
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_pair_text(value: str | None) -> str | None:
    """Apply the same safe text cleanup to a non-Wikipedia pair field."""
    if value is None:
        return None
    return normalize_document_text(str(value))


def find_overlap(left: str, right: str, expected: int) -> int:
    """Find a conservative old overlap around the expected character size."""
    maximum = min(expected, len(left), len(right))
    minimum = min(32, maximum)
    for size in range(maximum, minimum - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def stitch_rows(rows: Sequence[dict[str, Any]], overlap_chars: int) -> str:
    """Reconstruct one document from its old chunks."""
    document = str(rows[0]["positive"] or "")
    previous = document

    for row in rows[1:]:
        current = str(row["positive"] or "")
        overlap = find_overlap(previous, current, overlap_chars)
        document += current[overlap:]
        previous = current

    return document


def chunk_by_words(
    value: str,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[str]:
    """Create word-boundary chunks with a bounded character overlap."""
    words = value.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start
        current_length = 0

        while end < len(words):
            word = words[end]
            next_length = len(word) if end == start else current_length + 1 + len(word)
            if end > start and next_length > max_chars:
                break
            current_length = next_length
            end += 1

        if end == start:
            # Extremely long tokens are rare, but must not make the loop hang.
            end += 1

        chunk = " ".join(words[start:end])
        if len(chunk) < min_chars:
            break
        chunks.append(chunk)

        if end >= len(words):
            break

        overlap_start = end
        overlap_length = 0
        while overlap_start > start:
            word_length = len(words[overlap_start - 1])
            extra_length = word_length + (1 if overlap_length else 0)
            if overlap_length + extra_length > overlap_chars:
                break
            overlap_length += extra_length
            overlap_start -= 1

        start = max(start + 1, overlap_start)

    return chunks


def iter_wikipedia_documents(
    connection: Any,
) -> Iterator[list[dict[str, Any]]]:
    """Stream Wikipedia rows grouped by title in stable order."""
    result = connection.execution_options(
        stream_results=True,
        max_row_buffer=5_000,
    ).execute(
        text(
            f"""
            SELECT data_id, source, title, topic, anchor, positive, hard_negative
            FROM {quote_identifier(SOURCE_TABLE)}
            WHERE source = :source
              AND positive IS NOT NULL
            ORDER BY title NULLS LAST, data_id
            """
        ),
        {"source": WIKIPEDIA_SOURCE},
    ).mappings()

    current_key: str | None = None
    document_rows: list[dict[str, Any]] = []

    for row in result:
        title = str(row["title"] or "").strip()
        row_key = title or str(row["data_id"])
        if current_key is None:
            current_key = row_key
        if row_key != current_key:
            if document_rows:
                yield document_rows
            document_rows = []
            current_key = row_key
        document_rows.append(dict(row))

    if document_rows:
        yield document_rows


def iter_qa_rows(connection: Any) -> Iterator[dict[str, Any]]:
    """Stream non-Wikipedia rows without changing their pair semantics."""
    result = connection.execution_options(
        stream_results=True,
        max_row_buffer=5_000,
    ).execute(
        text(
            f"""
            SELECT data_id, source, title, topic, anchor, positive, hard_negative
            FROM {quote_identifier(SOURCE_TABLE)}
            WHERE source <> :source
              AND positive IS NOT NULL
              AND length(btrim(positive)) > 0
            ORDER BY data_id
            """
        ),
        {"source": WIKIPEDIA_SOURCE},
    ).mappings()
    for row in result:
        cleaned_row = dict(row)
        cleaned_row["anchor"] = normalize_pair_text(cleaned_row["anchor"])
        cleaned_row["positive"] = normalize_pair_text(cleaned_row["positive"])
        if cleaned_row["anchor"] and cleaned_row["positive"]:
            yield cleaned_row


def create_staging_table(connection: Any, reset: bool) -> None:
    """Create the empty staging table, optionally resetting only that table."""
    staging = quote_identifier(STAGING_TABLE)
    exists = connection.execute(
        text("SELECT to_regclass(:table_name) IS NOT NULL"),
        {"table_name": STAGING_TABLE},
    ).scalar_one()
    if exists and not reset:
        raise RuntimeError(
            f"Table {STAGING_TABLE!r} already exists. "
            "Use --reset-staging only to rebuild that staging table."
        )
    if exists:
        connection.execute(text(f"DROP TABLE {staging}"))

    connection.execute(
        text(
            f"""
            CREATE TABLE {staging} (
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
    )


def insert_rows(connection: Any, rows: list[dict[str, Any]]) -> None:
    """Insert one batch into staging."""
    if not rows:
        return
    connection.execute(
        text(
            f"""
            INSERT INTO {quote_identifier(STAGING_TABLE)}
                (data_id, source, title, topic, anchor, positive, hard_negative)
            VALUES
                (:data_id, :source, :title, :topic, :anchor, :positive, :hard_negative)
            """
        ),
        rows,
    )


def build_staging(
    engine: Any,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
    insert_batch_size: int,
    reset_staging: bool,
) -> dict[str, int]:
    """Build staging from all rows without a record limit."""
    with engine.begin() as connection:
        create_staging_table(connection, reset=reset_staging)

    stats = {
        "wikipedia_documents": 0,
        "wikipedia_skipped_documents": 0,
        "wikipedia_chunks": 0,
        "qa_rows": 0,
        "inserted_rows": 0,
    }

    wikipedia_batch: list[dict[str, Any]] = []
    qa_batch: list[dict[str, Any]] = []

    with engine.connect() as read_connection, engine.begin() as write_connection:
        for document_rows in iter_wikipedia_documents(read_connection):
            stats["wikipedia_documents"] += 1
            stitched = stitch_rows(document_rows, overlap_chars)
            cleaned = normalize_document_text(stitched)
            chunks = chunk_by_words(
                cleaned,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
            )

            if not chunks:
                stats["wikipedia_skipped_documents"] += 1
                continue

            first_row = document_rows[0]
            title = str(first_row["title"] or "").strip() or None
            anchor = normalize_document_text(str(first_row["anchor"] or ""))
            if not anchor and title:
                anchor = f'"{title}" là gì?'
            if not anchor:
                stats["wikipedia_skipped_documents"] += 1
                continue

            base_id = str(first_row["data_id"])
            for chunk_index, chunk in enumerate(chunks):
                wikipedia_batch.append(
                    {
                        "data_id": f"{base_id}_chunk_{chunk_index:06d}",
                        "source": first_row["source"],
                        "title": title,
                        "topic": first_row["topic"],
                        "anchor": anchor,
                        "positive": chunk,
                        "hard_negative": first_row["hard_negative"],
                    }
                )
                stats["wikipedia_chunks"] += 1

                if len(wikipedia_batch) >= insert_batch_size:
                    insert_rows(write_connection, wikipedia_batch)
                    stats["inserted_rows"] += len(wikipedia_batch)
                    wikipedia_batch.clear()

            if stats["wikipedia_documents"] % 1_000 == 0:
                print(
                    f"Wikipedia documents: {stats['wikipedia_documents']:,}; "
                    f"chunks: {stats['wikipedia_chunks']:,}",
                    flush=True,
                )

        if wikipedia_batch:
            insert_rows(write_connection, wikipedia_batch)
            stats["inserted_rows"] += len(wikipedia_batch)

        for row in iter_qa_rows(read_connection):
            qa_batch.append(row)
            stats["qa_rows"] += 1
            if len(qa_batch) >= insert_batch_size:
                insert_rows(write_connection, qa_batch)
                stats["inserted_rows"] += len(qa_batch)
                qa_batch.clear()
                if stats["qa_rows"] % 50_000 == 0:
                    print(f"QA rows: {stats['qa_rows']:,}", flush=True)

        if qa_batch:
            insert_rows(write_connection, qa_batch)
            stats["inserted_rows"] += len(qa_batch)

    return stats


def validate_staging(
    engine: Any,
    max_chars: int,
    min_chars: int,
) -> dict[str, Any]:
    """Validate staging before it can replace the live table."""
    staging = quote_identifier(STAGING_TABLE)
    with engine.connect() as connection:
        stats = dict(
            connection.execute(
                text(
                    f"""
                    SELECT
                        COUNT(*) AS total_rows,
                        COUNT(*) FILTER (WHERE positive IS NULL) AS positive_null,
                        COUNT(*) FILTER (WHERE positive IS NOT NULL AND length(btrim(positive)) = 0) AS positive_empty,
                        COUNT(*) FILTER (WHERE source = :wiki AND length(positive) < :min_chars) AS wiki_short,
                        COUNT(*) FILTER (WHERE source = :wiki AND length(positive) > :max_chars) AS wiki_long,
                        COUNT(*) FILTER (WHERE positive ~ :open_link OR positive ~ :close_link) AS residual_wikilinks,
                        COUNT(*) FILTER (WHERE position(:zero_width in positive) > 0 OR position(:bom in positive) > 0) AS hidden_chars,
                        COUNT(*) FILTER (WHERE positive LIKE :replacement) AS replacement_chars,
                        MIN(length(positive)) AS min_positive,
                        ROUND(AVG(length(positive))::numeric, 2) AS avg_positive,
                        MAX(length(positive)) AS max_positive
                    FROM {staging}
                    """
                ),
                {
                    "wiki": WIKIPEDIA_SOURCE,
                    "min_chars": min_chars,
                    "max_chars": max_chars,
                    "open_link": r"\[\[",
                    "close_link": r"\]\]",
                    "zero_width": "\u200b",
                    "bom": "\ufeff",
                    "replacement": "%�%",
                },
            ).mappings().one()
        )
        by_source = [
            dict(row)
            for row in connection.execute(
                text(
                    f"""
                    SELECT source, COUNT(*) AS rows,
                           MIN(length(positive)) AS min_positive,
                           ROUND(AVG(length(positive))::numeric, 2) AS avg_positive,
                           MAX(length(positive)) AS max_positive
                    FROM {staging}
                    GROUP BY source
                    ORDER BY rows DESC
                    """
                )
            ).mappings()
        ]

    if stats["total_rows"] == 0:
        raise RuntimeError("Staging table is empty")
    if stats["positive_null"] or stats["positive_empty"]:
        raise RuntimeError(f"Staging has invalid positives: {stats}")
    if stats["wiki_short"] or stats["wiki_long"]:
        raise RuntimeError(f"Wikipedia chunks violate size limits: {stats}")
    if stats["hidden_chars"] or stats["replacement_chars"]:
        raise RuntimeError(f"Staging still contains hidden/broken characters: {stats}")

    stats["by_source"] = by_source
    return stats


def commit_staging(engine: Any) -> str:
    """Atomically rename live general to a backup and promote staging."""
    backup_table = f"general_before_rebuild_{datetime.now():%Y%m%d_%H%M%S}"
    with engine.begin() as connection:
        exists = connection.execute(
            text("SELECT to_regclass(:table_name) IS NOT NULL"),
            {"table_name": backup_table},
        ).scalar_one()
        if exists:
            raise RuntimeError(f"Backup table already exists: {backup_table}")
        connection.execute(
            text(
                f"ALTER TABLE {quote_identifier(SOURCE_TABLE)} "
                f"RENAME TO {quote_identifier(backup_table)}"
            )
        )
        connection.execute(
            text(
                f"ALTER TABLE {quote_identifier(STAGING_TABLE)} "
                f"RENAME TO {quote_identifier(SOURCE_TABLE)}"
            )
        )
    return backup_table


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Promote validated staging to general and keep a backup table.",
    )
    parser.add_argument(
        "--promote-existing",
        action="store_true",
        help="Validate and promote an existing general_rebuilt table without rebuilding it.",
    )
    parser.add_argument(
        "--reset-staging",
        action="store_true",
        help="Drop only an existing general_rebuilt staging table before rebuilding.",
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument(
        "--insert-batch-size",
        type=int,
        default=DEFAULT_INSERT_BATCH_SIZE,
    )
    return parser.parse_args()


def main() -> None:
    """Build, validate, and optionally promote the rebuilt table."""
    args = parse_args()
    if args.promote_existing and args.commit:
        raise ValueError("Use either --commit or --promote-existing, not both")
    if args.max_chars <= 0 or args.overlap_chars <= 0 or args.min_chars <= 0:
        raise ValueError("Chunk sizes must be greater than zero")
    if args.overlap_chars >= args.max_chars:
        raise ValueError("--overlap-chars must be smaller than --max-chars")
    if args.insert_batch_size <= 0:
        raise ValueError("--insert-batch-size must be greater than zero")

    manager = SQL_Manager()
    try:
        if args.promote_existing:
            validation = validate_staging(
                manager.engine,
                max_chars=args.max_chars,
                min_chars=args.min_chars,
            )
            print("Existing staging validation:", validation)
            backup_table = commit_staging(manager.engine)
            print(f"Committed. Old general is kept as {backup_table!r}.")
            return

        print("Building PostgreSQL staging table general_rebuilt ...")
        build_stats = build_staging(
            manager.engine,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            min_chars=args.min_chars,
            insert_batch_size=args.insert_batch_size,
            reset_staging=args.reset_staging,
        )
        print("Build summary:", build_stats)

        validation = validate_staging(
            manager.engine,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
        )
        print("Validation summary:", validation)

        if args.commit:
            backup_table = commit_staging(manager.engine)
            print(f"Committed. Old general is kept as {backup_table!r}.")
        else:
            print("Staging kept. Re-run with --commit to promote it.")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
