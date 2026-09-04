"""Clean, re-chunk, and rebuild the general triplet SQLite database.

The script uses the current ``general_triplet`` table as its input because the
raw ``general`` database is not available in this workspace.  It reconstructs
each document from the old overlapping chunks, cleans the text, creates new
word-boundary chunks, and chooses a deterministic lexical hard negative from a
different title.

Run from the project root:

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate DL_Env
    python preprocess/rebuild_general_triplet.py

Use ``--no-replace`` to build and validate the cleaned database without
replacing the input file.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DB_PATH = PROJECT_DIR / "database" / "general_triplet.db"
OUTPUT_DB_PATH = PROJECT_DIR / "database" / "general_triplet.cleaned.db"
BACKUP_DB_PATH = PROJECT_DIR / "database" / "general_triplet.before_cleaning.db"

SOURCE_TABLE = "general_triplet"
CHUNK_SIZE = 384
CHUNK_OVERLAP = 48
MIN_CHUNK_CHARS = 80
INSERT_BATCH_SIZE = 1_000
PROGRESS_EVERY_DOCUMENTS = 10_000
MAX_FALLBACK_CANDIDATES = 100

STOPWORDS = {
    "a",
    "an",
    "and",
    "các",
    "cho",
    "của",
    "gi",
    "gì",
    "is",
    "là",
    "một",
    "of",
    "the",
    "và",
}


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata needed to create rows and choose a hard negative."""

    index: int
    source: str | None
    title: str | None
    topic: str | None
    anchor: str
    first_positive: str
    title_tokens: frozenset[str]


def validate_input_table(connection: sqlite3.Connection) -> None:
    """Validate the input table before reading it."""
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
            f"Input table {SOURCE_TABLE!r} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def input_rows(connection: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    """Stream input rows in the same order used to create the old chunks."""
    query = f"""
        SELECT data_id, source, title, topic, anchor, positive
        FROM {SOURCE_TABLE}
        WHERE positive IS NOT NULL
          AND length(trim(positive)) > 0
        ORDER BY data_id
    """
    yield from connection.execute(query)


def document_key(row: sqlite3.Row) -> tuple[str | None, str | None] | str:
    """Return a key for one document.

    A non-empty title identifies a document in the current Wikipedia data. If
    a future source has no title, each row is kept as its own document instead
    of accidentally merging the whole source together.
    """
    title = row["title"]
    if title is None or not str(title).strip():
        return str(row["data_id"])
    return row["source"], str(title).strip()


def iter_documents(
    connection: sqlite3.Connection,
) -> Iterator[list[sqlite3.Row]]:
    """Yield consecutive rows belonging to the same document."""
    current_key: tuple[str | None, str | None] | str | None = None
    rows: list[sqlite3.Row] = []

    for row in input_rows(connection):
        row_key = document_key(row)
        if current_key is None:
            current_key = row_key

        if row_key != current_key:
            yield rows
            rows = []
            current_key = row_key

        rows.append(row)

    if rows:
        yield rows


def find_overlap(left: str, right: str) -> int:
    """Find the old chunk overlap, checking the expected size first."""
    maximum = min(CHUNK_OVERLAP, len(left), len(right))
    if maximum == 0:
        return 0

    if left[-maximum:] == right[:maximum]:
        return maximum

    for size in range(maximum - 1, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def stitch_old_chunks(rows: list[sqlite3.Row]) -> str:
    """Reconstruct the old document before cleaning it."""
    document = str(rows[0]["positive"])
    previous_chunk = document

    for row in rows[1:]:
        chunk = str(row["positive"])
        overlap = find_overlap(previous_chunk, chunk)
        document += chunk[overlap:]
        previous_chunk = chunk

    return document


def clean_text(value: str) -> str:
    """Remove common Wikipedia markup and normalize whitespace."""
    text = html.unescape(value)

    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"<ref\b[^>]*>.*?</ref\s*>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<ref\b[^>]*/\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", " ", text)

    text = re.sub(
        r"\[\[(?:file|image|tập tin|category|thể loại):[^\]]*\]\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    for _ in range(3):
        text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)__.*?__(?!\w)", " ", text)
    text = re.sub(r"formula_\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\|\s*(?:group|class|style)\s*=\s*[^|]+", " ", text)
    text = re.sub(r"\[\s*\d+\s*\]", " ", text)

    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_tokens(title: str | None) -> frozenset[str]:
    """Extract useful title words for lexical hard-negative mining."""
    if not title:
        return frozenset()

    tokens = re.findall(r"\w+", title.casefold(), flags=re.UNICODE)
    return frozenset(token for token in tokens if token not in STOPWORDS)


def chunk_text(text: str) -> list[str]:
    """Create <=384-character chunks without splitting normal words.

    The overlap is up to 48 characters because it is made from complete words.
    A final chunk shorter than ``MIN_CHUNK_CHARS`` is dropped. This avoids
    training on punctuation-only or nearly empty tails.
    """
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start
        current_length = 0

        while end < len(words):
            word = words[end]
            next_length = len(word) if end == start else current_length + 1 + len(word)

            if end > start and next_length > CHUNK_SIZE:
                break

            current_length = next_length
            end += 1

        chunk = " ".join(words[start:end])
        if len(chunk) < MIN_CHUNK_CHARS:
            break

        chunks.append(chunk)
        if end >= len(words):
            break

        overlap_start = end
        overlap_length = 0
        while overlap_start > start:
            word_length = len(words[overlap_start - 1])
            extra_length = word_length + (1 if overlap_length else 0)
            if overlap_length + extra_length > CHUNK_OVERLAP:
                break
            overlap_length += extra_length
            overlap_start -= 1

        start = max(start + 1, overlap_start)

    return chunks


def prepare_document(
    rows: list[sqlite3.Row],
    document_index: int,
) -> tuple[DocumentMetadata, str, list[str]] | None:
    """Reconstruct, clean, and chunk one document."""
    raw_text = stitch_old_chunks(rows)
    text = clean_text(raw_text)
    if not any(character.isalpha() for character in text):
        return None

    chunks = chunk_text(text)
    if not chunks:
        return None

    title = rows[0]["title"]
    title = str(title).strip() if title is not None else None
    source = rows[0]["source"]
    topic = rows[0]["topic"]

    if title:
        anchor = f'"{title}" là gì?'
    else:
        anchor = clean_text(str(rows[0]["anchor"] or ""))

    metadata = DocumentMetadata(
        index=document_index,
        source=source,
        title=title,
        topic=topic,
        anchor=anchor,
        first_positive=chunks[0],
        title_tokens=title_tokens(title),
    )
    return metadata, text, chunks


def read_metadata(
    connection: sqlite3.Connection,
) -> tuple[list[DocumentMetadata], int]:
    """Read metadata for every valid document in one pass."""
    metadata: list[DocumentMetadata] = []
    skipped_documents = 0

    for raw_index, rows in enumerate(iter_documents(connection), start=1):
        prepared = prepare_document(rows, len(metadata))
        if prepared is None:
            skipped_documents += 1
            continue

        metadata.append(prepared[0])
        if raw_index % PROGRESS_EVERY_DOCUMENTS == 0:
            print(
                f"Read documents: {raw_index:,}; "
                f"valid: {len(metadata):,}; skipped: {skipped_documents:,}",
                flush=True,
            )

    return metadata, skipped_documents


def build_title_index(
    metadata: list[DocumentMetadata],
) -> dict[str, list[int]]:
    """Map each title token to document indexes."""
    index: dict[str, list[int]] = defaultdict(list)
    for item in metadata:
        for token in item.title_tokens:
            index[token].append(item.index)
    return index


def candidate_indexes(
    item: DocumentMetadata,
    metadata: list[DocumentMetadata],
    title_index: dict[str, list[int]],
) -> list[int]:
    """Return candidates with similar title words, then deterministic fallbacks."""
    scores: dict[int, int] = defaultdict(int)

    for token in item.title_tokens:
        posting = title_index[token]
        position = bisect_left(posting, item.index)
        start = max(0, position - 20)
        end = min(len(posting), position + 21)
        for candidate_index in posting[start:end]:
            if candidate_index == item.index:
                continue
            if metadata[candidate_index].title == item.title:
                continue
            scores[candidate_index] += 1

    ranked = sorted(
        scores,
        key=lambda candidate_index: (
            -scores[candidate_index],
            abs(candidate_index - item.index),
            candidate_index,
        ),
    )

    fallback_count = 0
    for offset in range(1, len(metadata)):
        candidate_index = (item.index + offset) % len(metadata)
        if candidate_index == item.index:
            continue
        if metadata[candidate_index].title == item.title:
            continue
        if candidate_index not in scores:
            ranked.append(candidate_index)
            fallback_count += 1
            if fallback_count >= MAX_FALLBACK_CANDIDATES:
                break

    return ranked


def choose_hard_negative(
    item: DocumentMetadata,
    document_text: str,
    metadata: list[DocumentMetadata],
    title_index: dict[str, list[int]],
) -> str:
    """Choose a clean passage from a different title."""
    for candidate_index in candidate_indexes(item, metadata, title_index):
        candidate = metadata[candidate_index]
        negative = candidate.first_positive
        if not negative or negative in document_text:
            continue
        return negative

    raise RuntimeError(
        f"Could not find a hard negative for document {item.index} "
        f"({item.title!r})."
    )


def create_output_database() -> sqlite3.Connection:
    """Create a fresh output database."""
    if OUTPUT_DB_PATH == INPUT_DB_PATH:
        raise ValueError("Output database must be different from the input database.")
    if OUTPUT_DB_PATH.exists():
        OUTPUT_DB_PATH.unlink()

    connection = sqlite3.connect(OUTPUT_DB_PATH)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute(
        """
        CREATE TABLE general_triplet (
            data_id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            topic TEXT,
            anchor TEXT NOT NULL,
            positive TEXT NOT NULL,
            hard_negative TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def rebuild_database(
    source_connection: sqlite3.Connection,
    metadata: list[DocumentMetadata],
) -> int:
    """Write every cleaned document to the new database."""
    title_index = build_title_index(metadata)
    output_connection = create_output_database()
    insert_query = """
        INSERT INTO general_triplet (
            data_id, source, title, topic, anchor, positive, hard_negative
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    batch: list[tuple[object, ...]] = []
    inserted_rows = 0
    document_count = 0

    try:
        for raw_rows in iter_documents(source_connection):
            prepared = prepare_document(raw_rows, document_count)
            if prepared is None:
                continue

            item, document_text, chunks = prepared
            expected_item = metadata[document_count]
            if item != expected_item:
                raise RuntimeError(
                    "Input changed between the metadata pass and the write pass."
                )

            hard_negative = choose_hard_negative(
                item,
                document_text,
                metadata,
                title_index,
            )

            for chunk_index, chunk in enumerate(chunks):
                batch.append(
                    (
                        f"data_clean_{item.index:07d}_{chunk_index:04d}",
                        item.source,
                        item.title,
                        item.topic,
                        item.anchor,
                        chunk,
                        hard_negative,
                    )
                )

            document_count += 1
            if len(batch) >= INSERT_BATCH_SIZE:
                output_connection.executemany(insert_query, batch)
                output_connection.commit()
                inserted_rows += len(batch)
                batch.clear()

            if document_count % PROGRESS_EVERY_DOCUMENTS == 0:
                print(
                    f"Rebuilt documents: {document_count:,}/{len(metadata):,}; "
                    f"rows written: {inserted_rows + len(batch):,}",
                    flush=True,
                )

        if batch:
            output_connection.executemany(insert_query, batch)
            output_connection.commit()
            inserted_rows += len(batch)

        if document_count != len(metadata):
            raise RuntimeError(
                f"Rebuilt {document_count:,} documents, expected {len(metadata):,}."
            )
    finally:
        output_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        output_connection.execute("PRAGMA journal_mode = DELETE")
        output_connection.close()

    return inserted_rows


def validate_output_database(expected_rows: int) -> None:
    """Run inexpensive correctness checks on the rebuilt database."""
    connection = sqlite3.connect(OUTPUT_DB_PATH)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")

        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(DISTINCT data_id) AS distinct_ids,
                SUM(anchor IS NULL OR length(trim(anchor)) = 0),
                SUM(positive IS NULL OR length(trim(positive)) = 0),
                SUM(hard_negative IS NULL OR length(trim(hard_negative)) = 0),
                MAX(length(positive)),
                SUM(positive = hard_negative)
            FROM general_triplet
            """
        ).fetchone()

        total, distinct_ids, empty_anchor, empty_positive, empty_negative, max_positive, same_text = row
        checks = {
            "row_count": total == expected_rows,
            "unique_ids": total == distinct_ids,
            "non_empty_anchor": empty_anchor == 0,
            "non_empty_positive": empty_positive == 0,
            "non_empty_negative": empty_negative == 0,
            "positive_max_<=_chunk_size": max_positive <= CHUNK_SIZE,
            "positive_not_equal_negative": same_text == 0,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            raise RuntimeError(f"Output validation failed: {failed_checks}")

        print(f"Output rows: {total:,}")
        print(f"Distinct IDs: {distinct_ids:,}")
        print(f"Max positive length: {max_positive}")
        print(f"Integrity check: {integrity}")
    finally:
        connection.close()


def replace_input_with_output() -> None:
    """Keep a backup and replace the input database atomically."""
    if not BACKUP_DB_PATH.exists():
        print(f"Creating backup: {BACKUP_DB_PATH}")
        shutil.copy2(INPUT_DB_PATH, BACKUP_DB_PATH)
    else:
        print(f"Keeping existing backup: {BACKUP_DB_PATH}")

    os.replace(OUTPUT_DB_PATH, INPUT_DB_PATH)
    print(f"Updated database: {INPUT_DB_PATH}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Build and validate the cleaned DB without replacing the input DB.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete rebuild."""
    args = parse_args()
    if not INPUT_DB_PATH.is_file():
        raise FileNotFoundError(f"Input database does not exist: {INPUT_DB_PATH}")

    source_connection = sqlite3.connect(
        f"file:{INPUT_DB_PATH}?mode=ro",
        uri=True,
    )
    source_connection.row_factory = sqlite3.Row

    try:
        validate_input_table(source_connection)
        source_count = source_connection.execute(
            f"SELECT COUNT(*) FROM {SOURCE_TABLE}"
        ).fetchone()[0]
        print(f"Input rows: {source_count:,}")

        metadata, skipped_documents = read_metadata(source_connection)
        print(
            f"Valid documents: {len(metadata):,}; "
            f"skipped documents: {skipped_documents:,}"
        )

        rebuilt_rows = rebuild_database(source_connection, metadata)
        validate_output_database(rebuilt_rows)
    finally:
        source_connection.close()

    if args.no_replace:
        print(f"Cleaned database kept at: {OUTPUT_DB_PATH}")
    else:
        replace_input_with_output()


if __name__ == "__main__":
    main()
