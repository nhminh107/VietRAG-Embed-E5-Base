"""Normalize the ``positive`` column in PostgreSQL table ``general``.

Run from the project root:

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate DL_Env
    python preprocess/normalize_general_positive_postgres.py --dry-run
    python preprocess/normalize_general_positive_postgres.py

The script only changes ``positive``. It keeps every row and every other
column unchanged. The live update runs in one transaction, so an error rolls
back all changes.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from sqlalchemy import text


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from database.sql_manager import SQL_Manager  # noqa: E402


TABLE_NAME = "general"
DEFAULT_BATCH_SIZE = 5_000


def normalize_positive(value: str | None) -> str | None:
    """Remove common Wikipedia markup and normalize Unicode and whitespace."""
    if value is None:
        return None

    cleaned = html.unescape(str(value))
    cleaned = unicodedata.normalize("NFC", cleaned)

    # Remove comments and references before removing the remaining tags.
    cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(
        r"<ref\b[^>]*>.*?</ref\s*>",
        " ",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"<ref\b[^>]*/\s*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?[^>]+>", " ", cleaned)

    # Keep the visible label of a wiki link and remove file/category links.
    cleaned = re.sub(
        r"\[\[(?:file|image|tập tin|category|thể loại):[^\]]*\]\]",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", cleaned)
    cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)

    # Templates can be nested. A few passes remove the inner and outer forms.
    for _ in range(10):
        next_value = re.sub(r"\{\{.*?\}\}", " ", cleaned, flags=re.DOTALL)
        if next_value == cleaned:
            break
        cleaned = next_value

    cleaned = re.sub(r"(?<!\w)__.*?__(?!\w)", " ", cleaned)
    cleaned = re.sub(r"formula_\d+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    cleaned = re.sub(r"[{}]", " ", cleaned)
    cleaned = re.sub(r"\|\s*(?:group|class|style)\s*=\s*[^|]+", " ", cleaned)
    cleaned = re.sub(r"\[\s*\d+\s*\]", " ", cleaned)

    # U+FFFD means that the original byte sequence could not be decoded.
    cleaned = cleaned.replace("\ufffd", " ")
    cleaned = html.unescape(cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def get_schema(connection: Any) -> str:
    """Return the schema used by the current PostgreSQL connection."""
    return str(connection.execute(text("SELECT current_schema()")).scalar_one())


def validate_table(connection: Any, schema: str) -> None:
    """Check that the target table has the columns required by this script."""
    columns = {
        row[0]
        for row in connection.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema_name
                  AND table_name = :table_name
                """
            ),
            {"schema_name": schema, "table_name": TABLE_NAME},
        ).fetchall()
    }
    required_columns = {"data_id", "positive"}
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        raise ValueError(
            f"Table {schema}.{TABLE_NAME} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def get_stats(connection: Any) -> dict[str, Any]:
    """Return basic counts and length statistics for ``positive``."""
    row = connection.execute(
        text(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (WHERE positive IS NULL) AS null_positive,
                COUNT(*) FILTER (
                    WHERE positive IS NOT NULL AND btrim(positive) = ''
                ) AS empty_positive,
                COALESCE(MIN(length(positive)), 0) AS min_length,
                COALESCE(ROUND(AVG(length(positive))::numeric, 2), 0) AS avg_length,
                COALESCE(MAX(length(positive)), 0) AS max_length
            FROM {TABLE_NAME}
            """
        )
    ).mappings().one()
    return dict(row)


def print_preview(connection: Any, preview_rows: int) -> None:
    """Print a small before/after preview for manual debugging."""
    if preview_rows <= 0:
        return

    rows = connection.execute(
        text(
            f"""
            SELECT data_id, positive
            FROM {TABLE_NAME}
            WHERE positive IS NOT NULL
            ORDER BY data_id
            LIMIT :preview_rows
            """
        ),
        {"preview_rows": preview_rows},
    ).mappings()

    print("Preview:")
    for row in rows:
        before = str(row["positive"])
        after = normalize_positive(before)
        print(f"  data_id={row['data_id']}")
        print(f"    before: {before[:160]!r}")
        print(f"    after : {after[:160]!r}")


def iter_batches(connection: Any, batch_size: int):
    """Read non-null positives in stable keyset-paginated batches."""
    last_data_id: str | None = None

    while True:
        if last_data_id is None:
            query = text(
                f"""
                SELECT data_id, positive
                FROM {TABLE_NAME}
                WHERE positive IS NOT NULL
                ORDER BY data_id
                LIMIT :batch_size
                """
            )
            parameters = {"batch_size": batch_size}
        else:
            query = text(
                f"""
                SELECT data_id, positive
                FROM {TABLE_NAME}
                WHERE positive IS NOT NULL
                  AND data_id > :last_data_id
                ORDER BY data_id
                LIMIT :batch_size
                """
            )
            parameters = {"last_data_id": last_data_id, "batch_size": batch_size}

        rows = connection.execute(query, parameters).mappings().all()
        if not rows:
            return

        yield rows
        last_data_id = str(rows[-1]["data_id"])


def normalize_table(connection: Any, batch_size: int, dry_run: bool) -> int:
    """Normalize all non-null positives and return the number of changed rows."""
    changed_rows = 0
    processed_rows = 0
    update_query = text(
        f"UPDATE {TABLE_NAME} SET positive = :positive WHERE data_id = :data_id"
    )

    for rows in iter_batches(connection, batch_size):
        updates = []
        for row in rows:
            before = str(row["positive"])
            after = normalize_positive(before)
            if after != before:
                updates.append({"data_id": row["data_id"], "positive": after})

        if updates and not dry_run:
            connection.execute(update_query, updates)

        changed_rows += len(updates)
        processed_rows += len(rows)
        print(
            f"Processed {processed_rows:,} non-null rows; "
            f"changed {changed_rows:,} rows"
        )

    return changed_rows


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Normalize positive in PostgreSQL table general."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and preview changes without updating PostgreSQL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows read per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of rows shown before scanning (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    """Validate, preview, and normalize the PostgreSQL table."""
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.preview < 0:
        raise ValueError("--preview cannot be negative")

    manager = SQL_Manager()
    try:
        with manager.engine.connect() as connection:
            database_name, schema = connection.execute(
                text("SELECT current_database(), current_schema()")
            ).one()
            validate_table(connection, str(schema))
            before_stats = get_stats(connection)
            print(f"Database: {database_name}; schema: {schema}; table: {TABLE_NAME}")
            print(f"Before: {before_stats}")
            print_preview(connection, args.preview)

        if args.dry_run:
            with manager.engine.connect() as connection:
                changed_rows = normalize_table(
                    connection, args.batch_size, dry_run=True
                )
            print(f"Dry-run finished. Rows that would change: {changed_rows:,}")
            return

        # One transaction keeps the update atomic. An exception rolls it back.
        with manager.engine.begin() as connection:
            changed_rows = normalize_table(
                connection, args.batch_size, dry_run=False
            )
        print(f"Update committed. Changed rows: {changed_rows:,}")

        with manager.engine.connect() as connection:
            after_stats = get_stats(connection)
            print(f"After: {after_stats}")
            if after_stats["total_rows"] != before_stats["total_rows"]:
                raise RuntimeError("Row count changed unexpectedly")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
