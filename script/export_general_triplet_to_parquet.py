"""Export the complete PostgreSQL ``general_triplet`` table to Parquet.

The PostgreSQL source is read-only. Rows are streamed in bounded batches to an
atomic output file so a failed export never replaces a completed Parquet file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import text


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from database.sql_manager import SQL_Manager  # noqa: E402


SOURCE_TABLE = "general_triplet"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "database" / "general_triplet.parquet"
DEFAULT_BATCH_SIZE = 50_000
COLUMNS = (
    "data_id",
    "source",
    "title",
    "topic",
    "anchor",
    "positive",
    "hard_negative",
)
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("data_id", pa.string(), nullable=False),
        pa.field("source", pa.string()),
        pa.field("title", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("anchor", pa.string(), nullable=False),
        pa.field("positive", pa.string()),
        pa.field("hard_negative", pa.string()),
    ]
)


def validate_source(connection: Any) -> int:
    """Validate the expected source columns and return the source row count."""
    source_columns = {
        row[0]
        for row in connection.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = :table_name
                """
            ),
            {"table_name": SOURCE_TABLE},
        )
    }
    missing_columns = set(COLUMNS).difference(source_columns)
    if missing_columns:
        raise ValueError(
            f"PostgreSQL table {SOURCE_TABLE!r} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return int(
        connection.execute(
            text(f"SELECT COUNT(*) FROM {SOURCE_TABLE}")
        ).scalar_one()
    )


def calculate_sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output(path: Path, expected_rows: int) -> dict[str, Any]:
    """Validate Parquet metadata, schema, and required training fields."""
    parquet_file = pq.ParquetFile(path)
    actual_rows = parquet_file.metadata.num_rows
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"Parquet has {actual_rows:,} rows, expected {expected_rows:,}."
        )
    if parquet_file.schema_arrow != PARQUET_SCHEMA:
        raise RuntimeError(
            "Parquet schema mismatch: "
            f"expected {PARQUET_SCHEMA}, found {parquet_file.schema_arrow}."
        )

    incomplete_rows = 0
    for batch in parquet_file.iter_batches(
        batch_size=DEFAULT_BATCH_SIZE,
        columns=["data_id", "anchor", "positive", "hard_negative"],
    ):
        for column_name in ("data_id", "anchor"):
            column = batch.column(column_name)
            incomplete_rows += column.null_count
            incomplete_rows += int(
                pa.compute.sum(pa.compute.equal(pa.compute.utf8_trim_whitespace(column), "")).as_py()
                or 0
            )
    if incomplete_rows:
        raise RuntimeError(
            f"Parquet contains {incomplete_rows:,} null/blank required values."
        )

    return {
        "rows": actual_rows,
        "row_groups": parquet_file.metadata.num_row_groups,
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
    }


def export_table(output_path: Path, batch_size: int, overwrite: bool) -> dict[str, Any]:
    """Stream PostgreSQL rows into a compressed, atomically published Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp.parquet",
        dir=output_path.parent,
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)
    manager = SQL_Manager()
    writer: pq.ParquetWriter | None = None

    try:
        with manager.engine.connect() as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            expected_rows = validate_source(connection)
            print(f"PostgreSQL rows to export: {expected_rows:,}", flush=True)

            query = text(
                f"SELECT {', '.join(COLUMNS)} "
                f"FROM {SOURCE_TABLE} ORDER BY data_id"
            )
            result = connection.execution_options(stream_results=True).execute(query)
            writer = pq.ParquetWriter(
                temporary_path,
                PARQUET_SCHEMA,
                compression="zstd",
                compression_level=3,
                use_dictionary=["source", "topic"],
                write_statistics=True,
            )

            exported_rows = 0
            for partition in result.mappings().partitions(batch_size):
                rows = [dict(row) for row in partition]
                table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
                writer.write_table(table, row_group_size=batch_size)
                exported_rows += len(rows)
                print(f"Exported {exported_rows:,} rows", flush=True)

            writer.close()
            writer = None

        metadata = validate_output(temporary_path, expected_rows)
        os.replace(temporary_path, output_path)
        return metadata
    finally:
        if writer is not None:
            writer.close()
        manager.close()
        if temporary_path.exists():
            temporary_path.unlink()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the PostgreSQL-to-Parquet export."""
    arguments = parse_arguments()
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    output_path = arguments.output.resolve()
    metadata = export_table(
        output_path=output_path,
        batch_size=arguments.batch_size,
        overwrite=arguments.overwrite,
    )
    print(f"Completed: {metadata['rows']:,} rows", flush=True)
    print(f"Row groups: {metadata['row_groups']:,}", flush=True)
    print(f"Size: {metadata['size_bytes']:,} bytes", flush=True)
    print(f"SHA-256: {metadata['sha256']}", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
