"""Build a query-disjoint mMARCO Vietnamese RAG/QA held-out benchmark.

The source rows are read with HTTP range requests from two mMARCO shards that
were not used by the local training database.  It intentionally does not
download the complete Hugging Face dataset or corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "benchmark" / "mmarco_vi_shardholdout_04_05"
DEFAULT_TRAIN_DB = PROJECT_DIR / "database" / "triplet_clean.db"
DATASET_ID = "minhnguyent546/mmarco-vietnamese-split"
# Pin the source revision so the benchmark can be reproduced if the dataset changes.
DEFAULT_REVISION = "afa7c5d9553883fc5a8dca2645ea86b95f0984f8"
DEFAULT_SHARDS = (
    "train-00004-of-00080.parquet",
    "train-00005-of-00080.parquet",
)

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize text for exact duplicate and leakage detection."""
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", html.unescape(value)).strip())


def text_key(value: str) -> bytes:
    """Return a compact, deterministic key for normalized text."""
    return hashlib.sha256(normalize_text(value).encode("utf-8")).digest()


def valid_triple(query: str, positive: str, negative: str) -> bool:
    """Keep the same basic quality envelope as the mMARCO train ingestion."""
    query = normalize_text(query)
    positive = normalize_text(positive)
    negative = normalize_text(negative)
    if not (8 <= len(query) <= 384):
        return False
    if not (80 <= len(positive) <= 2_000 and 80 <= len(negative) <= 2_000):
        return False
    return len({query, positive, negative}) == 3


def load_training_keys(train_db: Path) -> tuple[set[bytes], set[bytes]]:
    """Load exact normalized query/document fingerprints already used for training."""
    if not train_db.exists():
        raise FileNotFoundError(f"Training DB not found: {train_db}")

    query_keys: set[bytes] = set()
    document_keys: set[bytes] = set()
    with sqlite3.connect(f"file:{train_db}?mode=ro", uri=True) as conn:
        cursor = conn.execute("SELECT anchor, positive, hard_negative FROM triplet")
        for anchor, positive, hard_negative in cursor:
            query_keys.add(text_key(anchor))
            document_keys.add(text_key(positive))
            document_keys.add(text_key(hard_negative))
    return query_keys, document_keys


def evenly_spaced_indices(count: int, requested: int) -> list[int]:
    """Choose a small, deterministic spread of Parquet row groups."""
    if count < 1:
        return []
    requested = min(count, requested)
    if requested == 1:
        return [count // 2]
    return sorted({round(index * (count - 1) / (requested - 1)) for index in range(requested)})


def row_rank(query: str, positive: str, negative: str) -> bytes:
    """Use a stable pseudo-random rank, avoiding positional sampling bias."""
    return hashlib.sha256(f"{query}\x1f{positive}\x1f{negative}".encode("utf-8")).digest()


def iter_selected_rows(
    shard: str,
    revision: str,
    requested_rows: int,
    row_groups: int,
    train_query_keys: set[bytes],
    used_query_keys: set[bytes],
) -> Iterator[tuple[str, str, str, str]]:
    """Read selected remote row groups and yield query-disjoint triples.

    Each row group is fetched independently.  This keeps the source download
    bounded even though a shard itself contains nearly half a million rows.
    """
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/triples/{shard}"
    http_fs = fsspec.filesystem("https", block_size=2**20)
    with http_fs.open(url, "rb") as source:
        parquet_file = pq.ParquetFile(source)
        selected_groups = evenly_spaced_indices(parquet_file.num_row_groups, row_groups)
        quotas = [requested_rows // len(selected_groups)] * len(selected_groups)
        for index in range(requested_rows % len(selected_groups)):
            quotas[index] += 1

        for row_group, quota in zip(selected_groups, quotas, strict=True):
            table = parquet_file.read_row_group(
                row_group,
                columns=["query", "positive", "negative"],
            )
            candidates: list[tuple[bytes, str, str, str]] = []
            for query, positive, negative in zip(
                table.column("query").to_pylist(),
                table.column("positive").to_pylist(),
                table.column("negative").to_pylist(),
                strict=True,
            ):
                if not valid_triple(query, positive, negative):
                    continue
                query = normalize_text(query)
                positive = normalize_text(positive)
                negative = normalize_text(negative)
                query_key = text_key(query)
                if query_key in train_query_keys or query_key in used_query_keys:
                    continue
                candidates.append((row_rank(query, positive, negative), query, positive, negative))

            selected_count = 0
            for _, query, positive, negative in sorted(candidates):
                query_key = text_key(query)
                if query_key in used_query_keys:
                    continue
                used_query_keys.add(query_key)
                yield shard, query, positive, negative
                selected_count += 1
                if selected_count == quota:
                    break


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def build_artifact(
    output_dir: Path,
    train_db: Path,
    revision: str,
    shards: Iterable[str],
    rows_per_shard: int,
    row_groups_per_shard: int,
) -> dict[str, object]:
    """Create Parquet queries/corpus/qrels plus a reproducibility manifest."""
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Choose a new --output path to avoid overwriting it."
        )
    if rows_per_shard < 1 or row_groups_per_shard < 1:
        raise ValueError("--rows-per-shard and --row-groups-per-shard must both be positive")

    train_query_keys, train_document_keys = load_training_keys(train_db)
    used_query_keys: set[bytes] = set()
    queries: list[dict[str, str]] = []
    corpus: dict[str, dict[str, str]] = {}
    qrels: list[dict[str, object]] = []
    shard_counts: Counter[str] = Counter()

    for shard in shards:
        for source_shard, query, positive, negative in iter_selected_rows(
            shard=shard,
            revision=revision,
            requested_rows=rows_per_shard,
            row_groups=row_groups_per_shard,
            train_query_keys=train_query_keys,
            used_query_keys=used_query_keys,
        ):
            query_id = stable_id("q", query)
            positive_id = stable_id("d", positive)
            negative_id = stable_id("d", negative)
            queries.append({"id": query_id, "text": query, "source_shard": source_shard})
            corpus[positive_id] = {"id": positive_id, "title": "", "text": positive}
            corpus[negative_id] = {"id": negative_id, "title": "", "text": negative}
            qrels.append({"query_id": query_id, "corpus_id": positive_id, "score": 1})
            shard_counts[source_shard] += 1

    expected = rows_per_shard * len(tuple(shards))
    minimum_acceptable = (expected * 95 + 99) // 100
    if len(queries) < minimum_acceptable:
        raise RuntimeError(
            f"Only selected {len(queries):,}/{expected:,} requested rows; "
            f"the minimum acceptable count is {minimum_acceptable:,}. "
            "Increase --row-groups-per-shard or use a smaller --rows-per-shard."
        )
    query_ids = {row["id"] for row in queries}
    corpus_ids = set(corpus)
    if len(query_ids) != len(queries) or len(qrels) != len(queries):
        raise RuntimeError("Query IDs or qrels are not one-to-one")
    if not all(row["query_id"] in query_ids and row["corpus_id"] in corpus_ids for row in qrels):
        raise RuntimeError("A qrel refers to a missing query or corpus document")
    corpus_documents_seen_in_train = sum(
        text_key(row["text"]) in train_document_keys for row in corpus.values()
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temp_dir:
        temp_path = Path(temp_dir)
        queries_path = temp_path / "queries.parquet"
        corpus_path = temp_path / "corpus.parquet"
        qrels_path = temp_path / "qrels.parquet"
        pq.write_table(pa.Table.from_pylist(queries), queries_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(list(corpus.values())), corpus_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(qrels), qrels_path, compression="zstd")
        file_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (queries_path, corpus_path, qrels_path)
        }
        manifest = {
            "name": "mMARCO Vietnamese shard holdout 04-05",
            "kind": "in-domain held-out RAG/QA diagnostic benchmark",
            "not_an_official_vn_mteb_task": True,
            "source_dataset": DATASET_ID,
            "source_revision": revision,
            "source_shards": list(shards),
            "excluded_training_shards": [
                "train-00000-of-00080.parquet",
                "train-00001-of-00080.parquet",
                "train-00002-of-00080.parquet",
                "train-00003-of-00080.parquet",
            ],
            "selection": {
                "rows_per_shard": rows_per_shard,
                "requested_records": expected,
                "selected_records": len(queries),
                "row_groups_per_shard": row_groups_per_shard,
                "sampling": "deterministic hash-ranked sample from evenly spaced remote row groups",
                "quality_filter": "query 8-384 chars; positive/negative 80-2000 chars; three distinct normalized texts",
                "leakage_filter": "reject every exact normalized query text seen in triplet_clean.db",
                "corpus_policy": "documents may overlap the training corpus, as in standard corpus-shared retrieval splits",
            },
            "counts": {
                "queries": len(queries),
                "corpus_documents": len(corpus),
                "qrels": len(qrels),
                "records_by_shard": dict(sorted(shard_counts.items())),
                "corpus_documents_seen_in_train": corpus_documents_seen_in_train,
                "corpus_document_overlap_rate": round(
                    corpus_documents_seen_in_train / len(corpus), 6
                ),
            },
            "schema": {
                "queries.parquet": ["id", "text", "source_shard"],
                "corpus.parquet": ["id", "title", "text"],
                "qrels.parquet": ["query_id", "corpus_id", "score"],
            },
            "artifact_sha256": file_hashes,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        (temp_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(temp_path), str(output_dir))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-db", type=Path, default=DEFAULT_TRAIN_DB)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--shards", nargs="+", default=list(DEFAULT_SHARDS))
    parser.add_argument("--rows-per-shard", type=int, default=2_350)
    parser.add_argument("--row-groups-per-shard", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_artifact(
        output_dir=args.output.resolve(),
        train_db=args.train_db.resolve(),
        revision=args.revision,
        shards=tuple(args.shards),
        rows_per_shard=args.rows_per_shard,
        row_groups_per_shard=args.row_groups_per_shard,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote benchmark artifact to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
