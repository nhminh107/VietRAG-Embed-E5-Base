import argparse
import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

from database.models import GeneralModel, GeneralTriplet
from database.sql_manager import SQL_Manager


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "checkpoint-40236"
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "general_hard_negative"
INDEX_PATH = ARTIFACT_DIR / "general.faiss"
DATA_IDS_PATH = ARTIFACT_DIR / "general_data_ids.npy"
METADATA_PATH = ARTIFACT_DIR / "general_index_metadata.json"

EMBEDDING_BATCH_SIZE = 128
DATABASE_BATCH_SIZE = 256
INDEX_TRAINING_SAMPLE_SIZE = 200_000
MIN_TRAINING_SAMPLE_SIZE = 50_000
INDEX_NLIST = 4_096
INDEX_PQ_M = 64
INDEX_NPROBE = 32
SEARCH_TOP_K = 16
PROGRESS_EVERY = 10_000


def get_device() -> str:
    """Use CUDA when PyTorch can access it, otherwise use CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> SentenceTransformer:
    """Load the local fine-tuned embedding model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_PATH}")

    device = get_device()
    print(f"Loading model on: {device}")
    return SentenceTransformer(str(MODEL_PATH), device=device, local_files_only=True)


def encode_passages(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode document text with the E5 passage prefix."""
    embeddings = model.encode(
        [f"passage: {value}" for value in texts],
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_queries(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode query text with the E5 query prefix."""
    embeddings = model.encode(
        [f"query: {value}" for value in texts],
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def valid_general_filter() -> list[Any]:
    """Return filters for rows that can form a retrieval pair."""
    return [
        GeneralModel.anchor.is_not(None),
        GeneralModel.positive.is_not(None),
        func.length(func.trim(GeneralModel.anchor)) > 0,
        func.length(func.trim(GeneralModel.positive)) > 0,
    ]


def get_source_stats(sql_mng: SQL_Manager) -> tuple[int, int]:
    """Return the number of valid source rows and the maximum data ID length."""
    count_statement = select(func.count()).select_from(GeneralModel).where(
        *valid_general_filter()
    )
    id_length_statement = select(func.max(func.length(GeneralModel.data_id))).where(
        *valid_general_filter()
    )

    source_count = sql_mng.con.scalar(count_statement)
    max_data_id_length = sql_mng.con.scalar(id_length_statement)
    if not source_count or not max_data_id_length:
        raise ValueError("No valid rows were found in the general table.")

    return int(source_count), int(max_data_id_length)


def fetch_general_batch(
    sql_mng: SQL_Manager,
    last_data_id: str | None,
    include_completed: bool,
) -> list[dict[str, Any]]:
    """Read a stable, keyset-paginated batch from the general table."""
    statement = select(
        GeneralModel.data_id.label("data_id"),
        GeneralModel.source.label("source"),
        GeneralModel.title.label("title"),
        GeneralModel.topic.label("topic"),
        GeneralModel.anchor.label("anchor"),
        GeneralModel.positive.label("positive"),
    ).where(*valid_general_filter())

    if not include_completed:
        statement = statement.outerjoin(
            GeneralTriplet,
            GeneralTriplet.data_id == GeneralModel.data_id,
        ).where(GeneralTriplet.data_id.is_(None))

    if last_data_id is not None:
        statement = statement.where(GeneralModel.data_id > last_data_id)

    rows = sql_mng.con.execute(
        statement.order_by(GeneralModel.data_id).limit(DATABASE_BATCH_SIZE)
    ).mappings()
    return [dict(row) for row in rows]


def get_training_texts(sql_mng: SQL_Manager, source_count: int) -> list[str]:
    """Sample positives from random PostgreSQL pages to train the IVF-PQ index."""
    sampling_percent = min(
        100.0,
        max(5.0, 200.0 * INDEX_TRAINING_SAMPLE_SIZE / source_count),
    )
    statement = text(
        "SELECT positive "
        "FROM general TABLESAMPLE SYSTEM "
        f"({sampling_percent:.4f}) "
        "WHERE anchor IS NOT NULL "
        "AND positive IS NOT NULL "
        "AND length(trim(anchor)) > 0 "
        "AND length(trim(positive)) > 0 "
        "LIMIT :sample_size"
    )
    rows = sql_mng.con.execute(
        statement,
        {"sample_size": INDEX_TRAINING_SAMPLE_SIZE},
    )
    texts = [row[0] for row in rows]

    if len(texts) < MIN_TRAINING_SAMPLE_SIZE:
        raise RuntimeError(
            "FAISS training sample is too small "
            f"({len(texts):,} rows). Run the script again or increase the sample rate."
        )

    return texts


def expected_metadata(
    source_count: int,
    max_data_id_length: int,
    embedding_dimension: int,
) -> dict[str, int | str]:
    """Describe the source and index settings required to reuse an artifact."""
    return {
        "source_count": source_count,
        "max_data_id_length": max_data_id_length,
        "embedding_dimension": embedding_dimension,
        "index_nlist": INDEX_NLIST,
        "index_pq_m": INDEX_PQ_M,
        "index_nprobe": INDEX_NPROBE,
        "model_path": str(MODEL_PATH),
    }


def artifact_paths() -> tuple[Path, ...]:
    return INDEX_PATH, DATA_IDS_PATH, METADATA_PATH


def building_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.building")


def remove_index_artifacts() -> None:
    """Remove only known index artifacts after an explicit rebuild request."""
    paths = (*artifact_paths(), *(building_path(path) for path in artifact_paths()))
    for path in paths:
        if path.exists():
            path.unlink()


def load_saved_index(
    metadata: dict[str, int | str],
) -> tuple[faiss.Index, np.memmap] | None:
    """Load a complete matching index artifact, or return None when absent."""
    files_exist = [path.exists() for path in artifact_paths()]
    building_files = [
        building_path(path)
        for path in artifact_paths()
        if building_path(path).exists()
    ]

    if building_files:
        names = ", ".join(str(path) for path in building_files)
        raise RuntimeError(
            "An interrupted index build was found. Run with --rebuild-index to replace: "
            f"{names}"
        )

    if not any(files_exist):
        return None
    if not all(files_exist):
        raise RuntimeError(
            "Index artifacts are incomplete. Run with --rebuild-index to recreate them."
        )

    saved_metadata = json.loads(METADATA_PATH.read_text())
    if saved_metadata != metadata:
        raise RuntimeError(
            "The saved index does not match the current model or general table. "
            "Run with --rebuild-index."
        )

    index = faiss.read_index(str(INDEX_PATH))
    if hasattr(index, "nprobe"):
        index.nprobe = INDEX_NPROBE

    data_ids = np.load(DATA_IDS_PATH, mmap_mode="r")
    if index.ntotal != metadata["source_count"] or len(data_ids) != index.ntotal:
        raise RuntimeError(
            "The saved FAISS index and data ID map have different sizes. "
            "Run with --rebuild-index."
        )

    print(f"Loaded existing FAISS index with {index.ntotal:,} vectors.")
    return index, data_ids


def build_index(
    sql_mng: SQL_Manager,
    model: SentenceTransformer,
    metadata: dict[str, int | str],
) -> tuple[faiss.Index, np.memmap]:
    """Build and persist an IVF-PQ index over every valid general positive."""
    source_count = int(metadata["source_count"])
    max_data_id_length = int(metadata["max_data_id_length"])
    embedding_dimension = int(metadata["embedding_dimension"])

    if source_count < MIN_TRAINING_SAMPLE_SIZE:
        raise ValueError(
            f"At least {MIN_TRAINING_SAMPLE_SIZE:,} source rows are required for IVF-PQ."
        )
    if embedding_dimension % INDEX_PQ_M != 0:
        raise ValueError(
            f"Embedding dimension {embedding_dimension} is not divisible by PQ M={INDEX_PQ_M}."
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    index_path_building = building_path(INDEX_PATH)
    data_ids_path_building = building_path(DATA_IDS_PATH)
    metadata_path_building = building_path(METADATA_PATH)

    if any(
        path.exists()
        for path in (index_path_building, data_ids_path_building, metadata_path_building)
    ):
        raise RuntimeError(
            "Index build files already exist. Run with --rebuild-index to replace them."
        )

    print("Sampling positive passages to train FAISS IVF-PQ...")
    training_texts = get_training_texts(sql_mng, source_count)
    training_embeddings = encode_passages(model, training_texts)
    print(f"Training FAISS with {len(training_embeddings):,} passage embeddings...")

    quantizer = faiss.IndexFlatIP(embedding_dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        embedding_dimension,
        INDEX_NLIST,
        INDEX_PQ_M,
        8,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.train(training_embeddings)
    index.nprobe = INDEX_NPROBE

    data_ids = np.lib.format.open_memmap(
        data_ids_path_building,
        mode="w+",
        dtype=f"S{max_data_id_length}",
        shape=(source_count,),
    )

    last_data_id: str | None = None
    offset = 0
    while True:
        rows = fetch_general_batch(sql_mng, last_data_id, include_completed=True)
        if not rows:
            break

        positives = [row["positive"] for row in rows]
        embeddings = encode_passages(model, positives)
        index.add(embeddings)
        data_ids[offset : offset + len(rows)] = [
            row["data_id"].encode("utf-8") for row in rows
        ]

        offset += len(rows)
        last_data_id = rows[-1]["data_id"]

        if offset % PROGRESS_EVERY < len(rows):
            print(f"Indexed: {offset:,}/{source_count:,}")

    if offset != source_count:
        raise RuntimeError(
            f"Indexed {offset:,} rows, expected {source_count:,}. The general table changed during indexing."
        )

    data_ids.flush()
    faiss.write_index(index, str(index_path_building))
    metadata_path_building.write_text(json.dumps(metadata, indent=2))

    os.replace(data_ids_path_building, DATA_IDS_PATH)
    os.replace(index_path_building, INDEX_PATH)
    os.replace(metadata_path_building, METADATA_PATH)

    print(f"Saved FAISS index with {index.ntotal:,} vectors to {INDEX_PATH}")
    return index, np.load(DATA_IDS_PATH, mmap_mode="r")


def get_or_build_index(
    sql_mng: SQL_Manager,
    model: SentenceTransformer,
    rebuild_index: bool,
) -> tuple[faiss.Index, np.memmap]:
    """Load a compatible index or build it from the general table."""
    source_count, max_data_id_length = get_source_stats(sql_mng)
    embedding_dimension = model.get_embedding_dimension()
    metadata = expected_metadata(
        source_count,
        max_data_id_length,
        embedding_dimension,
    )

    if rebuild_index:
        print("Removing existing hard-negative index artifacts...")
        remove_index_artifacts()

    saved_index = load_saved_index(metadata)
    if saved_index is not None:
        return saved_index

    return build_index(sql_mng, model, metadata)


def get_data_id(data_ids: np.memmap, position: int) -> str:
    """Read one original general data ID from the FAISS position map."""
    return data_ids[position].tobytes().decode("utf-8").rstrip("\x00")


def fetch_candidate_positives(
    sql_mng: SQL_Manager,
    data_ids: set[str],
) -> dict[str, str]:
    """Fetch only candidate positives needed by the current FAISS search batch."""
    if not data_ids:
        return {}

    rows = sql_mng.con.execute(
        select(GeneralModel.data_id, GeneralModel.positive).where(
            GeneralModel.data_id.in_(data_ids)
        )
    )
    return {
        data_id: positive
        for data_id, positive in rows
        if positive is not None and positive.strip()
    }


def build_triplets(
    rows: list[dict[str, Any]],
    neighbor_positions: np.ndarray,
    data_ids: np.memmap,
    candidate_positives: dict[str, str],
) -> tuple[list[dict[str, Any]], int]:
    """Keep the nearest candidate whose passage differs from the source positive."""
    triplets = []
    skipped_without_negative = 0

    for row, positions in zip(rows, neighbor_positions, strict=True):
        hard_negative: str | None = None

        for position in positions:
            if position < 0:
                continue

            candidate_id = get_data_id(data_ids, int(position))
            candidate_positive = candidate_positives.get(candidate_id)
            if candidate_id == row["data_id"]:
                continue
            if candidate_positive is None or candidate_positive == row["positive"]:
                continue

            hard_negative = candidate_positive
            break

        if hard_negative is None:
            skipped_without_negative += 1
            continue

        triplets.append({**row, "hard_negative": hard_negative})

    return triplets, skipped_without_negative


def insert_triplets(sql_mng: SQL_Manager, triplets: list[dict[str, Any]]) -> int:
    """Insert a batch and keep already completed records unchanged on resume."""
    if not triplets:
        return 0

    statement = postgresql_insert(GeneralTriplet).values(triplets)
    statement = statement.on_conflict_do_nothing(
        index_elements=["data_id"]
    ).returning(GeneralTriplet.data_id)
    inserted_ids = sql_mng.con.scalars(statement).all()
    return len(inserted_ids)


def generate_hard_negatives(
    sql_mng: SQL_Manager,
    model: SentenceTransformer,
    index: faiss.Index,
    data_ids: np.memmap,
) -> None:
    """Create and store one deduplicated hard negative for every pending source row."""
    existing_triplets = sql_mng.con.scalar(
        select(func.count()).select_from(GeneralTriplet)
    )
    print(f"Existing GeneralTriplet records: {existing_triplets:,}")

    last_data_id: str | None = None
    processed = 0
    inserted = 0
    skipped_without_negative = 0

    while True:
        rows = fetch_general_batch(sql_mng, last_data_id, include_completed=False)
        if not rows:
            break

        query_embeddings = encode_queries(model, [row["anchor"] for row in rows])
        _, neighbor_positions = index.search(query_embeddings, SEARCH_TOP_K)

        candidate_ids = {
            get_data_id(data_ids, int(position))
            for positions in neighbor_positions
            for position in positions
            if position >= 0
        }
        candidate_positives = fetch_candidate_positives(sql_mng, candidate_ids)
        triplets, skipped = build_triplets(
            rows,
            neighbor_positions,
            data_ids,
            candidate_positives,
        )

        inserted += insert_triplets(sql_mng, triplets)
        sql_mng.con.commit()

        processed += len(rows)
        skipped_without_negative += skipped
        last_data_id = rows[-1]["data_id"]

        if processed % PROGRESS_EVERY < len(rows):
            print(
                f"Processed: {processed:,}; inserted: {inserted:,}; "
                f"without hard negative: {skipped_without_negative:,}"
            )

    print(
        f"Finished. Processed: {processed:,}; inserted: {inserted:,}; "
        f"without hard negative: {skipped_without_negative:,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GeneralTriplet hard negatives with a persistent FAISS IVF-PQ index."
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Delete only this script's saved FAISS artifacts and rebuild them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sql_mng = SQL_Manager()

    try:
        sql_mng.create_general_triplet()
        model = load_model()
        index, data_ids = get_or_build_index(sql_mng, model, args.rebuild_index)
        generate_hard_negatives(sql_mng, model, index, data_ids)
    except Exception:
        sql_mng.con.rollback()
        raise
    finally:
        sql_mng.close()


if __name__ == "__main__":
    main()
