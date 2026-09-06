import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator import science_data_generator as generator  # noqa: E402

REQUIRED_FIELDS = (
    "source",
    "title",
    "topic",
    "anchor",
    "positive",
    "hard_negative",
)

DEFAULT_INPUT_DIRS = (
    Path("/data/Science/generated_qa_v2"),
    Path("/data/Science/generated_qa"),
)
DEFAULT_OUTPUT_FILE = Path(
    "/data/Science/merged_high_quality/science_qa_merged.jsonl"
)

OLD_SOURCES = {
    "physics_qa.jsonl": (
        "OpenStax University Physics | https://openstax.org/details/books/university-physics-volume-1",
        "NIST Physical Measurement Laboratory | https://www.nist.gov/pml",
        "NASA Science | https://science.nasa.gov/",
    ),
    "biology_qa.jsonl": (
        "OpenStax Biology 2e | https://openstax.org/details/books/biology-2e",
        "NCBI Bookshelf | https://www.ncbi.nlm.nih.gov/books/",
        "HHMI BioInteractive | https://www.biointeractive.org/",
    ),
    "information_technology_qa.jsonl": (
        "NIST Computer Security Resource Center | https://csrc.nist.gov/",
        "IETF RFC Editor | https://www.rfc-editor.org/",
        "ACM Digital Library | https://dl.acm.org/",
    ),
    "chemistry_qa.jsonl": (
        "OpenStax Chemistry 2e | https://openstax.org/details/books/chemistry-2e",
        "IUPAC Gold Book | https://goldbook.iupac.org/",
        "NIST Chemistry WebBook | https://webbook.nist.gov/chemistry/",
    ),
    "mathematics_qa.jsonl": (
        "OpenStax Mathematics | https://openstax.org/subjects/math",
        "NIST Digital Library of Mathematical Functions | https://dlmf.nist.gov/",
        "Encyclopedia of Mathematics | https://encyclopediaofmath.org/",
    ),
    "astronomy_qa.jsonl": (
        "NASA Science | https://science.nasa.gov/",
        "ESA Science and Technology | https://sci.esa.int/",
        "OpenStax Astronomy 2e | https://openstax.org/details/books/astronomy-2e",
    ),
    "earth_science_qa.jsonl": (
        "USGS Science | https://www.usgs.gov/science",
        "NOAA | https://www.noaa.gov/",
        "NASA Earth Science | https://science.nasa.gov/earth/",
    ),
}

OLD_TOPICS = {
    "physics_qa.jsonl": (
        "cơ học",
        "nhiệt học",
        "điện và từ",
        "quang học",
        "sóng",
        "vật lí hiện đại",
    ),
    "biology_qa.jsonl": (
        "sinh học tế bào",
        "di truyền học",
        "sinh thái học",
        "tiến hóa",
        "sinh lí học",
        "vi sinh vật học",
    ),
    "information_technology_qa.jsonl": (
        "cấu trúc dữ liệu và giải thuật",
        "hệ điều hành",
        "mạng máy tính",
        "cơ sở dữ liệu",
        "an toàn thông tin",
        "kiến trúc máy tính",
    ),
    "chemistry_qa.jsonl": (
        "cấu tạo nguyên tử",
        "liên kết hóa học",
        "nhiệt động hóa học",
        "động học hóa học",
        "cân bằng hóa học",
        "hóa học hữu cơ",
    ),
    "mathematics_qa.jsonl": (
        "đại số",
        "giải tích",
        "hình học",
        "xác suất và thống kê",
        "toán rời rạc",
        "đại số tuyến tính",
    ),
    "astronomy_qa.jsonl": (
        "Hệ Mặt Trời",
        "sao và tiến hóa sao",
        "thiên hà",
        "vũ trụ học",
        "quan sát thiên văn",
    ),
    "earth_science_qa.jsonl": (
        "địa chất học",
        "khí tượng học",
        "hải dương học",
        "khí hậu học",
        "kiến tạo mảng",
        "chu trình địa hóa",
    ),
}

CURRENT_CONFIGS = {
    "physics_qa.jsonl": generator.PHYSICS,
    "biology_qa.jsonl": generator.BIOLOGY,
    "information_technology_qa.jsonl": generator.INFORMATION_TECHNOLOGY,
    "chemistry_qa.jsonl": generator.CHEMISTRY,
    "mathematics_qa.jsonl": generator.MATHEMATICS,
    "astronomy_qa.jsonl": generator.ASTRONOMY,
    "earth_science_qa.jsonl": generator.EARTH_SCIENCE,
}

DOMAIN_LABELS = {
    "physics_qa.jsonl": "Physics",
    "biology_qa.jsonl": "Biology",
    "information_technology_qa.jsonl": "Information Technology",
    "chemistry_qa.jsonl": "Chemistry",
    "mathematics_qa.jsonl": "Math",
    "astronomy_qa.jsonl": "Astronomy",
    "earth_science_qa.jsonl": "Earth Science",
}


@dataclass
class Candidate:
    document: dict[str, str]
    origin: str


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller = min(left_root, right_root)
        larger = max(left_root, right_root)
        self.parent[larger] = smaller


def normalize_whitespace(text: str) -> str:
    """Normalize Unicode and whitespace while preserving readable content."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def text_key(text: str) -> str:
    """Create a punctuation-insensitive key for exact deduplication."""
    normalized = normalize_whitespace(text).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return len([part for part in parts if part.strip()])


def build_allowed_metadata(
    filename: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    config = CURRENT_CONFIGS[filename]
    sources = tuple(dict.fromkeys((*config.sources, *OLD_SOURCES[filename])))
    topics = tuple(dict.fromkeys((*config.topics, *OLD_TOPICS[filename])))
    return sources, topics


def canonicalize_source(source: str, allowed_sources: tuple[str, ...]) -> str | None:
    candidate = text_key(source).rstrip("/")
    for canonical in allowed_sources:
        source_name, separator, source_url = canonical.partition(" | ")
        variants = {
            text_key(canonical).rstrip("/"),
            text_key(source_name).rstrip("/"),
        }
        if separator:
            variants.add(text_key(source_url).rstrip("/"))
        if candidate in variants:
            return canonical
        if text_key(source_name) in candidate:
            return canonical
        if separator and text_key(source_url) in candidate:
            return canonical
    return None


def canonicalize_topic(topic: str, allowed_topics: tuple[str, ...]) -> str | None:
    topic_map = {text_key(value): value for value in allowed_topics}
    candidate = text_key(topic)
    if candidate in topic_map:
        return topic_map[candidate]

    matches = get_close_matches(candidate, topic_map, n=1, cutoff=0.86)
    if not matches:
        return None
    return topic_map[matches[0]]


def normalize_document(
    raw_document: object,
    allowed_sources: tuple[str, ...],
    allowed_topics: tuple[str, ...],
) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(raw_document, dict):
        return None, "not_object"
    if any(field not in raw_document for field in REQUIRED_FIELDS):
        return None, "missing_field"

    document: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        value = raw_document[field]
        if not isinstance(value, str) or not value.strip():
            return None, "empty_field"
        document[field] = normalize_whitespace(value)

    source = canonicalize_source(document["source"], allowed_sources)
    if source is None:
        return None, "unknown_source"
    document["source"] = source

    topic = canonicalize_topic(document["topic"], allowed_topics)
    if topic is None:
        return None, "unknown_topic"
    document["topic"] = topic

    anchor_words = len(text_key(document["anchor"]).split())
    positive_words = len(text_key(document["positive"]).split())
    negative_words = len(text_key(document["hard_negative"]).split())
    if not 5 <= anchor_words <= 80:
        return None, "anchor_length"
    if not 20 <= positive_words <= 220:
        return None, "positive_length"
    if not 20 <= negative_words <= 220:
        return None, "negative_length"

    positive_key = text_key(document["positive"])
    negative_key = text_key(document["hard_negative"])
    if positive_key == negative_key:
        return None, "identical_passages"

    positive_words_set = set(positive_key.split())
    negative_words_set = set(negative_key.split())
    union = positive_words_set | negative_words_set
    overlap = len(positive_words_set & negative_words_set) / len(union)
    if overlap >= 0.80:
        return None, "passages_too_similar"

    if not 1 <= sentence_count(document["positive"]) <= 6:
        return None, "positive_sentence_count"
    if not 1 <= sentence_count(document["hard_negative"]) <= 6:
        return None, "negative_sentence_count"

    return document, None


def load_and_exact_deduplicate(
    input_dirs: tuple[Path, ...],
) -> tuple[dict[str, list[Candidate]], Counter[str]]:
    candidates_by_domain: dict[str, list[Candidate]] = defaultdict(list)
    stats: Counter[str] = Counter()
    seen_anchors: set[str] = set()
    seen_positives: set[str] = set()
    seen_negatives: set[str] = set()

    for input_dir in input_dirs:
        origin = input_dir.name
        for filename in CURRENT_CONFIGS:
            path = input_dir / filename
            if not path.exists():
                stats["missing_files"] += 1
                continue

            allowed_sources, allowed_topics = build_allowed_metadata(filename)
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    stats["input_records"] += 1
                    try:
                        raw_document = json.loads(line)
                    except json.JSONDecodeError:
                        stats["invalid_json"] += 1
                        continue

                    document, rejection_reason = normalize_document(
                        raw_document,
                        allowed_sources,
                        allowed_topics,
                    )
                    if document is None:
                        stats[f"rejected_{rejection_reason}"] += 1
                        continue

                    document["domain"] = DOMAIN_LABELS[filename]
                    del document["source"]

                    anchor = text_key(document["anchor"])
                    positive = text_key(document["positive"])
                    negative = text_key(document["hard_negative"])
                    if anchor in seen_anchors:
                        stats["duplicate_anchor"] += 1
                        continue
                    if positive in seen_positives:
                        stats["duplicate_positive"] += 1
                        continue
                    if negative in seen_negatives:
                        stats["duplicate_negative"] += 1
                        continue

                    seen_anchors.add(anchor)
                    seen_positives.add(positive)
                    seen_negatives.add(negative)
                    candidates_by_domain[filename].append(
                        Candidate(document=document, origin=origin)
                    )
                    stats["exact_unique_records"] += 1
                    stats[f"kept_from_{origin}"] += 1

    return candidates_by_domain, stats


def semantic_deduplicate(
    candidates: list[Candidate],
    threshold: float,
    neighbor_count: int,
) -> tuple[list[Candidate], int]:
    if len(candidates) < 2:
        return candidates, 0

    anchors = [candidate.document["anchor"] for candidate in candidates]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        dtype=np.float32,
    )
    vectors = vectorizer.fit_transform(anchors)
    neighbors = min(neighbor_count, len(candidates))
    distances, indices = NearestNeighbors(
        n_neighbors=neighbors,
        metric="cosine",
        n_jobs=-1,
    ).fit(vectors).kneighbors(vectors)

    groups = UnionFind(len(candidates))
    for left_index in range(len(candidates)):
        for distance, right_index in zip(
            distances[left_index, 1:],
            indices[left_index, 1:],
        ):
            if 1.0 - float(distance) < threshold:
                break
            groups.union(left_index, int(right_index))

    kept = [
        candidate
        for index, candidate in enumerate(candidates)
        if groups.find(index) == index
    ]
    return kept, len(candidates) - len(kept)


def merge_science_data(
    input_dirs: tuple[Path, ...],
    output_file: Path,
    semantic_threshold: float,
    neighbor_count: int,
    random_seed: int,
    overwrite: bool,
) -> dict[str, object]:
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"{output_file} already exists; pass --overwrite to replace it"
        )

    candidates_by_domain, stats = load_and_exact_deduplicate(input_dirs)
    final_candidates: list[Candidate] = []
    domain_report: dict[str, dict[str, int]] = {}

    for filename, candidates in candidates_by_domain.items():
        print(f"Semantic deduplication: {filename} ({len(candidates)} records)")
        kept, removed = semantic_deduplicate(
            candidates,
            threshold=semantic_threshold,
            neighbor_count=neighbor_count,
        )
        final_candidates.extend(kept)
        domain_report[filename] = {
            "before_semantic_deduplication": len(candidates),
            "semantic_duplicates_removed": removed,
            "final_records": len(kept),
        }
        stats["semantic_duplicates_removed"] += removed

    random.Random(random_seed).shuffle(final_candidates)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        for candidate in final_candidates:
            file.write(json.dumps(candidate.document, ensure_ascii=False) + "\n")
    temporary_file.replace(output_file)

    stats["final_records"] = len(final_candidates)
    report: dict[str, object] = {
        "input_directories": [str(path) for path in input_dirs],
        "output_file": str(output_file),
        "semantic_threshold": semantic_threshold,
        "semantic_neighbor_count": neighbor_count,
        "random_seed": random_seed,
        "stats": dict(stats),
        "domains": domain_report,
    }
    report_file = output_file.with_name("merge_report.json")
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and deduplicate generated Vietnamese science data."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        dest="input_dirs",
        help="Input directory. Repeat this option to add multiple directories.",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--semantic-threshold", type=float, default=0.95)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.semantic_threshold <= 1.0:
        raise ValueError("semantic-threshold must be in the range (0, 1]")
    if args.neighbors < 2:
        raise ValueError("neighbors must be at least 2")

    input_dirs = tuple(args.input_dirs or DEFAULT_INPUT_DIRS)
    missing_dirs = [path for path in input_dirs if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing input directories: {missing_dirs}")

    report = merge_science_data(
        input_dirs=input_dirs,
        output_file=args.output_file,
        semantic_threshold=args.semantic_threshold,
        neighbor_count=args.neighbors,
        random_seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
