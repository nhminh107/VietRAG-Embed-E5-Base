import asyncio
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://mkp-api.fptcloud.com/chat/completions"
MODEL_NAME = "gemma-4-26B-A4B-it"

REQUIRED_FIELDS = (
    "source",
    "title",
    "topic",
    "anchor",
    "positive",
    "hard_negative",
)
LEGACY_REQUIRED_FIELDS = set(REQUIRED_FIELDS) | {"answer"}

DEFAULT_TOTAL_DOCUMENTS = 350_000
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_CONCURRENCY = 14
DEFAULT_WORKERS_PER_DOMAIN = 2
DEFAULT_MAX_RETRIES = 5
DEFAULT_OUTPUT_DIR = Path("/data/Science/generated_qa_v2")

Document = dict[str, str]
DocumentGenerator = Callable[
    [int, str, set[str] | None, int | None],
    Awaitable[list[Document]],
]


@dataclass(frozen=True)
class DomainConfig:
    """Configuration used to generate one science domain."""

    name: str
    topics: tuple[str, ...]
    sources: tuple[str, ...]


PHYSICS = DomainConfig(
    name="vật lí",
    topics=(
        "động học chất điểm",
        "động lực học Newton",
        "cơ học chất lưu",
        "dao động và sóng",
        "nhiệt động lực học",
        "điện trường và điện thế",
        "mạch điện",
        "từ trường và cảm ứng điện từ",
        "quang hình học",
        "quang học sóng",
        "thuyết tương đối",
        "cơ học lượng tử",
        "vật lí nguyên tử",
        "vật lí hạt nhân",
    ),
    sources=(
        "OpenStax University Physics Volume 2 | https://openstax.org/details/books/university-physics-volume-2",
        "OpenStax University Physics Volume 3 | https://openstax.org/details/books/university-physics-volume-3",
        "CERN Physics | https://home.cern/science/physics",
        "IAEA Nuclear Physics | https://www.iaea.org/topics/nuclear-physics",
        "Physics LibreTexts | https://phys.libretexts.org/",
        "American Physical Society | https://www.aps.org/",
    ),
)

BIOLOGY = DomainConfig(
    name="sinh học",
    topics=(
        "sinh học tế bào",
        "sinh học phân tử",
        "di truyền học",
        "hệ gen học",
        "hóa sinh",
        "sinh thái học",
        "tiến hóa",
        "thực vật học",
        "động vật học",
        "miễn dịch học",
        "sinh học thần kinh",
        "sinh lí học",
        "vi sinh vật học",
    ),
    sources=(
        "EMBL-EBI Training | https://www.ebi.ac.uk/training/",
        "Genome.gov Genomics | https://www.genome.gov/about-genomics",
        "Nature Scitable | https://www.nature.com/scitable/",
        "Biology LibreTexts | https://bio.libretexts.org/",
        "Smithsonian Human Origins | https://humanorigins.si.edu/",
        "Encyclopedia of Life | https://eol.org/",
    ),
)

INFORMATION_TECHNOLOGY = DomainConfig(
    name="công nghệ thông tin và khoa học máy tính",
    topics=(
        "cấu trúc dữ liệu và giải thuật",
        "độ phức tạp tính toán",
        "hệ điều hành",
        "hệ thống phân tán",
        "mạng máy tính",
        "giao thức Internet",
        "cơ sở dữ liệu",
        "hệ quản trị cơ sở dữ liệu",
        "an toàn thông tin",
        "mật mã học",
        "kiến trúc máy tính",
        "ngôn ngữ lập trình",
    ),
    sources=(
        "OpenDSA | https://opendsa-server.cs.vt.edu/",
        "MIT OpenCourseWare Electrical Engineering and Computer Science | https://ocw.mit.edu/search/?d=Electrical%20Engineering%20and%20Computer%20Science",
        "PostgreSQL Documentation | https://www.postgresql.org/docs/",
        "Linux Kernel Documentation | https://docs.kernel.org/",
        "OWASP | https://owasp.org/",
        "MDN Web Docs | https://developer.mozilla.org/",
    ),
)

CHEMISTRY = DomainConfig(
    name="hóa học",
    topics=(
        "cấu tạo nguyên tử",
        "bảng tuần hoàn",
        "liên kết hóa học",
        "hóa học dung dịch",
        "hóa học axit-bazơ",
        "nhiệt động hóa học",
        "động học hóa học",
        "cân bằng hóa học",
        "hóa học hữu cơ",
        "hóa học vô cơ",
        "hóa phân tích",
        "hóa sinh",
    ),
    sources=(
        "PubChem | https://pubchem.ncbi.nlm.nih.gov/",
        "Chemistry LibreTexts | https://chem.libretexts.org/",
        "Royal Society of Chemistry | https://www.rsc.org/",
        "American Chemical Society | https://www.acs.org/",
        "IUPAC Periodic Table | https://iupac.org/what-we-do/periodic-table-of-elements/",
        "NIST Computational Chemistry Comparison and Benchmark Database | https://cccbdb.nist.gov/",
    ),
)

MATHEMATICS = DomainConfig(
    name="toán học",
    topics=(
        "đại số",
        "giải tích",
        "phương trình vi phân",
        "hình học",
        "tô pô",
        "xác suất và thống kê",
        "toán rời rạc",
        "lí thuyết số",
        "tối ưu hóa",
        "giải tích số",
        "đại số tuyến tính",
        "logic toán học",
    ),
    sources=(
        "MIT OpenCourseWare Mathematics | https://ocw.mit.edu/search/?d=Mathematics",
        "Mathematics LibreTexts | https://math.libretexts.org/",
        "American Mathematical Society | https://www.ams.org/",
        "Wolfram MathWorld | https://mathworld.wolfram.com/",
        "OEIS | https://oeis.org/",
        "NIST Digital Library of Mathematical Functions | https://dlmf.nist.gov/",
    ),
)

ASTRONOMY = DomainConfig(
    name="thiên văn học",
    topics=(
        "Hệ Mặt Trời",
        "hành tinh và vệ tinh",
        "tiểu hành tinh và sao chổi",
        "sao và tiến hóa sao",
        "ngoại hành tinh",
        "lỗ đen và sao neutron",
        "thiên hà",
        "vũ trụ học",
        "sóng hấp dẫn",
        "quan sát thiên văn",
    ),
    sources=(
        "NASA Jet Propulsion Laboratory | https://www.jpl.nasa.gov/",
        "European Southern Observatory | https://www.eso.org/public/science/",
        "HubbleSite Science | https://science.nasa.gov/mission/hubble/science/",
        "Chandra X-ray Observatory | https://chandra.harvard.edu/",
        "NASA Exoplanet Exploration | https://exoplanets.nasa.gov/",
        "LIGO Science | https://www.ligo.org/science/",
    ),
)

EARTH_SCIENCE = DomainConfig(
    name="khoa học Trái Đất",
    topics=(
        "địa chất học",
        "khoáng vật học",
        "núi lửa học",
        "địa chấn học",
        "khí tượng học",
        "hải dương học",
        "khí hậu học",
        "kiến tạo mảng",
        "chu trình địa hóa",
        "thủy văn học",
        "cổ khí hậu học",
    ),
    sources=(
        "IPCC Reports | https://www.ipcc.ch/reports/",
        "Smithsonian Global Volcanism Program | https://volcano.si.edu/",
        "UCAR Center for Science Education | https://scied.ucar.edu/",
        "Woods Hole Oceanographic Institution | https://www.whoi.edu/",
        "Earth Science LibreTexts | https://geo.libretexts.org/",
        "EPA Climate Change | https://www.epa.gov/climate-change",
    ),
)


def _remove_code_fence(content: str) -> str:
    """Remove a Markdown code fence sometimes added by the model."""
    content = content.strip()
    if not content.startswith("```"):
        return content

    lines = content.splitlines()
    if lines and lines[-1].strip() == "```":
        lines = lines[1:-1]
    else:
        lines = lines[1:]
    return "\n".join(lines).strip()


def _parse_json_documents(content: str) -> list[object]:
    """Parse a JSON array and salvage complete objects from a malformed response."""
    cleaned_content = _remove_code_fence(content)
    try:
        parsed = json.loads(cleaned_content)
        if not isinstance(parsed, list):
            raise ValueError("The model response must be a JSON array")
        return parsed
    except json.JSONDecodeError as original_error:
        documents: list[object] = []
        object_start: int | None = None
        object_depth = 0
        in_string = False
        escaped = False

        for index, character in enumerate(cleaned_content):
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue

            if character == '"':
                in_string = True
            elif character == "{":
                if object_depth == 0:
                    object_start = index
                object_depth += 1
            elif character == "}" and object_depth > 0:
                object_depth -= 1
                if object_depth == 0 and object_start is not None:
                    try:
                        documents.append(
                            json.loads(cleaned_content[object_start : index + 1])
                        )
                    except json.JSONDecodeError:
                        pass
                    object_start = None

        if not documents:
            raise original_error

        print(
            "Warning: malformed model JSON; salvaged "
            f"{len(documents)} complete records"
        )
        return documents


def _normalize_source(source: str, trusted_sources: tuple[str, ...]) -> str:
    """Normalize common source formats without rejecting an unknown source."""
    candidate = source.strip()
    normalized_candidate = candidate.casefold().rstrip("/")

    for trusted_source in trusted_sources:
        source_name, separator, source_url = trusted_source.partition(" | ")
        accepted_values = {
            trusted_source.casefold().rstrip("/"),
            source_name.casefold().rstrip("/"),
        }
        if separator:
            accepted_values.add(source_url.casefold().rstrip("/"))

        if normalized_candidate in accepted_values:
            return trusted_source
        if source_name.casefold() in normalized_candidate:
            return trusted_source
        if separator and source_url.casefold().rstrip("/") in normalized_candidate:
            return trusted_source

    return candidate


def _anchor_key(anchor: str) -> str:
    """Create a stable key for exact-question deduplication."""
    return " ".join(anchor.casefold().split()).rstrip("?.!")


def _filter_unique_documents(
    documents: list[Document],
    known_anchors: set[str] | None,
) -> list[Document]:
    """Drop duplicate questions without rejecting the whole generated batch."""
    seen_anchors = set(known_anchors or ())
    unique_documents: list[Document] = []

    for document in documents:
        anchor = _anchor_key(document["anchor"])
        if anchor in seen_anchors:
            continue
        seen_anchors.add(anchor)
        unique_documents.append(document)

    return unique_documents


def _validate_documents(
    documents: object,
    expected_count: int,
    config: DomainConfig,
) -> list[Document]:
    """Validate the generated retrieval records before writing them."""
    if not isinstance(documents, list):
        raise ValueError("The model response must be a JSON array")
    required_fields = set(REQUIRED_FIELDS)
    valid_documents: list[Document] = []
    for document in documents[:expected_count]:
        if not isinstance(document, dict):
            continue
        if not required_fields.issubset(document):
            continue
        for field in REQUIRED_FIELDS:
            if not isinstance(document[field], str) or not document[field].strip():
                break
        else:
            normalized_document = {
                field: document[field].strip() for field in REQUIRED_FIELDS
            }
            normalized_document["source"] = _normalize_source(
                normalized_document["source"],
                config.sources,
            )
            valid_documents.append(normalized_document)

    if not valid_documents:
        raise ValueError("The model did not return any complete records")
    return valid_documents


def _validate_existing_document(document: object, path: Path, index: int) -> Document:
    """Validate one stored record and return it with a precise type."""
    required_fields = set(REQUIRED_FIELDS)
    if not isinstance(document, dict) or set(document) not in (
        required_fields,
        LEGACY_REQUIRED_FIELDS,
    ):
        raise ValueError(f"{path} contains an incompatible record at index {index}")
    if any(
        not isinstance(document[field], str) or not document[field].strip()
        for field in document
    ):
        raise ValueError(f"{path} contains an invalid record at index {index}")
    return document


def _load_generation_state(output_file: str) -> tuple[int, set[str]]:
    """Load the current record count and anchors so generation can resume."""
    path = Path(output_file)
    if not path.exists():
        return 0, set()

    anchors: set[str] = set()
    count = 0

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if not line.strip():
                    continue
                document = _validate_existing_document(
                    json.loads(line),
                    path,
                    index,
                )
                anchor = _anchor_key(document["anchor"])
                anchors.add(anchor)
                count += 1
        return count, anchors

    documents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise ValueError(f"{path} must contain a JSON array")

    for index, item in enumerate(documents):
        document = _validate_existing_document(item, path, index)
        anchor = _anchor_key(document["anchor"])
        anchors.add(anchor)

    return len(documents), anchors


def _append_to_json(
    output_file: str,
    new_documents: list[Document],
    known_anchors: set[str] | None = None,
) -> None:
    """Append records to JSONL, or preserve JSON-array behavior for .json files."""
    path = Path(output_file)
    if known_anchors is None:
        _, existing_anchors = _load_generation_state(output_file)
    else:
        existing_anchors = known_anchors

    new_anchors = {
        _anchor_key(document["anchor"]) for document in new_documents
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".jsonl":
        serialized_documents = "".join(
            json.dumps(document, ensure_ascii=False) + "\n"
            for document in new_documents
        )
        with path.open("a", encoding="utf-8") as file:
            file.write(serialized_documents)
        existing_anchors.update(new_anchors)
        return

    existing_documents: list[Document] = []
    if path.exists():
        existing_documents = json.loads(path.read_text(encoding="utf-8"))
    existing_documents.extend(new_documents)
    path.write_text(
        json.dumps(existing_documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    existing_anchors.update(new_anchors)


async def _generate_science_documents(
    config: DomainConfig,
    number_of_documents: int,
    output_file: str,
    known_anchors: set[str] | None = None,
    generation_offset: int | None = None,
) -> list[Document]:
    """Generate one batch of Vietnamese science retrieval records."""
    if number_of_documents <= 0:
        raise ValueError("number_of_documents must be greater than zero")

    api_key = os.getenv("FPT_API_KEY")
    if not api_key:
        raise RuntimeError("FPT_API_KEY is not set")

    topics_text = ", ".join(config.topics)
    sources_text = "\n".join(f"- {source}" for source in config.sources)
    if generation_offset is None:
        generation_offset = len(known_anchors) if known_anchors is not None else 0
    batch_number = generation_offset // number_of_documents
    focus_topic = config.topics[batch_number % len(config.topics)]
    source_cycle = batch_number // len(config.topics)
    focus_source = config.sources[source_cycle % len(config.sources)]
    batch_id = secrets.token_hex(4)
    prompt = f"""
Tạo đúng {number_of_documents} mẫu retrieval hỏi đáp về {config.name} bằng tiếng Việt.
Mục tiêu là huấn luyện mô hình phân biệt tài liệu đúng với tài liệu gần nghĩa nhưng không
trả lời được câu hỏi.
Đây là batch tiếp theo, bắt đầu sau {generation_offset} mẫu đã tạo. Hãy đa dạng hóa cách
đặt câu hỏi và nội dung để hạn chế trùng với các batch trước.
Mã batch: {batch_id}.

Batch này chỉ tập trung vào chủ đề: {focus_topic}.
Nguồn ưu tiên cho batch này: {focus_source}.
Hãy khai thác các khái niệm hẹp, cơ chế, trường hợp áp dụng và quan hệ nguyên nhân-kết quả,
không chỉ tạo các câu hỏi định nghĩa phổ biến.

Các chủ đề được phép: {topics_text}.

Chỉ sử dụng kiến thức khoa học ổn định từ các nguồn sau:
{sources_text}

Mỗi mẫu phải có đúng 6 trường:
- source: chép nguyên văn một nguồn trong danh sách trên, không tự tạo URL.
- title: tiêu đề ngắn mô tả kiến thức chính.
- topic: một chủ đề trong danh sách được phép.
- anchor: câu hỏi rõ ràng, tự nhiên, không mơ hồ.
- positive: đoạn đúng về khoa học, tự chứa đủ thông tin để trả lời anchor.
- hard_negative: đoạn đúng về khoa học, cùng chủ đề và có từ vựng gần positive nhưng
  không trả lời được anchor và không được là phủ định sai của positive.

Yêu cầu chất lượng:
- Ưu tiên định luật, định nghĩa và hiện tượng đã được khoa học công nhận.
- Không dùng dữ kiện thời sự, số liệu dễ thay đổi hoặc tuyên bố chưa có đồng thuận.
- Positive và hard_negative dài 2-4 câu, tương đương nhau về độ dài và độ khó.
- Không lặp lại cùng một thông tin nhiều lần trong một đoạn.
- Không lặp câu hỏi hoặc tài liệu giữa các mẫu.
- Nếu không chắc chắn về một chi tiết thì không sử dụng chi tiết đó.
- Chỉ trả về một JSON array hợp lệ, không thêm markdown hay lời giải thích.
"""

    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia biên soạn dữ liệu khoa học cho hệ thống retrieval. "
                    "Mọi đoạn văn phải đúng về khoa học, có nguồn rõ ràng và bằng tiếng Việt."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0.4,
        "max_tokens": 8192,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            API_URL,
            headers=headers,
            json=request_data,
        )
        response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    parsed_documents = _parse_json_documents(content)
    new_documents = _validate_documents(
        parsed_documents,
        number_of_documents,
        config,
    )
    unique_documents = _filter_unique_documents(new_documents, known_anchors)
    duplicate_count = len(new_documents) - len(unique_documents)
    if duplicate_count:
        print(f"Skipped {duplicate_count} duplicate records")
    if not unique_documents:
        raise ValueError("The model returned only duplicate records")
    _append_to_json(output_file, unique_documents, known_anchors)
    return unique_documents


async def phy_doc(
    number_of_documents: int = 10,
    output_file: str = "data/physics_qa.json",
    known_anchors: set[str] | None = None,
    generation_offset: int | None = None,
) -> list[Document]:
    """Generate physics retrieval documents."""
    return await _generate_science_documents(
        PHYSICS,
        number_of_documents,
        output_file,
        known_anchors,
        generation_offset,
    )


async def bio_doc(
    number_of_documents: int = 10,
    output_file: str = "data/biology_qa.json",
    known_anchors: set[str] | None = None,
    generation_offset: int | None = None,
) -> list[Document]:
    """Generate biology retrieval documents."""
    return await _generate_science_documents(
        BIOLOGY,
        number_of_documents,
        output_file,
        known_anchors,
        generation_offset,
    )


async def it_doc(
    number_of_documents: int = 10,
    output_file: str = "data/information_technology_qa.json",
    known_anchors: set[str] | None = None,
    generation_offset: int | None = None,
) -> list[Document]:
    """Generate information technology retrieval documents."""
    return await _generate_science_documents(
        INFORMATION_TECHNOLOGY,
        number_of_documents,
        output_file,
        known_anchors,
        generation_offset,
    )


async def chemistry_doc(
    number_of_documents: int = 10,
    output_file: str = "data/chemistry_qa.json",
    known_anchors: set[str] | None = None,
    generation_offset: int | None = None,
) -> list[Document]:
    """Generate chemistry retrieval documents."""
    return await _generate_science_documents(
        CHEMISTRY,
        number_of_documents,
        output_file,
        known_anchors,
        generation_offset,
    )


async def math_doc(
    number_of_documents: int = 10,
    output_file: str = "data/mathematics_qa.json",
    known_anchors: set[str] | None = None,
    generation_offset: int | None = None,
) -> list[Document]:
    """Generate mathematics retrieval documents."""
    return await _generate_science_documents(
        MATHEMATICS,
        number_of_documents,
        output_file,
        known_anchors,
        generation_offset,
    )


async def astronomy_doc(
    number_of_documents: int = 10,
    output_file: str = "data/astronomy_qa.json",
    known_anchors: set[str] | None = None,
    generation_offset: int | None = None,
) -> list[Document]:
    """Generate astronomy retrieval documents."""
    return await _generate_science_documents(
        ASTRONOMY,
        number_of_documents,
        output_file,
        known_anchors,
        generation_offset,
    )


async def earth_science_doc(
    number_of_documents: int = 10,
    output_file: str = "data/earth_science_qa.json",
    known_anchors: set[str] | None = None,
    generation_offset: int | None = None,
) -> list[Document]:
    """Generate earth science retrieval documents."""
    return await _generate_science_documents(
        EARTH_SCIENCE,
        number_of_documents,
        output_file,
        known_anchors,
        generation_offset,
    )


# Aliases matching the names requested in earlier scripts.
IT_doc = it_doc
Chemistry_doc = chemistry_doc


async def _generate_domain_batches(
    domain_name: str,
    generator: DocumentGenerator,
    output_file: Path,
    target_count: int,
    batch_size: int,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    workers_per_domain: int,
) -> int:
    """Generate one domain concurrently and resume from existing data."""
    current_count, known_anchors = _load_generation_state(str(output_file))
    if current_count >= target_count:
        print(f"[{domain_name}] Already complete: {current_count}/{target_count}")
        return current_count

    print(f"[{domain_name}] Starting at {current_count}/{target_count}")
    state_lock = asyncio.Lock()
    reserved_count = 0
    completed_batches = 0

    async def worker() -> None:
        nonlocal current_count, reserved_count, completed_batches

        while True:
            async with state_lock:
                remaining = target_count - current_count - reserved_count
                if remaining <= 0:
                    return
                current_batch_size = min(batch_size, remaining)
                generation_offset = current_count + reserved_count
                reserved_count += current_batch_size

            last_error: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    async with semaphore:
                        documents = await generator(
                            current_batch_size,
                            str(output_file),
                            known_anchors,
                            generation_offset,
                        )
                    break
                except (
                    httpx.HTTPError,
                    json.JSONDecodeError,
                    KeyError,
                    ValueError,
                ) as error:
                    last_error = error
                    if attempt == max_retries:
                        async with state_lock:
                            reserved_count -= current_batch_size
                        raise RuntimeError(
                            f"[{domain_name}] Batch failed after "
                            f"{max_retries} attempts"
                        ) from error

                    delay_seconds = min(60, 2 ** attempt)
                    print(
                        f"[{domain_name}] Attempt {attempt}/{max_retries} failed: "
                        f"{error}. Retrying in {delay_seconds}s"
                    )
                    await asyncio.sleep(delay_seconds)
            else:
                async with state_lock:
                    reserved_count -= current_batch_size
                raise RuntimeError(f"[{domain_name}] Batch failed") from last_error

            async with state_lock:
                reserved_count -= current_batch_size
                current_count += len(documents)
                completed_batches += 1
                if completed_batches % 10 == 0 or current_count == target_count:
                    print(f"[{domain_name}] Progress: {current_count}/{target_count}")

    await asyncio.gather(*(worker() for _ in range(workers_per_domain)))

    return current_count


async def main(
    total_documents: int = DEFAULT_TOTAL_DOCUMENTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    workers_per_domain: int = DEFAULT_WORKERS_PER_DOMAIN,
) -> dict[str, int]:
    """Generate a balanced Vietnamese science retrieval dataset."""
    if total_documents <= 0:
        raise ValueError("total_documents must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero")
    if workers_per_domain <= 0:
        raise ValueError("workers_per_domain must be greater than zero")
    if max_retries <= 0:
        raise ValueError("max_retries must be greater than zero")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generators: tuple[tuple[str, DocumentGenerator], ...] = (
        ("physics", phy_doc),
        ("biology", bio_doc),
        ("information_technology", it_doc),
        ("chemistry", chemistry_doc),
        ("mathematics", math_doc),
        ("astronomy", astronomy_doc),
        ("earth_science", earth_science_doc),
    )

    documents_per_domain, remainder = divmod(total_documents, len(generators))
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks: list[Awaitable[int]] = []

    for index, (domain_name, generator) in enumerate(generators):
        target_count = documents_per_domain + (1 if index < remainder else 0)
        output_file = destination / f"{domain_name}_qa.jsonl"
        tasks.append(
            _generate_domain_batches(
                domain_name=domain_name,
                generator=generator,
                output_file=output_file,
                target_count=target_count,
                batch_size=batch_size,
                max_retries=max_retries,
                semaphore=semaphore,
                workers_per_domain=workers_per_domain,
            )
        )

    generated_counts = await asyncio.gather(*tasks)
    results = {
        domain_name: count
        for (domain_name, _), count in zip(generators, generated_counts)
    }
    print(f"Generation complete. Total records: {sum(results.values())}")
    return results


if __name__ == "__main__":
    asyncio.run(
        main(
            total_documents=int(
                os.getenv("SCIENCE_TOTAL_DOCUMENTS", DEFAULT_TOTAL_DOCUMENTS)
            ),
            batch_size=int(os.getenv("SCIENCE_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
            max_concurrency=int(
                os.getenv("SCIENCE_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
            ),
            workers_per_domain=int(
                os.getenv("SCIENCE_WORKERS_PER_DOMAIN", DEFAULT_WORKERS_PER_DOMAIN)
            ),
            max_retries=int(
                os.getenv("SCIENCE_MAX_RETRIES", DEFAULT_MAX_RETRIES)
            ),
            output_dir=os.getenv("SCIENCE_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
        )
    )
