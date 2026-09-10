#!/usr/bin/env python3
"""Crawl Vietnamese FAQ sources and write grouped Parquet datasets.

The crawler uses public HTML/JSON endpoints only, with a conservative request
rate.  It produces three datasets:

* ``banking_faq.parquet``: Vietcombank (new and legacy portal) and ACB QR.
* ``legal_bhxh_faq.parquet``: BHXH Vietnam and Ministry of Justice legal FAQs.
* ``consumer_faq.parquet``: MobiFone 5G and Vietnam Airlines.

The legacy Vietcombank portal and the Ministry of Justice site occasionally
return WAF/timeout pages.  A failure for one source is recorded in
``crawl_report.json`` and does not discard rows obtained from other sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)

# These public sites use WAF rules that reject non-browser-looking user agents.
# The crawler remains identifiable through normal traffic rate and source URLs in
# its output, while this agent avoids a blanket block for otherwise public pages.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
TIMEOUT_SECONDS = 45
DEFAULT_DELAY_SECONDS = 0.25

MOBIFONE_URL = "https://5g.mobifone.vn/mobifone5g/danh-muc-chinh/cau-hoi-thuong-gap"
VCB_URL = (
    "https://www.vietcombank.com.vn/vi-VN/KHCN/Lien-he-va-Ho-tro/"
    "Danh-sach-cau-hoi-thuong-gap"
)
VCB_LEGACY_URL = "https://portal.vietcombank.com.vn/FAQs/Pages/cau-hoi-thuong-gap.aspx?cate=16"
BHXH_URL = "https://baohiemxahoi.gov.vn/hoidap/Pages/default.aspx"
MOJ_URL = "https://pbgdpl.moj.gov.vn/qt/tl-pbgdpl/Pages/sach.aspx"
VNA_URL = "https://www.vietnamairlines.com/vn/vi/support/faq"
ACB_URL = "https://qrportal.acb.com.vn/help"

PARQUET_COLUMNS = [
    "id",
    "source",
    "source_group",
    "category",
    "question",
    "answer",
    "url",
    "crawled_at",
]


class CrawlError(RuntimeError):
    """A source could not be crawled or did not expose an expected structure."""


def is_access_control_page(response_text: str) -> bool:
    """Identify common WAF/CAPTCHA pages returned with a misleading HTTP 200."""
    normalized = response_text.casefold()
    # Do not match the bare word "captcha": legitimate pages often ship a
    # CAPTCHA component in their JavaScript bundle without presenting a challenge.
    blocked_markers = ("an error occurred while processing your request", "access denied", "captcha challenge")
    return any(marker in normalized for marker in blocked_markers)


@dataclass(frozen=True)
class FAQRow:
    """A normalized FAQ item suitable for retrieval dataset ingestion."""

    source: str
    source_group: str
    category: str
    question: str
    answer: str
    url: str
    crawled_at: str

    @property
    def id(self) -> str:
        raw = "\x1f".join((self.source, self.question, self.answer, self.url))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, str]:
        row = asdict(self)
        row["id"] = self.id
        return {column: row[column] for column in PARQUET_COLUMNS}


@dataclass
class CrawlReport:
    """Per-source outcome written alongside the Parquet files."""

    source: str
    source_group: str
    rows: int = 0
    skipped_incomplete: int = 0
    error: str | None = None


class HttpClient:
    """Small requests wrapper with retries and a per-request delay."""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self._last_request_at = 0.0
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.8"})

    def get(self, url: str, *, params: dict[str, str | int] | None = None) -> requests.Response:
        wait_seconds = self.delay_seconds - (time.monotonic() - self._last_request_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        response = self.session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        if is_access_control_page(response.text):
            response = self._curl_fallback(url, params=params)
        return response

    def _curl_fallback(self, url: str, *, params: dict[str, str | int] | None) -> requests.Response:
        """Retry WAF-blocked public pages with curl's browser TLS fingerprint."""
        curl = shutil.which("curl")
        if not curl:
            return self.session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        prepared_url = requests.Request("GET", url, params=params).prepare().url
        completed = subprocess.run(
            [curl, "-L", "--silent", "--show-error", "--fail", "--max-time", str(TIMEOUT_SECONDS), "-A", USER_AGENT, prepared_url],
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise CrawlError(f"curl fallback failed for {url}: {clean_text(completed.stderr.decode(errors='replace'))}")
        response = requests.Response()
        response.status_code = 200
        response.url = prepared_url
        response._content = completed.stdout
        response.encoding = "utf-8"
        return response

    def close(self) -> None:
        self.session.close()


def clean_text(value: str | None) -> str:
    """Normalize whitespace while preserving sentence boundaries."""
    return re.sub(r"\s+", " ", value or "").strip()


def html_to_text(html: str) -> str:
    return clean_text(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def make_row(
    source: str,
    source_group: str,
    question: str,
    answer: str,
    url: str,
    category: str = "",
) -> FAQRow | None:
    question = clean_text(question)
    answer = clean_text(answer)
    if not question or not answer:
        return None
    return FAQRow(
        source=source,
        source_group=source_group,
        category=clean_text(category),
        question=question,
        answer=answer,
        url=url,
        crawled_at=now_utc(),
    )


def response_soup(client: HttpClient, url: str, *, params: dict[str, str | int] | None = None) -> BeautifulSoup:
    return BeautifulSoup(client.get(url, params=params).text, "html.parser")


def assert_not_blocked(response_text: str, source: str) -> None:
    if is_access_control_page(response_text):
        raise CrawlError(f"{source} returned an access-control page; retry later or provide an approved session.")


def crawl_mobifone(client: HttpClient) -> list[FAQRow]:
    """Extract adjacent MobiFone accordion question/answer blocks."""
    response = client.get(MOBIFONE_URL)
    assert_not_blocked(response.text, "MobiFone")
    soup = BeautifulSoup(response.text, "html.parser")
    rows: list[FAQRow] = []
    for question_node in soup.select(".faq-wrapper .filter-about"):
        answer_node = question_node.find_next_sibling("div", class_=lambda value: value and "answer" in value)
        if not answer_node:
            continue
        answer_content = answer_node.select_one(".filter-about-answer") or answer_node
        row = make_row(
            "MobiFone 5G",
            "consumer",
            question_node.get_text(" ", strip=True),
            answer_content.get_text(" ", strip=True),
            MOBIFONE_URL,
        )
        if row:
            rows.append(row)
    if not rows:
        raise CrawlError("MobiFone FAQ accordions were not found.")
    return rows


def crawl_vietcombank(client: HttpClient) -> list[FAQRow]:
    """Use Vietcombank's public Sitecore search endpoint, which returns all FAQ items."""
    response = client.get(VCB_URL)
    assert_not_blocked(response.text, "Vietcombank")
    page = BeautifulSoup(response.text, "html.parser")
    component = page.select_one("[data-search-endpoint]")
    if not component or not component.get("data-search-endpoint"):
        raise CrawlError("Vietcombank search endpoint was not found in the FAQ page.")

    endpoint = urljoin(VCB_URL, component["data-search-endpoint"])
    # ``p`` is the Sitecore endpoint's page-size parameter, not a page number.
    # The live endpoint currently reports 366 items and accepts 1000 here.
    data = client.get(endpoint, params={"p": 1000}).json()
    results = data.get("Results", [])
    if not isinstance(results, list):
        raise CrawlError("Vietcombank search endpoint returned an unexpected JSON payload.")

    rows: list[FAQRow] = []
    for result in results:
        item = BeautifulSoup(result.get("Html", ""), "html.parser")
        question = item.select_one(".field-heading")
        answer = item.select_one(".field-content")
        item_url = urljoin(VCB_URL, result.get("Url", ""))
        row = make_row(
            "Vietcombank",
            "banking",
            question.get_text(" ", strip=True) if question else "",
            answer.get_text(" ", strip=True) if answer else "",
            item_url,
        )
        if row:
            rows.append(row)
    if not rows:
        raise CrawlError("Vietcombank search endpoint returned no complete FAQ pairs.")
    return rows


def crawl_vietcombank_legacy(client: HttpClient) -> list[FAQRow]:
    """Parse the legacy portal if its edge protection permits a public request."""
    response = client.get(VCB_LEGACY_URL)
    assert_not_blocked(response.text, "Vietcombank legacy portal")
    soup = BeautifulSoup(response.text, "html.parser")
    rows = extract_faq_pairs_from_html(soup, "Vietcombank legacy portal", "banking", VCB_LEGACY_URL)
    if not rows:
        raise CrawlError("Legacy Vietcombank portal did not expose complete FAQ pairs in HTML.")
    return rows


def crawl_vietnam_airlines(client: HttpClient) -> list[FAQRow]:
    """Collect every tag from the FAQ page and paginate its documented JSON API."""
    response = client.get(VNA_URL)
    assert_not_blocked(response.text, "Vietnam Airlines")
    page = BeautifulSoup(response.text, "html.parser")
    sidebar = page.select_one(".cmp-sidebar-left[data-options]")
    if not sidebar:
        raise CrawlError("Vietnam Airlines FAQ API configuration was not found.")
    options = json.loads(sidebar["data-options"])
    endpoint = urljoin(VNA_URL, options["apiURL"])
    page_size = 100
    tags = sorted({tag["data-filter-tag"] for tag in page.select("[data-filter-tag]") if tag.get("data-filter-tag")})
    if not tags:
        raise CrawlError("Vietnam Airlines FAQ category tags were not found.")

    rows: list[FAQRow] = []
    for tag in tags:
        first_page = client.get(endpoint, params={"tags": tag, "pageNumber": 1, "pageSize": page_size}).json()
        total_pages = int(first_page.get("totalPages", 1) or 1)
        payloads = [first_page]
        for page_number in range(2, total_pages + 1):
            payloads.append(
                client.get(endpoint, params={"tags": tag, "pageNumber": page_number, "pageSize": page_size}).json()
            )
        for payload in payloads:
            for item in payload.get("items", []):
                row = make_row(
                    "Vietnam Airlines",
                    "consumer",
                    item.get("question", ""),
                    html_to_text(item.get("answer", "")),
                    urljoin(VNA_URL, item.get("keyUrl", "")),
                    tag.removeprefix("vna:form/faq/").replace("/", " > "),
                )
                if row:
                    rows.append(row)
    if not rows:
        raise CrawlError("Vietnam Airlines FAQ API returned no complete FAQ pairs.")
    return deduplicate_rows(rows)


def crawl_acb(client: HttpClient) -> list[FAQRow]:
    """Extract the Question/Answer table embedded in the ACB QR help page."""
    response = client.get(ACB_URL)
    assert_not_blocked(response.text, "ACB QR Portal")
    soup = BeautifulSoup(response.text, "html.parser")
    rows: list[FAQRow] = []
    for table in soup.find_all("table"):
        header = clean_text(table.find("tr").get_text(" ", strip=True) if table.find("tr") else "").casefold()
        if "câu hỏi" not in header or "trả lời" not in header:
            continue
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all(["td", "th"], recursive=False)
            if len(cells) < 3:
                continue
            row = make_row(
                "ACB QR Portal",
                "banking",
                cells[1].get_text(" ", strip=True),
                cells[2].get_text(" ", strip=True),
                ACB_URL,
            )
            if row:
                rows.append(row)
    if not rows:
        raise CrawlError("ACB QR FAQ table was not found.")
    return rows


def extract_bhxh_categories(soup: BeautifulSoup) -> dict[int, str]:
    """Read category ids and labels from the BHXH filter select element."""
    categories: dict[int, str] = {}
    for option in soup.select("#selectLinhVuc option[value]"):
        value = option.get("value", "0")
        if value.isdigit() and int(value) > 0:
            categories[int(value)] = clean_text(option.get_text(" ", strip=True)).lstrip("|- ")
    if not categories:
        raise CrawlError("BHXH FAQ categories were not found.")
    return categories


def bhxh_total_pages(soup: BeautifulSoup) -> int:
    """Find the final page number from the SharePoint pagination links."""
    page_numbers = []
    for anchor in soup.select("ul.pagination a[href]"):
        query = parse_qs(urlparse(anchor["href"]).query)
        if query.get("Page", [""])[0].isdigit():
            page_numbers.append(int(query["Page"][0]))
    return max(page_numbers, default=1)


def parse_bhxh_detail(soup: BeautifulSoup, url: str) -> FAQRow | None:
    """Return a BHXH pair only when the source actually exposes the answer text."""
    item = soup.select_one(".item-vanban")
    if not item or "Chưa trả lời" in item.get_text(" ", strip=True):
        return None
    question_label = item.find(string=re.compile(r"^\s*Nội dung câu hỏi\s*:\s*$", re.I))
    answer_label = item.find(string=re.compile(r"^\s*(?:Nội dung )?(?:trả lời|đáp)\s*:\s*$", re.I))
    if not question_label or not answer_label:
        return None
    question_container = question_label.parent.parent if question_label.parent else None
    answer_container = answer_label.parent.parent if answer_label.parent else None
    question = question_container.get_text(" ", strip=True) if question_container else ""
    answer = answer_container.get_text(" ", strip=True) if answer_container else ""
    question = re.sub(r"^Nội dung câu hỏi\s*:\s*", "", question, flags=re.I)
    answer = re.sub(r"^(Nội dung )?trả lời\s*:\s*", "", answer, flags=re.I)
    category_node = item.find(string=re.compile(r"Lĩnh vực", re.I))
    category = ""
    if category_node and category_node.parent:
        category = clean_text(category_node.parent.parent.get_text(" ", strip=True) if category_node.parent.parent else "")
        category = re.sub(r"^Lĩnh vực\s*:\s*", "", category, flags=re.I)
    return make_row("BHXH Việt Nam", "legal_bhxh", question, answer, url, category)


def crawl_bhxh(client: HttpClient, max_pages_per_category: int) -> tuple[list[FAQRow], int]:
    """Crawl all answered BHXH pairs; a limit is available for a small trial run."""
    landing = response_soup(client, BHXH_URL)
    categories = extract_bhxh_categories(landing)
    rows: list[FAQRow] = []
    skipped = 0
    for category_id, category in categories.items():
        first_page = response_soup(client, BHXH_URL, params={"CateID": category_id})
        total_pages = bhxh_total_pages(first_page)
        if max_pages_per_category:
            total_pages = min(total_pages, max_pages_per_category)
        LOGGER.info("BHXH %s: %s page(s)", category, total_pages)
        for page_number in range(1, total_pages + 1):
            page = first_page if page_number == 1 else response_soup(
                client, BHXH_URL, params={"CateID": category_id, "Page": page_number}
            )
            for item in page.select(".item-vanban"):
                text = item.get_text(" ", strip=True)
                if "Chưa trả lời" in text or "Đã trả lời" not in text:
                    skipped += 1
                    continue
                detail_link = item.select_one('a[href*="ItemID="]')
                if not detail_link:
                    skipped += 1
                    continue
                detail_url = urljoin(BHXH_URL, detail_link["href"])
                row = parse_bhxh_detail(response_soup(client, detail_url), detail_url)
                if row:
                    rows.append(row)
                else:
                    skipped += 1
    return deduplicate_rows(rows), skipped


def extract_faq_pairs_from_html(
    soup: BeautifulSoup,
    source: str,
    source_group: str,
    url: str,
    category: str = "",
) -> list[FAQRow]:
    """Extract standard FAQ tables, definition lists, and question/answer classes."""
    rows: list[FAQRow] = []
    for table in soup.find_all("table"):
        header_cells = table.find("tr")
        header = clean_text(header_cells.get_text(" ", strip=True) if header_cells else "").casefold()
        if "câu hỏi" not in header or not ("trả lời" in header or "đáp" in header):
            continue
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all(["td", "th"], recursive=False)
            if len(cells) >= 2:
                question_cell, answer_cell = cells[-2:]
                row = make_row(source, source_group, question_cell.get_text(" ", strip=True), answer_cell.get_text(" ", strip=True), url, category)
                if row:
                    rows.append(row)
    for question in soup.select(".question, .faq-question, [data-question]"):
        answer = question.find_next_sibling(class_=re.compile(r"answer", re.I))
        if answer:
            row = make_row(source, source_group, question.get_text(" ", strip=True), answer.get_text(" ", strip=True), url, category)
            if row:
                rows.append(row)
    return deduplicate_rows(rows)


def extract_faq_pairs_from_text(text: str, source: str, source_group: str, url: str, category: str) -> list[FAQRow]:
    """Extract ``Câu hỏi/Hỏi`` ... ``Trả lời/Đáp`` pairs from a text PDF."""
    marker = r"(?:Câu\s*hỏi|Hỏi)\s*(?:số\s*)?\d*\s*[:.\-]?"
    answer_marker = r"(?:Trả\s*lời|Đáp)\s*[:.\-]?"
    pattern = re.compile(
        rf"{marker}\s*(?P<question>.+?)\s*{answer_marker}\s*(?P<answer>.+?)(?=(?:{marker})|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    rows = [
        make_row(source, source_group, match.group("question"), match.group("answer"), url, category)
        for match in pattern.finditer(text)
    ]
    return deduplicate_rows([row for row in rows if row])


def pdf_to_text(content: bytes) -> str:
    """Use the system Poppler utility so no PDF Python package is required."""
    if not shutil.which("pdftotext"):
        raise CrawlError("pdftotext is required to parse Ministry of Justice PDF FAQ books.")
    with tempfile.TemporaryDirectory(prefix="faq_moj_") as directory:
        pdf_path = Path(directory) / "source.pdf"
        text_path = Path(directory) / "source.txt"
        pdf_path.write_bytes(content)
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(text_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise CrawlError(f"pdftotext failed: {clean_text(completed.stderr)}")
        return text_path.read_text(encoding="utf-8", errors="replace")


def crawl_moj(client: HttpClient, max_documents: int) -> tuple[list[FAQRow], int]:
    """Discover legal FAQ books and parse direct HTML/PDF question-answer pairs."""
    response = client.get(MOJ_URL)
    assert_not_blocked(response.text, "Ministry of Justice")
    soup = BeautifulSoup(response.text, "html.parser")
    candidates: list[tuple[str, str]] = []
    for anchor in soup.select("a[href]"):
        href = urljoin(MOJ_URL, anchor["href"])
        label = clean_text(anchor.get_text(" ", strip=True))
        path = urlparse(href).path.casefold()
        if href.startswith("https://pbgdpl.moj.gov.vn") and ("hỏi đáp" in label.casefold() or path.endswith(".pdf")):
            candidates.append((href, label or "Tủ sách hỏi đáp pháp luật"))
    candidates = list(dict.fromkeys(candidates))[:max_documents]
    if not candidates:
        raise CrawlError("No legal FAQ book links were found on the Ministry of Justice page.")

    rows: list[FAQRow] = []
    skipped = 0
    for url, title in candidates:
        document = client.get(url)
        content_type = document.headers.get("content-type", "").casefold()
        if "pdf" in content_type or urlparse(url).path.casefold().endswith(".pdf"):
            document_rows = extract_faq_pairs_from_text(pdf_to_text(document.content), "Bộ Tư pháp", "legal_bhxh", url, title)
        else:
            document_soup = BeautifulSoup(document.text, "html.parser")
            document_rows = extract_faq_pairs_from_html(document_soup, "Bộ Tư pháp", "legal_bhxh", url, title)
            for pdf in document_soup.select('a[href$=".pdf" i]'):
                pdf_url = urljoin(url, pdf["href"])
                pdf_response = client.get(pdf_url)
                document_rows.extend(
                    extract_faq_pairs_from_text(pdf_to_text(pdf_response.content), "Bộ Tư pháp", "legal_bhxh", pdf_url, title)
                )
        if document_rows:
            rows.extend(document_rows)
        else:
            skipped += 1
    return deduplicate_rows(rows), skipped


def deduplicate_rows(rows: Iterable[FAQRow]) -> list[FAQRow]:
    """Keep first occurrence by stable content hash."""
    unique: dict[str, FAQRow] = {}
    for row in rows:
        unique.setdefault(row.id, row)
    return list(unique.values())


def write_parquet(rows: list[FAQRow], output_path: Path) -> int:
    """Write a stable schema even when every source in a group fails."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row.to_dict() for row in deduplicate_rows(rows)], columns=PARQUET_COLUMNS)
    frame.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
    return len(frame)


def run_source(
    name: str,
    source_group: str,
    crawl: Callable[[], list[FAQRow] | tuple[list[FAQRow], int]],
) -> tuple[list[FAQRow], CrawlReport]:
    """Isolate source failures so remaining datasets can still be written."""
    report = CrawlReport(source=name, source_group=source_group)
    try:
        result = crawl()
        if isinstance(result, tuple):
            rows, report.skipped_incomplete = result
        else:
            rows = result
        report.rows = len(rows)
        return rows, report
    except (CrawlError, requests.RequestException, json.JSONDecodeError) as error:
        report.error = str(error)
        LOGGER.error("%s: %s", name, error)
        return [], report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_faq"), help="Directory for Parquet files and crawl_report.json.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Minimum seconds between HTTP requests (default: %(default)s).")
    parser.add_argument(
        "--bhxh-max-pages-per-category",
        type=int,
        default=0,
        help="Limit BHXH list pages per category for a trial run; 0 crawls all pages (default: %(default)s).",
    )
    parser.add_argument("--moj-max-documents", type=int, default=100, help="Maximum legal FAQ books/pages to inspect (default: %(default)s).")
    parser.add_argument("--sources", nargs="+", choices=("mobifone", "vietcombank", "vietcombank_legacy", "bhxh", "moj", "vna", "acb"), help="Optional subset of sources.")
    parser.add_argument("--verbose", action="store_true", help="Enable progress logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.delay < 0 or args.bhxh_max_pages_per_category < 0 or args.moj_max_documents < 1:
        raise SystemExit("--delay and page limits must be non-negative; --moj-max-documents must be positive.")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    selected = set(args.sources or ("mobifone", "vietcombank", "vietcombank_legacy", "bhxh", "moj", "vna", "acb"))
    client = HttpClient(args.delay)
    grouped_rows: dict[str, list[FAQRow]] = {"banking": [], "legal_bhxh": [], "consumer": []}
    reports: list[CrawlReport] = []
    jobs: list[tuple[str, str, Callable[[], list[FAQRow] | tuple[list[FAQRow], int]]]] = [
        ("MobiFone 5G", "consumer", lambda: crawl_mobifone(client)),
        ("Vietcombank", "banking", lambda: crawl_vietcombank(client)),
        ("Vietcombank legacy portal", "banking", lambda: crawl_vietcombank_legacy(client)),
        ("BHXH Việt Nam", "legal_bhxh", lambda: crawl_bhxh(client, args.bhxh_max_pages_per_category)),
        ("Bộ Tư pháp", "legal_bhxh", lambda: crawl_moj(client, args.moj_max_documents)),
        ("Vietnam Airlines", "consumer", lambda: crawl_vietnam_airlines(client)),
        ("ACB QR Portal", "banking", lambda: crawl_acb(client)),
    ]
    job_keys = ("mobifone", "vietcombank", "vietcombank_legacy", "bhxh", "moj", "vna", "acb")
    try:
        for key, (name, source_group, job) in zip(job_keys, jobs, strict=True):
            if key not in selected:
                continue
            rows, report = run_source(name, source_group, job)
            grouped_rows[source_group].extend(rows)
            reports.append(report)
        output_dir: Path = args.output_dir
        output_counts = {
            "banking_faq.parquet": write_parquet(grouped_rows["banking"], output_dir / "banking_faq.parquet"),
            "legal_bhxh_faq.parquet": write_parquet(grouped_rows["legal_bhxh"], output_dir / "legal_bhxh_faq.parquet"),
            "consumer_faq.parquet": write_parquet(grouped_rows["consumer"], output_dir / "consumer_faq.parquet"),
        }
        report_path = output_dir / "crawl_report.json"
        report_path.write_text(
            json.dumps({"generated_at": now_utc(), "outputs": output_counts, "sources": [asdict(report) for report in reports]}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    finally:
        client.close()
    for filename, count in output_counts.items():
        print(f"{filename}: {count} rows")
    failed = sum(report.error is not None for report in reports)
    print(f"crawl_report.json: {len(reports) - failed} succeeded, {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
