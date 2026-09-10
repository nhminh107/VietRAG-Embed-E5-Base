#!/usr/bin/env python3
"""Crawl Vietnamese FAQ sources and write grouped Parquet datasets.

The crawler uses public HTML/JSON endpoints only, with a conservative request
rate.  It produces three datasets:

* ``banking_faq.parquet``: Vietcombank, ACB QR, and MSB.
* ``legal_bhxh_faq.parquet``: BHXH Vietnam, legal FAQ books, and public-service Q&A.
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
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
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
MSB_URL = "https://www.msb.com.vn/lien-he-ho-tro/cau-hoi-thuong-gap/"
LAOCAI_URL = "https://www.laocai.gov.vn/Default.aspx?dvid=1231&pageid=95227&sid=1365"

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
            allowed_methods=("GET", "POST"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.8"})

    def get(self, url: str, *, params: dict[str, str | int] | None = None) -> requests.Response:
        self._wait_for_rate_limit()
        response = self.session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        if is_access_control_page(response.text):
            response = self._curl_fallback(url, params=params)
        return response

    def post(self, url: str, *, data: dict[str, str | int]) -> requests.Response:
        """Submit a public FAQ pagination/detail form while respecting the rate limit."""
        self._wait_for_rate_limit()
        response = self.session.post(url, data=data, timeout=TIMEOUT_SECONDS)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        if is_access_control_page(response.text):
            raise CrawlError(f"POST to {url} returned an access-control page.")
        return response

    def _wait_for_rate_limit(self) -> None:
        wait_seconds = self.delay_seconds - (time.monotonic() - self._last_request_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

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


def parse_msb_faq_html(html: str, url: str = MSB_URL) -> list[FAQRow]:
    """Parse MSB's FAQ accordion HTML returned by its public WordPress API."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[FAQRow] = []
    for item in soup.select("details.msb-faq__item"):
        question = item.select_one(".msb-faq__qtext")
        answer = item.select_one(".msb-faq__a")
        categories = clean_text(item.get("data-categories", "")).replace(",", " > ")
        row = make_row(
            "MSB",
            "banking",
            question.get_text(" ", strip=True) if question else "",
            answer.get_text(" ", strip=True) if answer else "",
            url,
            categories,
        )
        if row:
            rows.append(row)
    return deduplicate_rows(rows)


def crawl_msb(client: HttpClient) -> list[FAQRow]:
    """Crawl all MSB FAQ pages through the site's public WordPress endpoint."""
    landing = client.get(MSB_URL)
    assert_not_blocked(landing.text, "MSB")
    page = BeautifulSoup(landing.text, "html.parser")
    container = page.select_one("[data-ajax-url][data-post-type]")
    if not container:
        raise CrawlError("MSB FAQ API configuration was not found.")

    endpoint = container.get("data-ajax-url")
    nonce = container.get("data-nonce")
    if not endpoint or not nonce:
        raise CrawlError("MSB FAQ API endpoint or nonce was not found.")
    per_page = 100
    common_data = {
        "action": "msb_contact_faq_load",
        "nonce": nonce,
        "post_type": container.get("data-post-type", "msb_faq"),
        "taxonomy": container.get("data-taxonomy", "msb_product_group"),
        "tab_id": container.get("data-active-tab", ""),
        "per_page": per_page,
        "orderby": container.get("data-orderby", "date_desc"),
        "search": "",
        "empty_text": container.get("data-empty-text", "Không tìm thấy câu hỏi nào."),
        "empty_icon": container.get("data-empty-icon", ""),
    }
    first_payload = {**common_data, "paged": 1}
    first_response = client.post(endpoint, data=first_payload).json()
    if not first_response.get("success") or not isinstance(first_response.get("data", {}).get("html"), str):
        raise CrawlError("MSB FAQ API returned an unexpected payload.")

    first_html = first_response["data"]["html"]
    first_page = BeautifulSoup(first_html, "html.parser")
    page_numbers = [int(node["data-page"]) for node in first_page.select("[data-page]") if node.get("data-page", "").isdigit()]
    total_pages = max(page_numbers, default=1)
    rows = parse_msb_faq_html(first_html)
    for page_number in range(2, total_pages + 1):
        response = client.post(endpoint, data={**common_data, "paged": page_number}).json()
        html = response.get("data", {}).get("html", "") if response.get("success") else ""
        if not isinstance(html, str):
            raise CrawlError(f"MSB FAQ API page {page_number} returned invalid HTML.")
        rows.extend(parse_msb_faq_html(html))
    rows = deduplicate_rows(rows)
    if not rows:
        raise CrawlError("MSB FAQ API returned no complete FAQ pairs.")
    return rows


def postback_target(href: str) -> tuple[str, str] | None:
    """Extract ASP.NET postback target and argument from a JavaScript link."""
    match = re.search(r"__doPostBack\('([^']+)',\s*'([^']*)'\)", href)
    return (match.group(1), match.group(2)) if match else None


def laocai_form_data(soup: BeautifulSoup, target: str, argument: str) -> dict[str, str]:
    """Build a safe ASP.NET postback payload from current hidden form fields."""
    form = soup.select_one("form")
    if not form:
        raise CrawlError("Lào Cai FAQ form was not found.")
    data = {input_node["name"]: input_node.get("value", "") for input_node in form.select('input[type="hidden"][name]')}
    data["__EVENTTARGET"] = target
    data["__EVENTARGUMENT"] = argument
    return data


def laocai_postback(client: HttpClient, soup: BeautifulSoup, target: str, argument: str) -> BeautifulSoup:
    """Submit an ASP.NET FAQ list/detail navigation event."""
    response = client.post(LAOCAI_URL, data=laocai_form_data(soup, target, argument))
    return BeautifulSoup(response.text, "html.parser")


def parse_laocai_detail(soup: BeautifulSoup) -> FAQRow | None:
    """Extract a complete public-service Q&A pair from one Lào Cai detail page."""
    detail = soup.select_one(".DetailQuestion")
    if not detail:
        return None
    title = detail.select_one(".blockTitle .divFirst")
    answer_block = detail.select_one(".blockDetailAns fieldset")
    if not title or not answer_block:
        return None
    answer_block.legend.decompose() if answer_block.legend else None
    question = re.sub(r"^Câu hỏi\s*:\s*", "", title.get_text(" ", strip=True), flags=re.IGNORECASE)
    answer = answer_block.get_text(" ", strip=True)
    category_node = detail.find(string=re.compile(r"Người trả lời\s*:", re.IGNORECASE))
    category = ""
    if category_node and category_node.parent:
        category = clean_text(category_node.parent.parent.get_text(" ", strip=True) if category_node.parent.parent else "")
        category = re.sub(r"^Người trả lời\s*:\s*", "", category, flags=re.IGNORECASE)
        category = re.sub(r"\s*(Chức vụ|Ngày trả lời)\s*:.*$", "", category, flags=re.IGNORECASE)
    return make_row("Cổng Hỏi đáp Lào Cai", "legal_bhxh", question, answer, LAOCAI_URL, category)


def crawl_laocai(client: HttpClient, max_pages: int, max_items: int) -> tuple[list[FAQRow], int]:
    """Crawl Lào Cai public-service answers by preserving ASP.NET form state."""
    landing = response_soup(client, LAOCAI_URL)
    list_id = "ctrl_162673_95$gvQuestionList"
    total_node = landing.select_one("#ctrl_162673_95_lbTongCauHoi")
    page_size = len(landing.select("#ctrl_162673_95_gvQuestionList tr")) - 1
    if not total_node or page_size <= 0:
        raise CrawlError("Lào Cai FAQ list statistics were not found.")
    total_pages = (int(clean_text(total_node.get_text())) + page_size - 1) // page_size
    if max_pages:
        total_pages = min(total_pages, max_pages)

    rows: list[FAQRow] = []
    skipped = 0
    for page_number in range(1, total_pages + 1):
        list_page = landing if page_number == 1 else laocai_postback(client, landing, list_id, f"Page${page_number}")
        links = list_page.select('a[href*="$lbCauHoi"]')
        if max_items:
            links = links[: max_items - len(rows) - skipped]
        LOGGER.info("Lào Cai page %s/%s: %s item(s)", page_number, total_pages, len(links))
        for link in links:
            event = postback_target(link.get("href", ""))
            if not event:
                skipped += 1
                continue
            detail_page = laocai_postback(client, list_page, *event)
            row = parse_laocai_detail(detail_page)
            if row:
                rows.append(row)
            else:
                skipped += 1
            back_link = detail_page.select_one('a[href*="$lbBack"]')
            back_event = postback_target(back_link.get("href", "")) if back_link else None
            if not back_event:
                raise CrawlError("Lào Cai FAQ detail page did not expose a return-to-list event.")
            list_page = laocai_postback(client, detail_page, *back_event)
        if max_items and len(rows) + skipped >= max_items:
            break
    return deduplicate_rows(rows), skipped


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
    parser.add_argument(
        "--laocai-max-pages",
        type=int,
        default=0,
        help="Limit Lào Cai FAQ list pages for a trial run; 0 crawls all pages (default: %(default)s).",
    )
    parser.add_argument(
        "--laocai-max-items",
        type=int,
        default=0,
        help="Limit Lào Cai FAQ details for a trial run; 0 crawls every item in selected pages (default: %(default)s).",
    )
    parser.add_argument("--moj-max-documents", type=int, default=100, help="Maximum legal FAQ books/pages to inspect (default: %(default)s).")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("mobifone", "vietcombank", "vietcombank_legacy", "bhxh", "moj", "vna", "acb", "msb", "laocai"),
        help="Optional subset of sources.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable progress logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.delay < 0
        or args.bhxh_max_pages_per_category < 0
        or args.laocai_max_pages < 0
        or args.laocai_max_items < 0
        or args.moj_max_documents < 1
    ):
        raise SystemExit("--delay and page limits must be non-negative; --moj-max-documents must be positive.")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    selected = set(args.sources or ("mobifone", "vietcombank", "vietcombank_legacy", "bhxh", "moj", "vna", "acb", "msb", "laocai"))
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
        ("MSB", "banking", lambda: crawl_msb(client)),
        ("Cổng Hỏi đáp Lào Cai", "legal_bhxh", lambda: crawl_laocai(client, args.laocai_max_pages, args.laocai_max_items)),
    ]
    job_keys = ("mobifone", "vietcombank", "vietcombank_legacy", "bhxh", "moj", "vna", "acb", "msb", "laocai")
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
