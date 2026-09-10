"""Focused parser tests for script.crawl_faq_sources."""

import sys
from pathlib import Path

from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from script.crawl_faq_sources import (
    extract_faq_pairs_from_html,
    extract_faq_pairs_from_text,
    is_access_control_page,
    parse_laocai_detail,
    parse_msb_faq_html,
    parse_bhxh_detail,
    postback_target,
)


def test_extract_faq_pairs_from_html_table() -> None:
    soup = BeautifulSoup(
        "<table><tr><th>STT</th><th>Câu hỏi</th><th>Trả lời</th></tr>"
        "<tr><td>1</td><td> Câu hỏi A </td><td> Trả lời A </td></tr></table>",
        "html.parser",
    )

    rows = extract_faq_pairs_from_html(soup, "Test", "banking", "https://example.test")

    assert len(rows) == 1
    assert rows[0].question == "Câu hỏi A"
    assert rows[0].answer == "Trả lời A"


def test_extract_faq_pairs_from_pdf_text() -> None:
    rows = extract_faq_pairs_from_text(
        "Câu hỏi 1: Tôi cần làm gì?\nTrả lời: Làm theo hướng dẫn.\n"
        "Hỏi 2: Có mất phí không?\nĐáp: Không mất phí.",
        "Test",
        "legal_bhxh",
        "https://example.test/book.pdf",
        "Book",
    )

    assert [row.question for row in rows] == ["Tôi cần làm gì?", "Có mất phí không?"]
    assert [row.answer for row in rows] == ["Làm theo hướng dẫn.", "Không mất phí."]


def test_bhxh_detail_without_answer_is_not_emitted() -> None:
    soup = BeautifulSoup(
        '<div class="item-vanban"><strong>Nội dung câu hỏi:</strong> Câu hỏi chưa trả lời</div>',
        "html.parser",
    )

    assert parse_bhxh_detail(soup, "https://example.test") is None


def test_captcha_component_is_not_mistaken_for_a_block_page() -> None:
    assert not is_access_control_page("<script>const captchaLabel = 'CAPTCHA';</script>")
    assert is_access_control_page("Access denied")


def test_parse_msb_faq_html() -> None:
    rows = parse_msb_faq_html(
        '<details class="msb-faq__item" data-categories="Tài khoản, Thẻ">'
        '<span class="msb-faq__qtext">Câu hỏi MSB</span>'
        '<div class="msb-faq__a"><p>Trả lời MSB</p></div></details>'
    )

    assert len(rows) == 1
    assert rows[0].source == "MSB"
    assert rows[0].category == "Tài khoản > Thẻ"
    assert rows[0].answer == "Trả lời MSB"


def test_parse_laocai_detail_and_postback_target() -> None:
    soup = BeautifulSoup(
        '<div class="DetailQuestion"><div class="blockTitle"><div class="divFirst">'
        'Câu hỏi: Câu hỏi Lào Cai</div></div><div class="blockDetailAns"><fieldset>'
        '<legend>Nội dung câu trả lời</legend><p>Trả lời Lào Cai</p></fieldset></div></div>',
        "html.parser",
    )

    row = parse_laocai_detail(soup)

    assert row is not None
    assert row.question == "Câu hỏi Lào Cai"
    assert row.answer == "Trả lời Lào Cai"
    assert postback_target("javascript:__doPostBack('grid','Page$2')") == ("grid", "Page$2")
