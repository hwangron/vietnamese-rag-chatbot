"""Kiểm thử nhanh các bước xử lý văn bản (không cần mạng, không cần API key).

Chạy:  python -m pytest tests/test_rag.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunker import Chunk, chunk_pages, is_heading  # noqa: E402
from rag.loader import Page, clean_text  # noqa: E402
from rag.retriever import SearchResult, reciprocal_rank_fusion, tokenize  # noqa: E402


def test_clean_text_normalizes_unicode_and_whitespace():
    decomposed = "Nghỉ phép"            # chữ và dấu tách rời (dạng NFD)
    assert clean_text(f"  {decomposed}\xa0năm\n ") == "Nghỉ phép năm"
    assert clean_text("TC­07 ꞏ A") == "TC-07 · A"


def test_is_heading():
    assert is_heading("Điều 7. Nghỉ phép năm")
    assert is_heading("Chương 2. Thời gian làm việc")
    assert is_heading("2.1. Phạm vi áp dụng")
    assert not is_heading("Điều kiện để được thưởng là làm đủ 12 tháng.")
    assert not is_heading("1.200.000 đồng")


def test_chunks_follow_sections_and_heading_moves_to_next_page():
    pages = [
        Page("a.pdf", 1, ["Điều 1. Phạm vi", "Nội dung điều 1.", "Điều 2. Tạm ứng"]),
        Page("a.pdf", 2, ["Được tạm ứng tối đa 70%."]),
    ]
    chunks = chunk_pages(pages, "Quy định", chunk_size=800, overlap=0)
    assert [c.section for c in chunks] == ["Điều 1. Phạm vi", "Điều 2. Tạm ứng"]
    assert chunks[1].page == 2
    assert chunks[1].text == "[Quy định | Điều 2. Tạm ứng]\nĐiều 2. Tạm ứng\nĐược tạm ứng tối đa 70%."


def test_long_section_is_split_with_overlap_and_keeps_heading():
    body = " ".join(f"Câu số {i} nói về một quy định của công ty." for i in range(30))
    pages = [Page("a.pdf", 1, ["Điều 1. Quy định dài", body])]
    chunks = chunk_pages(pages, "T", chunk_size=300, overlap=60)
    assert len(chunks) > 3
    assert "Điều 1. Quy định dài\nCâu số 0" in chunks[0].text          # tiêu đề không bị tách riêng
    assert all(c.section == "Điều 1. Quy định dài" for c in chunks)
    first, second = (c.text.split("\n", 1)[1] for c in chunks[:2])  # bỏ dòng ngữ cảnh
    assert first.endswith(second.split("\n")[0])                       # đoạn sau lặp lại phần cuối đoạn trước


def test_tokenize_adds_syllable_pairs():
    assert tokenize("Nghỉ phép năm") == ["nghỉ", "phép", "năm", "nghỉ_phép", "phép_năm"]


def test_reciprocal_rank_fusion_prefers_chunks_ranked_high_in_both_lists():
    a, b, c = (Chunk(i, i, "x.pdf", 1, "") for i in "abc")
    vector = [SearchResult(a, 0.9), SearchResult(b, 0.8)]
    bm25 = [SearchResult(b, 7.0), SearchResult(c, 5.0)]
    assert [r.chunk.id for r in reciprocal_rank_fusion([vector, bm25], k=3)] == ["b", "a", "c"]
