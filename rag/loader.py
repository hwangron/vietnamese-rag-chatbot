"""Đọc file PDF thành danh sách trang; mỗi trang gồm các khối văn bản (đoạn văn, tiêu đề, dòng bảng)."""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

pymupdf.no_recommend_layout()  # tắt dòng gợi ý cài thêm gói pymupdf_layout


@dataclass
class Page:
    source: str                      # tên file PDF
    number: int                      # số trang, bắt đầu từ 1
    blocks: list[str] = field(default_factory=list)


# Ký tự bị font trong PDF ánh xạ sai khi trích xuất → ký tự đúng
_CHAR_FIXES = str.maketrans({"­": "-", "ꞏ": "·"})  # soft hyphen → "-", "ꞏ" → "·"


def clean_text(text: str) -> str:
    """Chuẩn hoá Unicode về dạng NFC và gộp mọi khoảng trắng (kể cả \\xa0, xuống dòng) thành một dấu cách.

    Một số PDF lưu tiếng Việt ở dạng tổ hợp (chữ + dấu rời) hoặc dùng dấu cách đặc biệt;
    nếu không chuẩn hoá thì cùng một từ có thể bị coi là hai từ khác nhau khi tìm kiếm.
    """
    text = unicodedata.normalize("NFC", text).translate(_CHAR_FIXES)
    return re.sub(r"\s+", " ", text).strip()


def load_pdf(path: str | Path) -> tuple[str, list[Page]]:
    """Trả về (tiêu đề tài liệu, danh sách trang)."""
    path = Path(path)
    with pymupdf.open(path) as doc:
        title = clean_text(doc.metadata.get("title") or "") or path.stem
        pages = [Page(path.name, number, _read_blocks(page)) for number, page in enumerate(doc, start=1)]

    _remove_headers_footers(pages)
    if not any(page.blocks for page in pages):
        raise ValueError(
            f"{path.name}: không đọc được chữ nào. File có thể là bản scan (ảnh) và cần OCR trước."
        )
    return title, pages


def _read_blocks(page: pymupdf.Page) -> list[str]:
    """Đọc các khối văn bản của một trang theo thứ tự từ trên xuống; bảng được xử lý riêng.

    Nếu đọc bảng như văn bản thường, mỗi hàng chỉ còn là một dãy số liền nhau
    ("Trưởng phòng 1.800.000 đồng 1.300.000 đồng 180 USD") và LLM dễ ghép nhầm cột.
    Vì vậy mỗi hàng của bảng được viết lại kèm tên cột:
    "Cấp bậc: Trưởng phòng | Hà Nội, TP. Hồ Chí Minh: 1.800.000 đồng | ..."
    """
    # find_tables() đôi khi nhận nhầm khung có nền màu thành "bảng" 1 hàng → chỉ giữ bảng thật (≥ 2×2)
    tables = [t for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
    items: list[tuple[float, float, str]] = []      # (y, x, nội dung) để sắp xếp theo vị trí

    for table in tables:
        rows = [[clean_text(cell or "") for cell in row] for row in table.extract()]
        header, body = rows[0], rows[1:]
        y0, x0 = table.bbox[1], table.bbox[0]
        for i, row in enumerate(body):
            text = " | ".join(f"{h}: {c}" if h else c for h, c in zip(header, row) if c)
            items.append((y0 + i * 0.01, x0, text))   # giữ đúng thứ tự các hàng

    table_rects = [pymupdf.Rect(t.bbox) for t in tables]
    # Mỗi block là (x0, y0, x1, y1, text, block_no, block_type); block_type 1 là ảnh
    for x0, y0, x1, y1, text, _, block_type in page.get_text("blocks"):
        if block_type != 0 or any(pymupdf.Rect(x0, y0, x1, y1).intersects(r) for r in table_rects):
            continue
        items.append((y0, x0, clean_text(text)))

    return [text for _, _, text in sorted(items) if text]


def _remove_headers_footers(pages: list[Page]) -> None:
    """Bỏ các dòng ngắn lặp lại ở phần lớn số trang, ví dụ header, footer, "Trang 2/5"."""
    if len(pages) < 2:
        return

    def key(text: str) -> str:
        return re.sub(r"\d+", "#", text.lower())  # "Trang 2/5" và "Trang 3/5" có cùng key

    counts = Counter(key(b) for page in pages for b in set(page.blocks) if len(b) <= 150)
    min_pages = max(2, math.ceil(0.6 * len(pages)))
    repeated = {k for k, n in counts.items() if n >= min_pages}
    for page in pages:
        page.blocks = [b for b in page.blocks if len(b) > 150 or key(b) not in repeated]
