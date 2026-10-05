"""Chia các trang PDF thành những đoạn (chunk) ngắn để tìm kiếm.

Cách chia:
1. Gom các khối văn bản liền nhau thuộc cùng một mục (Chương/Điều/Mục...) và cùng một trang.
   Không gộp qua ranh giới trang để mỗi đoạn có đúng một số trang khi trích dẫn.
2. Mục dài hơn `chunk_size` ký tự được cắt thành nhiều đoạn; đoạn sau lặp lại khoảng
   `overlap` ký tự cuối của đoạn trước để không cắt mất ý ở chỗ nối.
3. Đầu mỗi đoạn có dòng ngữ cảnh "[Tên tài liệu | Chương > Điều]", giúp cả tìm kiếm lẫn LLM
   biết đoạn đó nói về mục nào, kể cả khi đoạn nằm ở giữa một mục dài.
"""

import re
from dataclasses import dataclass, field

from .loader import Page

_HEADING_RE = re.compile(r"^(chương|điều|mục|phần|phụ lục)\s+[\divxlc]+\b", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.\s+\w")   # "1. Giới thiệu", "2.3. Phạm vi"
_TOP_LEVEL_RE = re.compile(r"^(chương|phần|phụ lục)\b", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?;:])\s+")


@dataclass
class Chunk:
    id: str
    text: str       # dòng ngữ cảnh + nội dung
    source: str     # tên file PDF
    page: int
    section: str    # ví dụ "Chương 3. Nghỉ phép > Điều 7. Nghỉ phép năm"


@dataclass
class _Section:
    page: int
    name: str
    blocks: list[str] = field(default_factory=list)


def is_heading(block: str) -> bool:
    if len(block) > 100 or block.endswith((".", ",", ";", ":")):
        return False
    return bool(_HEADING_RE.match(block) or _NUMBERED_HEADING_RE.match(block))


def chunk_pages(pages: list[Page], title: str, chunk_size: int = 800, overlap: int = 120,
                add_header: bool = True) -> list[Chunk]:
    """add_header=False chỉ dùng khi đánh giá, để đo tác dụng của dòng ngữ cảnh."""
    chunks: list[Chunk] = []
    for section in _split_sections(pages):
        header = f"[{title} | {section.name}]" if section.name else f"[{title}]"
        for body in _pack(section.blocks, chunk_size, overlap):
            source = pages[0].source
            chunks.append(Chunk(
                id=f"{source}:p{section.page}:c{len(chunks)}",
                text=f"{header}\n{body}" if add_header else body,
                source=source,
                page=section.page,
                section=section.name,
            ))
    return chunks


def _split_sections(pages: list[Page]) -> list[_Section]:
    """Gom các khối liền nhau có cùng tiêu đề mục và cùng trang."""
    sections: list[_Section] = []
    chapter = article = ""          # tiêu đề cấp 1 (Chương/Phần) và cấp 2 (Điều/Mục) đang áp dụng
    carried: _Section | None = None  # tiêu đề nằm cuối trang trước, nội dung ở trang này
    for page in pages:
        current: _Section | None = None
        if carried:
            current = _Section(page.number, carried.name, carried.blocks)
            sections.append(current)
            carried = None
        for block in page.blocks:
            if is_heading(block):
                if _TOP_LEVEL_RE.match(block):
                    chapter, article = block, ""
                else:
                    article = block
                name = " > ".join(s for s in (chapter, article) if s)
                # "Chương 2" ngay sau đó là "Điều 3": gộp 2 tiêu đề vào cùng một mục
                if current and all(is_heading(b) for b in current.blocks):
                    current.name = name
                    current.blocks.append(block)
                    continue
                current = _Section(page.number, name, [block])
                sections.append(current)
            else:
                if current is None:  # đầu trang, nội dung tiếp nối mục ở trang trước
                    current = _Section(page.number, " > ".join(s for s in (chapter, article) if s))
                    sections.append(current)
                current.blocks.append(block)
        # Trang kết thúc bằng tiêu đề (nội dung ở trang sau) → chuyển tiêu đề sang trang sau
        if current and all(is_heading(b) for b in current.blocks):
            carried = sections.pop()
    return sections


def _pack(blocks: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Gom các khối thành những đoạn ≤ chunk_size ký tự, có chồng lặp giữa hai đoạn liền nhau."""
    pieces: list[str] = []
    for block in blocks:
        parts = _split_long(block, chunk_size)
        if pieces and is_heading(pieces[-1]):   # tiêu đề luôn đi cùng nội dung ngay sau nó
            pieces[-1] = f"{pieces[-1]}\n{parts[0]}"
            parts = parts[1:]
        pieces += parts
    result: list[str] = []
    current: list[str] = []
    fresh = 0                       # số mảnh mới (không tính phần chồng lặp) trong đoạn đang gom
    for piece in pieces:
        if fresh and len("\n".join(current + [piece])) > chunk_size:
            result.append("\n".join(current))
            tail = _tail("\n".join(current), overlap)
            current, fresh = ([tail] if tail else []), 0
        current.append(piece)
        fresh += 1
    if fresh:
        result.append("\n".join(current))
    return result


def _split_long(block: str, chunk_size: int) -> list[str]:
    """Cắt khối dài hơn chunk_size theo câu; câu quá dài thì cắt theo từ."""
    if len(block) <= chunk_size:
        return [block]
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END_RE.split(block):
        if current and len(current) + 1 + len(sentence) > chunk_size:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)

    result: list[str] = []
    for part in parts:
        while len(part) > chunk_size:
            cut = part.rfind(" ", 0, chunk_size)
            cut = cut if cut > 0 else chunk_size
            result.append(part[:cut])
            part = part[cut:].strip()
        if part:
            result.append(part)
    return result


def _tail(text: str, overlap: int) -> str:
    """Lấy khoảng `overlap` ký tự cuối, bắt đầu từ đầu một câu (hoặc đầu một từ)."""
    if overlap <= 0:
        return ""
    if len(text) <= overlap:
        return text
    cut = text[-overlap:]
    match = _SENTENCE_END_RE.search(cut)
    if match and match.end() < len(cut):
        return cut[match.end():]
    space = cut.find(" ")
    return cut[space + 1:] if space >= 0 else cut
