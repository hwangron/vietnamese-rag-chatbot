"""Ghép các bước RAG: nạp tài liệu → tìm đoạn liên quan → tạo prompt → gọi LLM."""

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from . import config
from .chunker import chunk_pages
from .embedder import Embedder
from .llm import LLMClient
from .loader import load_pdf
from .retriever import Retriever, SearchResult
from .store import VectorStore, collection_name

NOT_FOUND = "Tôi không tìm thấy thông tin này trong tài liệu."
# Số trích dẫn trong câu trả lời: [1] hoặc 【1】 (gpt-oss đôi khi dùng ngoặc kiểu này)
CITATION_RE = re.compile(r"[\[【](\d+)[\]】]")

# Phiên bản 2 (sau khi phân tích lỗi, xem eval/results/report.md): thêm quy tắc 3 để câu trả lời
# không bỏ sót điều kiện / ngoại lệ quan trọng, ví dụ "chỉ áp dụng sau thử việc".
SYSTEM_PROMPT = f"""Bạn là trợ lý hỏi đáp dựa trên tài liệu nội bộ của công ty.
Quy tắc bắt buộc:
1. Chỉ dùng thông tin trong phần TÀI LIỆU được cung cấp. Không dùng kiến thức bên ngoài, không suy đoán.
2. Nếu tài liệu không có thông tin để trả lời, chỉ trả lời đúng một câu: "{NOT_FOUND}"
3. Trả lời thẳng vào câu hỏi trước, sau đó nêu ngắn gọn các điều kiện, ngoại lệ hoặc thời hạn quan trọng liên quan trực tiếp tới câu hỏi (nếu tài liệu có).
4. Sau mỗi ý, ghi số thứ tự của đoạn tài liệu đã dùng trong ngoặc vuông, ví dụ [1] hoặc [2][3].
5. Trả lời ngắn gọn, chính xác, bằng tiếng Việt."""


def build_messages(question: str, results: list[SearchResult], system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    context = "\n\n".join(
        f"[{i}] (Nguồn: {r.chunk.source}, trang {r.chunk.page})\n{r.chunk.text}"
        for i, r in enumerate(results, start=1)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TÀI LIỆU:\n{context}\n\nCÂU HỎI: {question}"},
    ]


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]
    retrieval_ms: float
    llm_seconds: float


class RAGPipeline:
    def __init__(self, embed_model: str | None = None, chunk_size: int | None = None,
                 chunk_overlap: int | None = None, persist_dir: str | Path | None = config.CHROMA_DIR,
                 llm: LLMClient | None = None):
        self.embed_model = embed_model or config.EMBED_MODEL
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.embedder = Embedder(self.embed_model)
        self.store = VectorStore(collection_name(self.embed_model, self.chunk_size), persist_dir)
        self.retriever = Retriever(self.store, self.embedder)
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        # Chỉ tạo khi cần gọi LLM → vẫn nạp được tài liệu khi chưa có API key
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    # ---------- Nạp tài liệu ----------
    def index_file(self, path: str | Path) -> int:
        """Đọc, chia đoạn, tạo embedding và lưu một file PDF. Trả về số đoạn đã lưu."""
        path = Path(path)
        title, pages = load_pdf(path)
        chunks = chunk_pages(pages, title, self.chunk_size, self.chunk_overlap)
        self.store.delete_source(path.name)  # nạp lại file cũ thì thay thế, không bị trùng
        self.store.add(chunks, self.embedder.embed_documents([c.text for c in chunks]))
        self.retriever.refresh()
        return len(chunks)

    def reset(self) -> None:
        self.store.reset()
        self.retriever.refresh()

    # ---------- Hỏi đáp ----------
    def search(self, question: str, k: int | None = None, mode: str | None = None) -> list[SearchResult]:
        return self.retriever.search(question, k or config.TOP_K, mode or config.RETRIEVAL_MODE)

    def answer(self, question: str, k: int | None = None, mode: str | None = None,
               model: str | None = None) -> Answer:
        start = time.perf_counter()
        results = self.search(question, k, mode)
        retrieval_ms = (time.perf_counter() - start) * 1000
        if not results:
            return Answer(f"{NOT_FOUND} (Chưa có tài liệu nào được nạp.)", [], retrieval_ms, 0.0)
        reply = self.llm.chat(build_messages(question, results), model=model)
        return Answer(reply["content"], results, retrieval_ms, reply["seconds"])

    def stream_answer(self, question: str, results: list[SearchResult], model: str | None = None) -> Iterator[str]:
        yield from self.llm.stream(build_messages(question, results), model=model)
