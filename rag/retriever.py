"""Tìm các đoạn tài liệu liên quan tới câu hỏi.

Ba cách tìm:
- "vector": so sánh ý nghĩa bằng embedding (hiểu được câu hỏi diễn đạt khác tài liệu).
- "bm25":   so khớp từ khoá (mạnh với con số, tên riêng, thuật ngữ như "VPN", "minibar").
- "hybrid": chạy cả hai rồi gộp thứ hạng bằng Reciprocal Rank Fusion (RRF).
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .embedder import Embedder
from .store import VectorStore

MODES = ("hybrid", "vector", "bm25")


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    """Tách từ đơn giản cho tiếng Việt: các âm tiết + từng cặp âm tiết liền nhau.

    Tiếng Việt có nhiều từ ghép ("nghỉ phép", "thử việc"); thêm cặp âm tiết như "nghỉ_phép"
    giúp BM25 phân biệt được "nghỉ phép" với "nghỉ ốm" mà không cần thư viện tách từ.
    """
    syllables = re.findall(r"\w+", text.lower())
    return syllables + [f"{a}_{b}" for a, b in zip(syllables, syllables[1:])]


def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], k: int, rrf_k: int = 60) -> list[SearchResult]:
    """Gộp nhiều bảng xếp hạng: điểm của một đoạn = tổng 1 / (60 + thứ hạng) ở mỗi bảng.

    RRF chỉ dùng thứ hạng nên không cần chuẩn hoá điểm BM25 và điểm cosine về cùng thang đo.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk.id] = scores.get(result.chunk.id, 0.0) + 1 / (rrf_k + rank)
            chunks[result.chunk.id] = result.chunk
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [SearchResult(chunks[i], scores[i]) for i in best]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.refresh()

    def refresh(self) -> None:
        """Dựng lại chỉ mục BM25; gọi sau khi thêm hoặc xoá tài liệu."""
        self.chunks = self.store.all_chunks()
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[SearchResult]:
        if mode not in MODES:
            raise ValueError(f"mode phải là một trong {MODES}")
        if not self.chunks:
            return []
        if mode == "vector":
            return self._vector_search(query, k)
        if mode == "bm25":
            return self._bm25_search(query, k)
        candidates = max(4 * k, 20)  # lấy rộng ở mỗi bên rồi mới gộp
        return reciprocal_rank_fusion(
            [self._vector_search(query, candidates), self._bm25_search(query, candidates)], k
        )

    def _vector_search(self, query: str, k: int) -> list[SearchResult]:
        hits = self.store.search(self.embedder.embed_query(query), k)
        return [SearchResult(chunk, score) for chunk, score in hits]

    def _bm25_search(self, query: str, k: int) -> list[SearchResult]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]
