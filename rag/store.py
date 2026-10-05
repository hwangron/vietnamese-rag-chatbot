"""Lưu và tìm kiếm vector bằng ChromaDB.

Mỗi tổ hợp (model embedding, kích thước chunk) dùng một collection riêng, vì vector của
model này không so sánh được với vector của model khác.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from .chunker import Chunk


def collection_name(embed_model: str, chunk_size: int, prefix: str = "docs") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", embed_model.split("/")[-1].lower()).strip("-")[:40]
    digest = hashlib.sha1(embed_model.encode()).hexdigest()[:6]
    return f"{prefix}-{slug}-{chunk_size}-{digest}"


class VectorStore:
    def __init__(self, name: str, persist_dir: str | Path | None = None):
        """persist_dir=None → chỉ lưu trong RAM (dùng khi đánh giá nhiều cấu hình)."""
        settings = Settings(anonymized_telemetry=False)
        if persist_dir is None:
            self.client = chromadb.EphemeralClient(settings=settings)
        else:
            self.client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
        self.name = name
        self._open()

    def _open(self) -> None:
        self.collection = self.client.get_or_create_collection(
            self.name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # tự tạo embedding bằng fastembed, không dùng model mặc định của Chroma
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page, "section": c.section} for c in chunks],
        )

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Trả về k đoạn gần nhất kèm độ tương đồng cosine (càng gần 1 càng giống)."""
        k = min(k, self.count())
        if k == 0:
            return []
        res = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (self._to_chunk(id_, doc, meta), 1 - dist)
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def all_chunks(self) -> list[Chunk]:
        res = self.collection.get(include=["documents", "metadatas"])
        chunks = [self._to_chunk(*row) for row in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: (c.source, c.page, c.id))

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def sources(self) -> dict[str, int]:
        """Tên các file đã nạp và số đoạn của mỗi file."""
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return dict(sorted(Counter(m["source"] for m in metas).items()))

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self._open()

    @staticmethod
    def _to_chunk(id_: str, text: str, meta: dict) -> Chunk:
        return Chunk(id=id_, text=text, source=meta["source"], page=meta["page"], section=meta["section"])
