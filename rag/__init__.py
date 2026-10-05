"""Chatbot hỏi đáp tài liệu PDF theo kiến trúc RAG.

PDF → chia đoạn → embedding → ChromaDB → truy xuất (BM25 + vector) → LLM trả lời kèm trích dẫn.
"""

# Nạp cấu hình đầu tiên để biến môi trường (thư mục cache...) có hiệu lực trước mọi thư viện khác
from . import config  # noqa: F401
