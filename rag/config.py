"""Cấu hình chung của dự án.

Mọi đường dẫn đều tính từ thư mục dự án (ROOT), nên dự án chạy được ở bất kỳ
đâu sau khi clone. Model, cache và vector DB đều nằm trong thư mục dự án.

Lưu ý: file này phải được import TRƯỚC fastembed / chromadb / huggingface_hub
để các biến môi trường bên dưới có tác dụng (rag/__init__.py đã làm việc này).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"            # model tải về, cache gọi LLM khi đánh giá
MODELS_DIR = CACHE_DIR / "models"      # file ONNX của model embedding
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"           # các file PDF cần hỏi đáp
CHROMA_DIR = DATA_DIR / "chroma"       # vector DB (tự sinh khi chạy ingest)

load_dotenv(ROOT / ".env")

# Giữ mọi file tải về của các thư viện bên trong thư mục dự án
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODELS_DIR))
# Tắt gửi thống kê ẩn danh của Hugging Face và ChromaDB
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# --- LLM trên Ollama Cloud ---
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:20b")

# --- Truy xuất (retrieval) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # số ký tự tối đa mỗi đoạn
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # số ký tự chồng lặp giữa 2 đoạn liền nhau
TOP_K = int(os.getenv("TOP_K", "4"))                    # số đoạn đưa vào prompt
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # hybrid | vector | bm25
