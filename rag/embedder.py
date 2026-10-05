"""Biến văn bản thành vector (embedding) bằng fastembed.

fastembed chạy model ở định dạng ONNX trên CPU, không cần PyTorch hay GPU.
File model được tải về một lần vào .cache/models/ trong thư mục dự án.
"""

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from .config import MODELS_DIR

# multilingual-e5-small (hỗ trợ ~100 ngôn ngữ, có tiếng Việt) chưa có sẵn trong danh sách
# của fastembed → đăng ký thêm từ bản ONNX chính thức trên Hugging Face (~470 MB).
_CUSTOM_MODELS = {
    "intfloat/multilingual-e5-small": dict(
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf="intfloat/multilingual-e5-small"),
        dim=384,
        model_file="onnx/model.onnx",
        description="Multilingual E5 small, ONNX, 512 input tokens",
        license="mit",
        size_in_gb=0.47,
    ),
}

# Họ model E5 được huấn luyện với tiền tố "query: " cho câu hỏi và "passage: " cho tài liệu;
# thiếu tiền tố thì chất lượng tìm kiếm giảm rõ rệt.
_PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


def _register_custom_model(model_name: str) -> None:
    if model_name not in _CUSTOM_MODELS:
        return
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    if model_name not in supported:
        TextEmbedding.add_custom_model(model=model_name, **_CUSTOM_MODELS[model_name])


class Embedder:
    def __init__(self, model_name: str):
        _register_custom_model(model_name)
        self.model_name = model_name
        self.model = TextEmbedding(model_name, cache_dir=str(MODELS_DIR))
        self.query_prefix, self.passage_prefix = _PREFIXES.get(model_name, ("", ""))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = [self.passage_prefix + t for t in texts]
        return np.array(list(self.model.embed(texts, batch_size=32)))

    def embed_query(self, text: str) -> np.ndarray:
        return next(iter(self.model.embed([self.query_prefix + text])))
