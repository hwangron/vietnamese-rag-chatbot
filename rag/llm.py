"""Gọi LLM trên Ollama Cloud: model chạy trên server của Ollama, máy không phải tải model."""

import time
from collections.abc import Iterator

import httpx
import ollama

from . import config


class LLMError(RuntimeError):
    """Lỗi khi gọi LLM, kèm thông báo dễ hiểu để hiển thị cho người dùng."""


def _think_option(model: str) -> str | None:
    # gpt-oss luôn suy luận trước khi trả lời; mức "low" đủ cho hỏi đáp tài liệu và nhanh hơn nhiều
    return "low" if model.startswith("gpt-oss") else None


def _friendly_error(error: Exception, model: str) -> LLMError:
    if isinstance(error, ollama.ResponseError):
        status = error.status_code
        if status in (401, 403):
            return LLMError("API key của Ollama không hợp lệ hoặc đã bị thu hồi (lỗi 401). Kiểm tra OLLAMA_API_KEY trong file .env.")
        if status == 402:
            return LLMError(f"Model '{model}' cần gói trả phí của Ollama Cloud. Hãy chọn model khác, ví dụ gpt-oss:20b.")
        if status == 404:
            return LLMError(f"Không tìm thấy model '{model}' trên Ollama Cloud.")
        if status == 429:
            return LLMError("Đã vượt giới hạn lượt gọi của gói Ollama Cloud. Hãy thử lại sau ít phút.")
        if status >= 500:
            return LLMError(f"Máy chủ Ollama đang gặp sự cố (lỗi {status}). Hãy thử lại sau.")
        return LLMError(f"Ollama trả về lỗi: {error.error}")
    if isinstance(error, httpx.TimeoutException):
        return LLMError("Hết thời gian chờ phản hồi từ Ollama Cloud.")
    if isinstance(error, (ConnectionError, httpx.TransportError)):
        return LLMError("Không kết nối được tới Ollama Cloud. Hãy kiểm tra kết nối Internet.")
    return LLMError(f"Lỗi khi gọi LLM: {error}")


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, ollama.ResponseError):
        return error.status_code == 429 or error.status_code >= 500
    return isinstance(error, (httpx.TimeoutException, httpx.TransportError, ConnectionError))


class LLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None, host: str | None = None, timeout: float = 120):
        self.model = model or config.LLM_MODEL
        api_key = config.OLLAMA_API_KEY if api_key is None else api_key
        if not api_key:
            raise LLMError("Chưa có OLLAMA_API_KEY. Hãy tạo file .env từ .env.example rồi điền API key của Ollama.")
        self.client = ollama.Client(
            host=host or config.OLLAMA_HOST,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def chat(self, messages: list[dict], model: str | None = None, temperature: float = 0.0,
             json_mode: bool = False, retries: int = 3) -> dict:
        """Gọi LLM và chờ trả lời đầy đủ.

        Trả về dict gồm: content, seconds, prompt_tokens, output_tokens.
        Tự thử lại khi gặp lỗi tạm thời (quá tải, mất mạng, giới hạn lượt gọi).
        """
        model = model or self.model
        for attempt in range(retries):
            start = time.perf_counter()
            try:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    think=_think_option(model),
                    format="json" if json_mode else None,
                    options={"temperature": temperature},
                )
                return {
                    "content": (response.message.content or "").strip(),
                    "seconds": round(time.perf_counter() - start, 3),
                    "prompt_tokens": response.prompt_eval_count,
                    "output_tokens": response.eval_count,
                }
            except Exception as error:  # noqa: BLE001 — chuyển mọi lỗi thành LLMError dễ hiểu
                if attempt < retries - 1 and _is_retryable(error):
                    time.sleep(2 * 2**attempt)
                    continue
                raise _friendly_error(error, model) from error
        raise AssertionError("unreachable")

    def stream(self, messages: list[dict], model: str | None = None, temperature: float = 0.0) -> Iterator[str]:
        """Trả về từng phần câu trả lời ngay khi LLM sinh ra (hiển thị dần trên giao diện)."""
        model = model or self.model
        try:
            for part in self.client.chat(
                model=model,
                messages=messages,
                think=_think_option(model),
                stream=True,
                options={"temperature": temperature},
            ):
                if part.message.content:
                    yield part.message.content
        except Exception as error:  # noqa: BLE001
            raise _friendly_error(error, model) from error
