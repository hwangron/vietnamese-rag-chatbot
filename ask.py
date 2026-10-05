"""Hỏi nhanh từ dòng lệnh.

Cách dùng:
    python ask.py "Tôi được nghỉ phép bao nhiêu ngày mỗi năm?"
    python ask.py "..." --mode vector -k 6 --model gpt-oss:120b
"""

import argparse
import sys

from rag import config
from rag.llm import LLMError
from rag.pipeline import RAGPipeline
from rag.retriever import MODES


def main() -> None:
    parser = argparse.ArgumentParser(description="Hỏi đáp tài liệu từ dòng lệnh")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="số đoạn tài liệu đưa vào LLM")
    parser.add_argument("--mode", choices=MODES, default=config.RETRIEVAL_MODE)
    parser.add_argument("--model", default=config.LLM_MODEL)
    args = parser.parse_args()

    pipe = RAGPipeline()
    if pipe.store.count() == 0:
        sys.exit("Chưa có chỉ mục. Hãy chạy trước: python ingest.py")

    try:
        answer = pipe.answer(args.question, args.k, args.mode, args.model)
    except LLMError as error:
        sys.exit(f"Lỗi: {error}")

    print(f"\n{answer.text}\n")
    print("Nguồn:")
    for i, result in enumerate(answer.sources, start=1):
        print(f"  [{i}] {result.chunk.source} – trang {result.chunk.page} – {result.chunk.section or '(đầu tài liệu)'}")
    print(f"\n(Tìm kiếm {answer.retrieval_ms:.0f} ms · LLM {answer.llm_seconds:.1f} s · "
          f"{args.model} · {args.mode}, top-{args.k})")


if __name__ == "__main__":
    main()
