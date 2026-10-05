"""Nạp (index) các file PDF trong data/docs vào vector DB.

Cách dùng:
    python ingest.py            # nạp / cập nhật tất cả PDF trong data/docs
    python ingest.py --reset    # xoá chỉ mục cũ rồi nạp lại từ đầu
"""

import argparse
import time

from rag import config
from rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Nạp file PDF vào vector DB")
    parser.add_argument("--reset", action="store_true", help="xoá chỉ mục cũ trước khi nạp")
    args = parser.parse_args()

    pdfs = sorted(config.DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Không có file PDF nào trong {config.DOCS_DIR}")
        return

    start = time.perf_counter()
    print(f"Đang nạp model embedding '{config.EMBED_MODEL}' (chỉ lần chạy đầu tiên mới tải về .cache/models)...")
    pipe = RAGPipeline()
    print(f"  xong sau {time.perf_counter() - start:.1f}s")
    if args.reset:
        pipe.reset()

    for pdf in pdfs:
        t = time.perf_counter()
        try:
            n = pipe.index_file(pdf)
            print(f"  ✓ {pdf.name}: {n} đoạn ({time.perf_counter() - t:.1f}s)")
        except ValueError as error:
            print(f"  ✗ {error}")

    print(f"Tổng cộng {pipe.store.count()} đoạn trong collection '{pipe.store.name}' "
          f"(chunk {pipe.chunk_size} ký tự, overlap {pipe.chunk_overlap}).")


if __name__ == "__main__":
    main()
