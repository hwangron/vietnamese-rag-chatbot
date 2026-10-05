"""Đánh giá chatbot RAG trên bộ câu hỏi eval/questions.json.

Phần 1 — Truy xuất (chạy hoàn toàn trên máy, không tốn lượt gọi API):
    so sánh cách chia đoạn × model embedding × cách tìm kiếm bằng Hit@k và MRR.
Phần 2 — Câu trả lời (gọi Ollama Cloud):
    so sánh LLM không có RAG với RAG; một LLM khác họ làm giám khảo chấm điểm (LLM-as-a-judge).

Chạy:
    python eval/run_eval.py                    # cả hai phần
    python eval/run_eval.py --retrieval-only   # chỉ phần 1
Kết quả: eval/results/report.md (bảng tổng hợp) và eval/results/details.json (chi tiết từng câu).
Kết quả gọi LLM được lưu trong .cache/llm_cache.json nên chạy lại không tốn thêm lượt gọi API.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import config  # noqa: E402  — nạp cấu hình trước các thư viện AI
from rag.chunker import Chunk, chunk_pages  # noqa: E402
from rag.embedder import Embedder  # noqa: E402
from rag.llm import LLMClient  # noqa: E402
from rag.loader import Page, load_pdf  # noqa: E402
from rag.pipeline import CITATION_RE, NOT_FOUND, SYSTEM_PROMPT, RAGPipeline, build_messages  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.store import VectorStore, collection_name  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
LLM_CACHE = config.CACHE_DIR / "llm_cache.json"

EMBED_MODELS = {
    "intfloat/multilingual-e5-small": "e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "MiniLM-L12",
}
CHUNKERS = {
    "fixed": "Cắt cố định theo số ký tự",
    "structure": "Theo cấu trúc (Chương/Điều)",
    "structure+header": "Theo cấu trúc + dòng ngữ cảnh",
}
CHUNK_SIZES = (400, 800)
MODES = ("bm25", "vector", "hybrid")
MODE_NAMES = {"bm25": "BM25 (từ khoá)", "vector": "Vector (ngữ nghĩa)", "hybrid": "Hybrid (BM25 + vector)"}
EVAL_TOP_K = 5

# Prompt phiên bản 1 (trước khi phân tích lỗi), giữ lại để so sánh với phiên bản 2 trong rag/pipeline.py
PROMPT_V1 = f"""Bạn là trợ lý hỏi đáp dựa trên tài liệu nội bộ của công ty.
Quy tắc bắt buộc:
1. Chỉ dùng thông tin trong phần TÀI LIỆU được cung cấp. Không dùng kiến thức bên ngoài, không suy đoán.
2. Nếu tài liệu không có thông tin để trả lời, chỉ trả lời đúng một câu: "{NOT_FOUND}"
3. Sau mỗi ý, ghi số thứ tự của đoạn tài liệu đã dùng trong ngoặc vuông, ví dụ [1] hoặc [2][3].
4. Trả lời ngắn gọn, chính xác, bằng tiếng Việt."""

# (tên hiển thị, system prompt — None nghĩa là không dùng RAG, model sinh câu trả lời)
SYSTEMS = [
    ("LLM không RAG", None, "gpt-oss:20b"),
    ("RAG + gpt-oss:20b, prompt v1", PROMPT_V1, "gpt-oss:20b"),
    ("RAG + gpt-oss:20b, prompt v2", SYSTEM_PROMPT, "gpt-oss:20b"),
    ("RAG + gpt-oss:120b, prompt v2", SYSTEM_PROMPT, "gpt-oss:120b"),
]
DEFAULT_SYSTEM = f"RAG + {config.LLM_MODEL}, prompt v2"   # đúng cấu hình ứng dụng đang dùng
JUDGE_MODEL = "gemma4:31b"  # khác họ với gpt-oss để giảm thiên vị "tự chấm bài mình"

BASELINE_SYSTEM = ("Bạn là trợ lý hỏi đáp nội bộ của Công ty Cổ phần Công nghệ Lam Tinh. "
                   "Trả lời câu hỏi của nhân viên ngắn gọn, chính xác, bằng tiếng Việt. "
                   f'Nếu không chắc chắn về thông tin, chỉ trả lời: "{NOT_FOUND}"')

JUDGE_PROMPT = """Bạn là giám khảo chấm câu trả lời của một chatbot hỏi đáp tài liệu nội bộ công ty.

Câu hỏi: {question}
Ý chính cần có: {answer}
Chi tiết bổ sung (không bắt buộc): {details}
Câu trả lời của chatbot: {response}

Chấm 2 tiêu chí:
1. "diem" — độ chính xác của ý chính:
   - "dung": nêu đúng ý chính và không có thông tin nào sai hoặc mâu thuẫn.
   - "mot_phan": chỉ đúng một phần của ý chính.
   - "sai": sai ý chính, có thông tin bịa đặt, hoặc nói không tìm thấy trong khi ý chính có trong tài liệu.
2. "day_du" — true nếu câu trả lời có nêu các chi tiết bổ sung (hoặc không có chi tiết bổ sung nào), false nếu bỏ sót.
Không trừ điểm vì cách diễn đạt, độ dài hay các ký hiệu trích dẫn như [1].

Chỉ trả về JSON: {{"diem": "dung" | "mot_phan" | "sai", "day_du": true | false, "ly_do": "<một câu ngắn>"}}"""

UNANSWERABLE_REFERENCE = ("Tài liệu KHÔNG có thông tin này. Câu trả lời đúng là nói rõ không tìm thấy thông tin. "
                          "Nếu chatbot đưa ra con số, chính sách hoặc khẳng định cụ thể thì là bịa đặt (sai).")


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()


def is_relevant(chunk: Chunk, question: dict) -> bool:
    """Một đoạn được coi là đúng nếu thuộc đúng file và chứa nguyên văn câu bằng chứng (evidence)."""
    return chunk.source == question["source"] and norm(question["evidence"]) in norm(chunk.text)


def is_refusal(answer: str) -> bool:
    return "không tìm thấy" in norm(answer)


def has_citation(answer: str) -> bool:
    return bool(CITATION_RE.search(answer))


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phần 1 — Truy xuất
# ---------------------------------------------------------------------------
def chunk_fixed(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    """Cách chia "ngây thơ" để so sánh: cắt mỗi trang thành các cửa sổ `size` ký tự,
    không quan tâm ranh giới câu hay mục."""
    chunks = []
    for page in pages:
        text = " ".join(page.blocks)
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(Chunk(f"{page.source}:p{page.number}:f{len(chunks)}", text[start:end],
                                page.source, page.number, ""))
            if end >= len(text):
                break
            start = end - overlap
    return chunks


def make_chunks(docs: list[tuple[str, list[Page]]], chunker: str, size: int) -> list[Chunk]:
    overlap = round(size * config.CHUNK_OVERLAP / config.CHUNK_SIZE)  # cùng tỉ lệ chồng lặp với cấu hình mặc định
    chunks: list[Chunk] = []
    for title, pages in docs:
        if chunker == "fixed":
            chunks += chunk_fixed(pages, size, overlap)
        else:
            chunks += chunk_pages(pages, title, size, overlap, add_header=chunker == "structure+header")
    return chunks


def evaluate_retrieval(retriever: Retriever, questions: list[dict], mode: str) -> dict:
    ranks: dict[str, int | None] = {}
    seconds = []
    for q in questions:
        start = time.perf_counter()
        results = retriever.search(q["question"], EVAL_TOP_K, mode)
        seconds.append(time.perf_counter() - start)
        ranks[q["id"]] = next((i for i, r in enumerate(results, start=1) if is_relevant(r.chunk, q)), None)
    n = len(questions)
    return {
        **{f"hit@{k}": sum(1 for r in ranks.values() if r and r <= k) / n for k in (1, 3, 5)},
        "mrr": sum(1 / r for r in ranks.values() if r) / n,
        "query_ms": mean(seconds) * 1000,
        "ranks": ranks,
    }


def run_retrieval(docs: list[tuple[str, list[Page]]], questions: list[dict]) -> list[dict]:
    rows = []
    bm25_done: set[tuple[str, int]] = set()     # BM25 không phụ thuộc model embedding → chỉ chạy 1 lần
    for model_name, short in EMBED_MODELS.items():
        print(f"\n[Phần 1] Model embedding: {model_name}")
        embedder = Embedder(model_name)
        for chunker in CHUNKERS:
            for size in CHUNK_SIZES:
                chunks = make_chunks(docs, chunker, size)
                coverage = sum(any(is_relevant(c, q) for c in chunks) for q in questions) / len(questions)
                start = time.perf_counter()
                embeddings = embedder.embed_documents([c.text for c in chunks])
                index_seconds = time.perf_counter() - start

                store = VectorStore(collection_name(model_name, size, prefix=f"eval-{chunker.replace('+', '-')}"))
                store.reset()
                store.add(chunks, embeddings)
                retriever = Retriever(store, embedder)
                for mode in MODES:
                    if mode == "bm25":
                        if (chunker, size) in bm25_done:
                            continue
                        bm25_done.add((chunker, size))
                    metrics = evaluate_retrieval(retriever, questions, mode)
                    rows.append({
                        "embed_model": "—" if mode == "bm25" else short, "chunker": chunker,
                        "chunk_size": size, "mode": mode, "n_chunks": len(chunks), "coverage": coverage,
                        "index_seconds": 0.0 if mode == "bm25" else index_seconds, **metrics,
                    })
                    print(f"  {chunker:17} {size:4}  {mode:6}  {rows[-1]['embed_model']:10}  "
                          f"Hit@1={metrics['hit@1']:.2f}  Hit@5={metrics['hit@5']:.2f}  MRR={metrics['mrr']:.3f}")
    return rows


# ---------------------------------------------------------------------------
# Phần 2 — Chất lượng câu trả lời
# ---------------------------------------------------------------------------
class CachedLLM:
    """Gọi LLM có lưu kết quả trên đĩa: chạy lại đánh giá không tốn thêm lượt gọi API."""

    def __init__(self):
        self.client = LLMClient()
        self.cache = json.loads(LLM_CACHE.read_text("utf-8")) if LLM_CACHE.exists() else {}
        self.new_calls = 0

    def chat(self, messages: list[dict], model: str, json_mode: bool = False) -> dict:
        key = hashlib.sha256(json.dumps([model, messages, json_mode], ensure_ascii=False).encode()).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.client.chat(messages, model=model, json_mode=json_mode)
            self.new_calls += 1
            LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
            LLM_CACHE.write_text(json.dumps(self.cache, ensure_ascii=False), "utf-8")
        return self.cache[key]


def parse_verdict(text: str) -> tuple[str, bool | None, str]:
    """Đọc JSON của giám khảo → (điểm, có đầy đủ chi tiết không, lý do)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)  # một số model bọc JSON trong ```json ... ```
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("diem", "")).strip().lower()
    if verdict not in ("dung", "mot_phan", "sai"):
        found = re.search(r"\b(dung|mot_phan|sai)\b", text.lower())
        verdict = found.group(1) if found else "loi"
    complete = data.get("day_du")
    if isinstance(complete, str):
        complete = complete.strip().lower() == "true"
    return verdict, complete if isinstance(complete, bool) else None, str(data.get("ly_do", "")).strip()


def judge(llm: CachedLLM, question: dict, response: str) -> tuple[str, bool | None, str]:
    if question.get("unanswerable"):
        answer, details = UNANSWERABLE_REFERENCE, "(không có)"
    else:
        answer, details = question["answer"], question.get("details", "(không có)")
    prompt = JUDGE_PROMPT.format(question=question["question"], answer=answer, details=details, response=response)
    reply = llm.chat([{"role": "user", "content": prompt}], model=JUDGE_MODEL, json_mode=True)
    return parse_verdict(reply["content"])


def run_answers(questions: list[dict]) -> dict:
    print("\n[Phần 2] Dựng pipeline RAG với cấu hình mặc định (giống ứng dụng)...")
    pipe = RAGPipeline(persist_dir=None)          # vector DB trong RAM, không đụng tới chỉ mục của ứng dụng
    for pdf in sorted(config.DOCS_DIR.glob("*.pdf")):
        pipe.index_file(pdf)
    llm = CachedLLM()

    records: dict[str, list[dict]] = {name: [] for name, _, _ in SYSTEMS}
    print("  Cột kết quả chấm: " + "  |  ".join(name for name, _, _ in SYSTEMS))
    for i, q in enumerate(questions, start=1):
        start = time.perf_counter()
        results = pipe.search(q["question"])
        retrieval_ms = (time.perf_counter() - start) * 1000
        rank = None if q.get("unanswerable") else next(
            (j for j, r in enumerate(results, start=1) if is_relevant(r.chunk, q)), None)

        for name, prompt, model in SYSTEMS:
            use_rag = prompt is not None
            if use_rag:
                messages = build_messages(q["question"], results, prompt)
            else:
                messages = [{"role": "system", "content": BASELINE_SYSTEM},
                            {"role": "user", "content": q["question"]}]
            reply = llm.chat(messages, model=model)
            verdict, complete, reason = judge(llm, q, reply["content"])
            records[name].append({
                "id": q["id"], "question": q["question"], "reference": q.get("answer"),
                "details": q.get("details"), "unanswerable": bool(q.get("unanswerable")),
                "answer": reply["content"], "verdict": verdict, "complete": complete, "reason": reason,
                "refusal": is_refusal(reply["content"]),
                "citation": has_citation(reply["content"]), "llm_seconds": reply["seconds"],
                "prompt_tokens": reply.get("prompt_tokens"),
                "retrieval_ms": retrieval_ms if use_rag else None,
                "evidence_rank": rank if use_rag else None,
                "sources": [f"{r.chunk.source} tr.{r.chunk.page}" for r in results] if use_rag else None,
            })
        verdicts = "  ".join(f"{records[name][-1]['verdict']:8}" for name, _, _ in SYSTEMS)
        print(f"  [{i:2}/{len(questions)}] {q['id']}  {verdicts}", flush=True)

    print(f"  Số lượt gọi API mới: {llm.new_calls} (phần còn lại lấy từ cache)")
    return records


VERDICT_NAMES = {"dung": "Đúng", "mot_phan": "Một phần", "sai": "Sai", "loi": "Lỗi chấm"}


def summarize(records: list[dict]) -> dict:
    answerable = [r for r in records if not r["unanswerable"]]
    unanswerable = [r for r in records if r["unanswerable"]]
    with_details = [r for r in answerable if r["details"]]
    answered = [r for r in answerable if not r["refusal"]]
    tokens = [r["prompt_tokens"] for r in records if r["prompt_tokens"]]
    return {
        "dung": sum(r["verdict"] == "dung" for r in answerable) / len(answerable),
        "mot_phan": sum(r["verdict"] == "mot_phan" for r in answerable) / len(answerable),
        "sai": sum(r["verdict"] in ("sai", "loi") for r in answerable) / len(answerable),
        "day_du": sum(r["complete"] is True for r in with_details) / len(with_details),
        "n_details": len(with_details),
        "tu_choi_dung": sum(r["verdict"] == "dung" for r in unanswerable) / len(unanswerable),
        "bia_dat": sum(r["verdict"] == "sai" and not r["refusal"] for r in records),
        "trich_dan": sum(r["citation"] for r in answered) / len(answered) if answered else 0.0,
        "llm_seconds": mean(r["llm_seconds"] for r in records),
        "prompt_tokens": mean(tokens) if tokens else None,
    }


# ---------------------------------------------------------------------------
# Báo cáo
# ---------------------------------------------------------------------------
def write_report(docs_info: dict, questions: list[dict], retrieval: list[dict], answers: dict | None) -> str:
    n_ans = sum(not q.get("unanswerable") for q in questions)
    n_una = len(questions) - n_ans
    is_default = lambda r: (r["chunker"] == "structure+header" and r["chunk_size"] == config.CHUNK_SIZE  # noqa: E731
                            and r["mode"] == config.RETRIEVAL_MODE and r["embed_model"] == "e5-small")
    out = [
        "# Kết quả đánh giá chatbot RAG",
        "",
        f"_Chạy lúc {datetime.now():%d/%m/%Y %H:%M} bằng `python eval/run_eval.py`._",
        "",
        f"- **Dữ liệu:** {docs_info['files']} file PDF tiếng Việt ({docs_info['pages']} trang) của một công ty hư cấu, "
        f"nên LLM không thể biết trước nội dung.",
        f"- **Bộ câu hỏi:** {n_ans} câu có đáp án trong tài liệu + {n_una} câu **không có** đáp án "
        f"(để kiểm tra chatbot có bịa hay không). Câu hỏi được diễn đạt khác với văn bản gốc.",
        f"- **Cấu hình mặc định của ứng dụng (⭐):** chia đoạn theo cấu trúc + dòng ngữ cảnh, "
        f"{config.CHUNK_SIZE} ký tự, embedding `{config.EMBED_MODEL}`, tìm kiếm {config.RETRIEVAL_MODE}, "
        f"top-{config.TOP_K}, LLM `{config.LLM_MODEL}`.",
        "",
        "## 1. Truy xuất (retrieval)",
        "",
        f"Một câu hỏi được tính là **tìm đúng** nếu trong top-k đoạn trả về có đoạn chứa nguyên văn câu bằng chứng "
        f"(evidence) của câu hỏi đó. **Hit@k** = tỉ lệ câu hỏi tìm đúng trong top-k; "
        f"**MRR** = trung bình của 1/(thứ hạng của đoạn đúng đầu tiên), càng gần 1 càng tốt.",
        "",
        f"### 1.1. Cách tìm kiếm và model embedding (chia đoạn theo cấu trúc + ngữ cảnh, {config.CHUNK_SIZE} ký tự)",
        "",
    ]
    subset = [r for r in retrieval if r["chunker"] == "structure+header" and r["chunk_size"] == config.CHUNK_SIZE]
    subset.sort(key=lambda r: (MODES.index(r["mode"]), r["embed_model"]))
    out.append(md_table(
        ["Cách tìm kiếm", "Model embedding", "Hit@1", "Hit@3", "Hit@5", "MRR", "Thời gian / truy vấn"],
        [[MODE_NAMES[r["mode"]] + (" ⭐" if is_default(r) else ""), r["embed_model"], pct(r["hit@1"]),
          pct(r["hit@3"]), pct(r["hit@5"]), f"{r['mrr']:.3f}",
          f"{r['query_ms']:.1f} ms" if r["query_ms"] < 10 else f"{r['query_ms']:.0f} ms"] for r in subset]))

    out += ["", "### 1.2. Cách chia đoạn (embedding e5-small)", ""]
    rows = []
    for chunker, label in CHUNKERS.items():
        for size in CHUNK_SIZES:
            by_mode = {r["mode"]: r for r in retrieval if r["chunker"] == chunker and r["chunk_size"] == size
                       and r["embed_model"] in ("e5-small", "—")}
            vec, hyb = by_mode["vector"], by_mode["hybrid"]
            star = " ⭐" if chunker == "structure+header" and size == config.CHUNK_SIZE else ""
            rows.append([label + star, size, vec["n_chunks"], pct(vec["coverage"]),
                         pct(vec["hit@1"]), f"{vec['mrr']:.3f}", pct(hyb["hit@1"]), f"{hyb['mrr']:.3f}"])
    out.append(md_table(["Cách chia đoạn", "Kích thước", "Số đoạn", "Bằng chứng còn nguyên",
                         "Vector Hit@1", "Vector MRR", "Hybrid Hit@1", "Hybrid MRR"], rows))
    out += ["", "_\"Bằng chứng còn nguyên\" = tỉ lệ câu hỏi mà câu bằng chứng nằm trọn trong ít nhất một đoạn "
                "(không bị cắt đôi)._"]

    if answers:
        summaries = {name: summarize(records) for name, records in answers.items()}
        n_details = summaries[DEFAULT_SYSTEM]["n_details"]
        out += ["", "## 2. Chất lượng câu trả lời", "",
                f"Giám khảo là LLM `{JUDGE_MODEL}` (khác họ với gpt-oss để giảm thiên vị \"tự chấm bài mình\"). "
                f"Đáp án chuẩn của mỗi câu gồm **ý chính** (bắt buộc) và có thể có **chi tiết bổ sung** như điều kiện, "
                f"ngoại lệ, thời hạn. Giám khảo chấm ý chính theo 3 mức **Đúng / Một phần / Sai**, và **Đầy đủ** nếu "
                f"câu trả lời nêu được chi tiết bổ sung. **Bịa đặt** = câu trả lời sai nhưng không thừa nhận là không biết.",
                ""]
        out.append(md_table(
            ["Hệ thống", f"Đúng ({n_ans} câu)", "Một phần", "Sai", f"Đầy đủ chi tiết ({n_details} câu)",
             f"Từ chối đúng ({n_una} câu không có đáp án)", "Số câu bịa đặt", "Có trích dẫn",
             "Thời gian LLM TB", "Token prompt TB"],
            [[name + (" ⭐" if name == DEFAULT_SYSTEM else ""), pct(s["dung"]), pct(s["mot_phan"]), pct(s["sai"]),
              pct(s["day_du"]), pct(s["tu_choi_dung"]), f"{s['bia_dat']}/{len(questions)}", pct(s["trich_dan"]),
              f"{s['llm_seconds']:.1f} s", f"{s['prompt_tokens']:.0f}" if s["prompt_tokens"] else "—"]
             for name, s in summaries.items()]))

        v1, v2 = summaries[SYSTEMS[1][0]], summaries[SYSTEMS[2][0]]
        out += ["", "### 2.1. Cải tiến prompt sau khi phân tích lỗi", "",
                "Phân tích lỗi lần đầu cho thấy prompt v1 trả lời đúng ý chính nhưng hay bỏ sót điều kiện quan trọng, "
                "ví dụ trả lời \"tối đa 2 ngày làm từ xa\" mà không nói \"chỉ áp dụng sau thử việc\". Prompt v2 "
                "(`rag/pipeline.py`) thêm quy tắc: *trả lời thẳng vào câu hỏi trước, sau đó nêu ngắn gọn các điều kiện, "
                "ngoại lệ hoặc thời hạn quan trọng liên quan trực tiếp tới câu hỏi*.", "",
                f"- Đầy đủ chi tiết: {pct(v1['day_du'])} → **{pct(v2['day_du'])}**",
                f"- Đúng ý chính: {pct(v1['dung'])} → **{pct(v2['dung'])}**",
                f"- Từ chối đúng câu không có đáp án: {pct(v1['tu_choi_dung'])} → **{pct(v2['tu_choi_dung'])}**",
                f"- Thời gian LLM trung bình: {v1['llm_seconds']:.1f} s → {v2['llm_seconds']:.1f} s"]

        issues = [r for r in answers[DEFAULT_SYSTEM]
                  if r["verdict"] != "dung" or (r["details"] and r["complete"] is False)]
        out += ["", f"### 2.2. Phân tích lỗi của cấu hình mặc định ({DEFAULT_SYSTEM})", ""]
        if issues:
            out.append(md_table(
                ["Câu", "Câu hỏi", "Câu trả lời của chatbot", "Ý chính", "Đầy đủ", "Đoạn đúng ở hạng",
                 "Nhận xét của giám khảo"],
                [[r["id"], r["question"], r["answer"].replace("\n", " ").replace("|", "/")[:260],
                  VERDICT_NAMES[r["verdict"]], {True: "✓", False: "✗"}.get(r["complete"], "—"),
                  "—" if r["unanswerable"] else (r["evidence_rank"] or f"ngoài top-{config.TOP_K}"),
                  r["reason"].replace("|", "/")[:180]] for r in issues]))
        else:
            out.append("Tất cả câu trả lời đều đúng ý chính và đầy đủ chi tiết.")

        baseline = answers[SYSTEMS[0][0]]
        out += ["", "### 2.3. Ví dụ: cùng một câu hỏi, có và không có RAG", ""]
        rag = {r["id"]: r for r in answers[DEFAULT_SYSTEM]}
        examples = [r for r in baseline if not r["unanswerable"] and not r["refusal"] and r["verdict"] == "sai"][:3]
        examples = examples or [r for r in baseline if not r["unanswerable"]][:3]
        for r in examples:
            out += [f"**{r['question']}**", "",
                    f"- Không RAG: _{' '.join(r['answer'].split())}_ → **{VERDICT_NAMES[r['verdict']]}**",
                    f"- RAG: _{' '.join(rag[r['id']]['answer'].split())}_ → "
                    f"**{VERDICT_NAMES[rag[r['id']]['verdict']]}**",
                    f"- Đáp án chuẩn: {r['reference']}", ""]

    report = "\n".join(out) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.md").write_text(report, "utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá chatbot RAG")
    parser.add_argument("--retrieval-only", action="store_true", help="chỉ chạy phần 1 (không gọi API)")
    args = parser.parse_args()

    questions = json.loads((ROOT / "eval" / "questions.json").read_text("utf-8"))
    answerable = [q for q in questions if not q.get("unanswerable")]
    pdfs = sorted(config.DOCS_DIR.glob("*.pdf"))
    loaded = [load_pdf(p) for p in pdfs]
    docs = [(title, pages) for title, pages in loaded]
    docs_info = {"files": len(pdfs), "pages": sum(len(pages) for _, pages in loaded)}

    # Kiểm tra bộ câu hỏi: câu bằng chứng phải có thật trong tài liệu nguồn
    full_text = {pages[0].source: norm(" ".join(b for p in pages for b in p.blocks)) for _, pages in docs}
    missing = [q["id"] for q in answerable if norm(q["evidence"]) not in full_text.get(q["source"], "")]
    if missing:
        sys.exit(f"Câu bằng chứng không có trong tài liệu: {missing}")
    print(f"Bộ câu hỏi hợp lệ: {len(answerable)} câu có đáp án, {len(questions) - len(answerable)} câu không có đáp án.")

    started = time.perf_counter()
    retrieval = run_retrieval(docs, answerable)
    answers = None if args.retrieval_only else run_answers(questions)

    write_report(docs_info, questions, retrieval, answers)
    (RESULTS_DIR / "details.json").write_text(
        json.dumps({"retrieval": retrieval, "answers": answers}, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nXong sau {time.perf_counter() - started:.0f}s → {RESULTS_DIR.relative_to(ROOT)}/report.md, details.json")


if __name__ == "__main__":
    main()
