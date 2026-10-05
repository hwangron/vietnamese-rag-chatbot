"""Giao diện chat hỏi đáp tài liệu PDF.

Chạy:  streamlit run app.py   →  mở http://localhost:8501
"""

import time
from pathlib import Path

import streamlit as st

from rag import config
from rag.llm import LLMError
from rag.pipeline import CITATION_RE, NOT_FOUND, RAGPipeline
from rag.retriever import MODES

# Các model dùng được với gói miễn phí của Ollama Cloud (kiểm tra ngày 04/10/2026)
LLM_CHOICES = list(dict.fromkeys([config.LLM_MODEL, "gpt-oss:20b", "gpt-oss:120b", "gemma4:31b", "nemotron-3-super"]))
MODE_LABELS = {"hybrid": "Kết hợp (BM25 + vector)", "vector": "Ngữ nghĩa (vector)", "bm25": "Từ khoá (BM25)"}
EXAMPLES = [
    "Tôi được nghỉ phép bao nhiêu ngày mỗi năm?",
    "Trưởng phòng đi công tác ở TP. Hồ Chí Minh được thanh toán khách sạn tối đa bao nhiêu một đêm?",
    "Có được dán mã nguồn vào ChatGPT để nhờ sửa lỗi không?",
]

st.set_page_config(page_title="Chatbot hỏi đáp tài liệu", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Đang nạp model embedding...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def render_sources(sources: list[dict], cited: set[int]) -> None:
    with st.expander(f"📎 Nguồn tham khảo ({len(sources)} đoạn, ✅ = được trích dẫn)"):
        for i, s in enumerate(sources, start=1):
            mark = "✅ " if i in cited else ""
            st.markdown(f"{mark}**[{i}] {s['source']} — trang {s['page']}** · {s['section'] or 'đầu tài liệu'}")
            st.caption(s["text"].replace("\n", "  \n"))


def render_message(message: dict) -> None:
    st.markdown(message["content"])
    if message.get("sources"):
        render_sources(message["sources"], set(message.get("cited", [])))
    if message.get("meta"):
        st.caption(message["meta"])


def answer_question(question: str, model: str, mode: str, top_k: int) -> dict | None:
    """Tìm đoạn liên quan, hiển thị câu trả lời dạng stream và trả về tin nhắn để lưu vào lịch sử."""
    start = time.perf_counter()
    results = pipe.search(question, top_k, mode)
    retrieval_ms = (time.perf_counter() - start) * 1000
    if not results:
        text = "Chưa có tài liệu nào được nạp. Hãy thêm file PDF ở thanh bên trái."
        st.markdown(text)
        return {"role": "assistant", "content": text}

    try:
        start = time.perf_counter()
        text = st.write_stream(pipe.stream_answer(question, results, model))
        llm_seconds = time.perf_counter() - start
    except LLMError as error:
        st.error(str(error))
        return None

    message = {
        "role": "assistant",
        "content": text,
        "sources": [{"source": r.chunk.source, "page": r.chunk.page, "section": r.chunk.section,
                     "text": r.chunk.text} for r in results],
        # Khi từ chối trả lời thì không có nguồn nào thực sự được dùng, dù model có ghi kèm [1]
        "cited": [] if NOT_FOUND in text else sorted({int(n) for n in CITATION_RE.findall(text)}),
        "meta": f"🔎 Tìm kiếm {retrieval_ms:.0f} ms · 🤖 {model} {llm_seconds:.1f} s · "
                f"{MODE_LABELS[mode]}, top-{top_k}",
    }
    render_sources(message["sources"], set(message["cited"]))
    st.caption(message["meta"])
    return message


pipe = get_pipeline()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)

# ---------------- Thanh bên: tài liệu và cài đặt ----------------
with st.sidebar:
    st.header("📚 Tài liệu")
    if notice := st.session_state.pop("notice", None):
        st.success(notice)
    sources = pipe.store.sources()
    if sources:
        st.markdown("\n".join(f"- {name} · {count} đoạn" for name, count in sources.items()))
    else:
        st.info("Chưa có tài liệu nào. Hãy tải file PDF lên bên dưới.")

    uploads = st.file_uploader("Thêm file PDF", type="pdf", accept_multiple_files=True,
                               key=f"uploader_{st.session_state.uploader_key}")
    if uploads and st.button("Nạp vào chỉ mục", type="primary", width="stretch"):
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        indexed = []
        for upload in uploads:
            path = config.DOCS_DIR / Path(upload.name).name
            path.write_bytes(upload.getvalue())
            try:
                with st.spinner(f"Đang xử lý {upload.name}..."):
                    indexed.append(f"{upload.name} ({pipe.index_file(path)} đoạn)")
            except ValueError as error:
                path.unlink(missing_ok=True)
                st.error(str(error))
        if indexed:
            st.session_state.notice = "Đã nạp: " + ", ".join(indexed)
            st.session_state.uploader_key += 1      # làm trống ô tải file
            st.rerun()

    st.divider()
    st.header("⚙️ Cài đặt")
    model = st.selectbox("Model LLM (Ollama Cloud)", LLM_CHOICES)
    mode = st.radio("Cách tìm kiếm", MODES, index=MODES.index(config.RETRIEVAL_MODE), format_func=MODE_LABELS.get)
    top_k = st.slider("Số đoạn tài liệu gửi cho LLM", 1, 10, config.TOP_K)
    if st.button("🗑️ Xoá lịch sử chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Embedding `{pipe.embed_model}` chạy trên máy · chunk {pipe.chunk_size} ký tự · "
               f"vector DB: ChromaDB")

# ---------------- Khung chat ----------------
st.title("📄 Chatbot hỏi đáp tài liệu")
st.caption("Trả lời dựa trên nội dung các file PDF, kèm trích dẫn nguồn. "
           "Kiến trúc RAG: tìm các đoạn liên quan trong tài liệu trước, rồi mới nhờ LLM trả lời.")

if not config.OLLAMA_API_KEY:
    st.error("Chưa có OLLAMA_API_KEY. Hãy tạo file `.env` từ `.env.example` rồi điền API key của Ollama.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message)

question = st.chat_input("Đặt câu hỏi về tài liệu...") or st.session_state.pop("pending_question", None)

if not st.session_state.messages and not question:
    st.markdown("**Thử một câu hỏi mẫu:**")
    for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if column.button(example, width="stretch"):
            st.session_state.pending_question = example
            st.rerun()

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        reply = answer_question(question, model, mode, top_k)
    if reply:
        st.session_state.messages += [{"role": "user", "content": question}, reply]
