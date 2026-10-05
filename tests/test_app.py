"""Kiểm thử E2E giao diện Streamlit: mở app, đặt câu hỏi, kiểm tra câu trả lời có trích dẫn và nguồn.

Cần: đã chạy `python ingest.py` và có OLLAMA_API_KEY trong .env (test gọi LLM thật).
Chạy:  python -m pytest tests/test_app.py -q -s
"""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.pipeline import CITATION_RE  # noqa: E402

APP = ROOT / "app.py"


def ask(question: str) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    assert not app.exception, app.exception
    app.chat_input[0].set_value(question).run()
    assert not app.exception, app.exception
    return app


def assistant_text(app: AppTest) -> str:
    message = [m for m in app.chat_message if m.name == "assistant"][-1]
    return " ".join(md.value for md in message.markdown)


def test_sidebar_lists_indexed_documents():
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    sidebar = " ".join(md.value for md in app.sidebar.markdown)
    assert "so_tay_nhan_vien.pdf" in sidebar and "quy_dinh_cong_tac_phi.pdf" in sidebar


def test_answer_from_table_with_citation():
    app = ask("Trưởng phòng đi công tác ở TP. Hồ Chí Minh được thanh toán khách sạn tối đa bao nhiêu một đêm?")
    answer = assistant_text(app)
    print("\nTrả lời:", answer)
    assert "1.800.000" in answer          # giá trị nằm trong bảng của PDF
    assert CITATION_RE.search(answer)     # có trích dẫn nguồn dạng [1]
    assert any("Nguồn tham khảo" in e.label for e in app.expander)


def test_refuses_when_information_is_missing():
    answer = assistant_text(ask("Công ty có xe đưa đón nhân viên không?"))
    print("\nTrả lời:", answer)
    assert "không tìm thấy" in answer.lower()
