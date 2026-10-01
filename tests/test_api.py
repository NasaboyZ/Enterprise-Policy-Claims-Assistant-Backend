import io
from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen.canvas import Canvas

from app.agent import AgentResult, GroundedAnswer, InsuranceAgent
from app.config import Settings
from app.main import create_app
from app.rag_engine import RagEngine
from app.reports import EvaluationReport, save_report
from app.uploads import MAX_UPLOAD_BYTES, upload_pdf


REQUEST = {"query": "Was ist versichert?", "claim": {"customer_age": 40, "claim_amount": 100, "claim_type": "water"}}


def pdf_bytes(text="Wasser ist versichert."):
    buffer = io.BytesIO()
    canvas = Canvas(buffer, invariant=1)
    canvas.drawString(30, 800, text)
    canvas.save()
    return buffer.getvalue()


@pytest.fixture
def setup(tmp_path, embeddings):
    settings = replace(Settings(), data_dir=tmp_path / "data", chroma_dir=tmp_path / "chroma", reports_dir=tmp_path / "reports")
    rag = RagEngine(settings, embeddings)
    agent = Mock()
    agent.run.return_value = AgentResult(status="manual_review", ml_score=0.9, final_answer="Manuelle Prüfung", sources=[])
    return settings, rag, agent, TestClient(create_app(settings, rag=rag, agent=agent))


def test_health_chat_validation_and_manual_review(setup):
    _, _, agent, client = setup
    assert client.get("/health").json() == {"status": "ok"}
    agent.run.assert_not_called()
    assert client.post("/api/chat", json={}).status_code == 422
    response = client.post("/api/chat", json=REQUEST)
    assert response.status_code == 200 and response.json()["status"] == "manual_review"
    assert agent.run.call_args.args[0].query == REQUEST["query"]


@pytest.mark.parametrize("code,status", [("aion_quota_exceeded", 429), ("aion_invalid_response", 502),
                                        ("aion_auth_failed", 503), ("model_missing", 503)])
def test_chat_errors_have_http_status(setup, code, status):
    _, _, agent, client = setup
    agent.run.return_value = AgentResult(status="error", ml_score=None, final_answer="Bereinigter Fehler", sources=[], error_code=code)
    response = client.post("/api/chat", json=REQUEST)
    assert response.status_code == status and response.json()["error_code"] == code


def test_unexpected_agent_error_is_safe(setup):
    _, _, agent, client = setup
    agent.run.side_effect = RuntimeError("secret-value")
    response = client.post("/api/chat", json=REQUEST)
    assert response.status_code == 503 and "secret-value" not in response.text


def test_upload_indexes_deduplicates_and_preserves_other_documents(setup):
    settings, rag, _, client = setup
    content = pdf_bytes()
    first = client.post("/api/upload", files={"file": ("terms.pdf", content, "application/pdf")})
    assert first.status_code == 201
    filename = first.json()["filename"]
    assert (settings.data_dir / filename).read_bytes() == content
    assert rag.retrieve("Wasser")[0].filename == filename
    duplicate = client.post("/api/upload", files={"file": ("renamed.pdf", content, "application/pdf")})
    assert duplicate.status_code == 200 and duplicate.json()["status"] == "duplicate"
    second = client.post("/api/upload", files={"file": ("terms.pdf", pdf_bytes("Feuer ist versichert."), "application/pdf")})
    assert second.status_code == 201 and second.json()["filename"] != filename
    assert len(rag._store(create=False).get(include=[])["ids"]) == 2
    assert rag.index()["added"] == 0
    assert not list(settings.data_dir.glob(".upload-*"))


@pytest.mark.parametrize("filename,content,status", [
    ("../bad.pdf", pdf_bytes(), 422), ("bad\\file.pdf", pdf_bytes(), 422),
    ("file.txt", pdf_bytes(), 422), ("file.pdf", b"invalid", 422),
    ("file.pdf", b"%PDF-broken", 422), ("file.pdf", pdf_bytes(""), 422),
    ("file.pdf", b"%PDF-" + b"x" * MAX_UPLOAD_BYTES, 413),
    ("file.pdf", b"x" * (MAX_UPLOAD_BYTES + 70000), 413),
])
def test_invalid_uploads_are_rejected(setup, filename, content, status):
    settings, _, _, client = setup
    response = client.post("/api/upload", files={"file": (filename, content, "application/pdf")})
    assert response.status_code == status
    assert not list(settings.data_dir.glob("*.pdf"))


def test_one_file_encrypted_and_page_limits(setup):
    _, _, _, client = setup
    response = client.post("/api/upload", files=[("file", ("one.pdf", pdf_bytes())), ("file", ("two.pdf", pdf_bytes()))])
    assert response.status_code == 422
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(pdf_bytes())))
    writer.encrypt("password")
    buffer = io.BytesIO()
    writer.write(buffer)
    assert client.post("/api/upload", files={"file": ("encrypted.pdf", buffer.getvalue())}).json()["error_code"] == "encrypted_pdf"
    writer = PdfWriter()
    for _ in range(101):
        writer.add_blank_page(width=100, height=100)
    buffer = io.BytesIO()
    writer.write(buffer)
    assert client.post("/api/upload", files={"file": ("long.pdf", buffer.getvalue())}).json()["error_code"] == "too_many_pages"


def test_failed_index_write_rolls_back_only_new_upload(setup, monkeypatch):
    settings, rag, _, client = setup
    first = client.post("/api/upload", files={"file": ("first.pdf", pdf_bytes())})
    store = rag._store(create=False)
    before = store.get(include=[])["ids"]
    original_upsert = store._collection.upsert

    def partial_failure(**kwargs):
        original_upsert(**kwargs)
        raise RuntimeError("private-details")

    monkeypatch.setattr(rag, "_store", lambda **kwargs: store)
    monkeypatch.setattr(store._collection, "upsert", partial_failure)
    response = client.post("/api/upload", files={"file": ("new.pdf", pdf_bytes("Feuer"))})
    assert response.status_code == 503 and "private-details" not in response.text
    assert store.get(include=[])["ids"] == before
    assert [p.name for p in settings.data_dir.glob("*.pdf")] == [first.json()["filename"]]


def test_embedding_failure_never_publishes_pdf(setup, monkeypatch):
    settings, rag, _, client = setup
    monkeypatch.setattr(rag.embeddings, "embed_documents", Mock(side_effect=RuntimeError("private-details")))
    response = client.post("/api/upload", files={"file": ("new.pdf", pdf_bytes())})
    assert response.status_code == 503
    assert not list(settings.data_dir.glob("*.pdf"))
    assert rag._store(create=False).get(include=[])["ids"] == []


def test_metrics_read_only_missing_valid_and_corrupt(setup):
    settings, _, agent, client = setup
    assert client.get("/api/metrics").status_code == 404
    report = EvaluationReport(total_cases=5, selected_cases=1, config={}, corpus_sha256={}, dataset_sha256="test", thresholds={})
    save_report(report, settings.reports_dir)
    response = client.get("/api/metrics")
    assert response.status_code == 200 and response.json()["status"] == "incomplete"
    agent.run.assert_not_called()
    (settings.reports_dir / "latest.json").write_text("invalid")
    assert client.get("/api/metrics").status_code == 503


def test_upload_then_real_agent_workflow(setup):
    settings, rag, _, _ = setup
    generator = Mock(generate=Mock(return_value=GroundedAnswer(supported=True, statements=[{"text": "Wasser ist versichert.", "source_ids": ["S1"]}])))
    agent = InsuranceAgent(settings, retriever=rag, scorer=Mock(score=Mock(return_value=0.1)), generator=generator)
    client = TestClient(create_app(settings, rag=rag, agent=agent))
    assert client.post("/api/upload", files={"file": ("terms.pdf", pdf_bytes())}).status_code == 201
    result = client.post("/api/chat", json=REQUEST)
    assert result.status_code == 200 and result.json()["final_answer"].endswith("[S1]")
    assert result.json()["sources"][0]["page"] == 1


def test_failed_rollback_blocks_chat_and_further_uploads(setup, monkeypatch):
    from app import main
    from app.uploads import UploadError
    _, _, agent, client = setup
    monkeypatch.setattr(main, "upload_pdf", Mock(side_effect=UploadError(503, "upload_rollback_failed", "Reparatur nötig.")))
    response = client.post("/api/upload", files={"file": ("terms.pdf", pdf_bytes())})
    assert response.json()["error_code"] == "upload_rollback_failed"
    assert client.post("/api/chat", json=REQUEST).json()["error_code"] == "index_unavailable"
    assert client.post("/api/upload", files={"file": ("terms.pdf", pdf_bytes())}).json()["error_code"] == "index_unavailable"
    agent.run.assert_not_called()
