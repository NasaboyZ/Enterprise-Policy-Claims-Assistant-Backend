from dataclasses import replace

import chromadb
from chromadb.config import Settings as ChromaSettings
import pytest
from reportlab.pdfgen.canvas import Canvas

from app.config import Settings
from app.rag_engine import RagEngine, RagError


def write_pdf(path, pages):
    canvas = Canvas(str(path), invariant=1)
    for page in pages:
        canvas.drawString(30, 800, page)
        canvas.showPage()
    canvas.save()


@pytest.fixture
def engine(tmp_path, embeddings):
    data = tmp_path / "data"
    data.mkdir()
    settings = replace(Settings(), data_dir=data, chroma_dir=tmp_path / "chroma", embedding_model="test-hash")
    return RagEngine(settings, embeddings)


def test_chunking_pages_and_cosine_top_three(engine):
    write_pdf(engine.settings.data_dir / "terms.pdf", ["Wasser " * 220, "Feuer " * 220])
    chunks = engine.load_chunks()
    assert len(chunks) >= 4
    assert all(len(c.page_content) <= 600 for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    report = engine.index()
    assert report["chunks"] == len(chunks)
    results = engine.retrieve("Wasser")
    assert len(results) == 3
    assert results[0].page == 1 and results[0].filename == "terms.pdf"
    assert results[0].cosine_similarity == pytest.approx(1, abs=1e-5)
    assert results[0].source_id == "S1"
    collection = chromadb.PersistentClient(path=str(engine.settings.chroma_dir),
                                           settings=ChromaSettings(anonymized_telemetry=False)).get_collection(engine.settings.collection_name)
    assert collection.configuration["hnsw"]["space"] == "cosine"
    reopened = RagEngine(engine.settings, engine.embeddings)
    assert reopened.retrieve("Wasser") == results


def test_index_idempotence_updates_and_removals(engine):
    first = engine.settings.data_dir / "one.pdf"
    second = engine.settings.data_dir / "two.pdf"
    write_pdf(first, ["Wasser ist versichert."])
    write_pdf(second, ["Feuer ist versichert."])
    assert engine.index()["added"] == 2
    calls = engine.embeddings.document_calls
    assert engine.index() == {"chunks": 2, "added": 0, "removed": 0, "unchanged": 2}
    assert engine.embeddings.document_calls == calls
    write_pdf(first, ["Wasser hat einen Selbstbehalt von CHF 200."])
    second.unlink()
    assert engine.index() == {"chunks": 1, "added": 1, "removed": 2, "unchanged": 0}
    results = engine.retrieve("Wasser")
    assert len(results) == 1 and "200" in results[0].text
    first.unlink()
    assert engine.index()["removed"] == 1
    with pytest.raises(RagError, match="leer"):
        engine.retrieve("Wasser")


def test_bad_pdf_or_embedding_failure_preserves_existing_index(engine, monkeypatch):
    path = engine.settings.data_dir / "one.pdf"
    write_pdf(path, ["Original Wasser"])
    engine.index()
    original = engine.retrieve("Wasser")[0].chunk_id
    bad = engine.settings.data_dir / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(RagError, match="gelesen"):
        engine.index()
    assert engine.retrieve("Wasser")[0].chunk_id == original
    bad.unlink()
    write_pdf(path, ["Changed Wasser"])
    def fail(_):
        raise RuntimeError("Embedding service unavailable")
    monkeypatch.setattr(engine.embeddings, "embed_documents", fail)
    with pytest.raises(RuntimeError):
        engine.index()
    assert engine.retrieve("Wasser")[0].chunk_id == original


def test_missing_empty_and_incompatible_index(engine, monkeypatch):
    with pytest.raises(RagError, match="fehlt"):
        engine.retrieve("Wasser")
    with pytest.raises(RagError, match="Keine PDFs"):
        engine.index()
    write_pdf(engine.settings.data_dir / "one.pdf", ["Wasser"])
    engine.index()
    incompatible = RagEngine(replace(engine.settings, embedding_model="other"), engine.embeddings)
    with pytest.raises(RagError, match="passt nicht"):
        incompatible.retrieve("Wasser")
    with pytest.raises(ValueError):
        engine.retrieve(" ")
