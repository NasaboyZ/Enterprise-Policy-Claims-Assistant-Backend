"""Local embeddings and the RAG path must work without cloud credentials."""

import json
from dataclasses import replace
from unittest.mock import Mock

import chromadb
from chromadb.config import Settings as ChromaSettings
import fastembed
import numpy as np
import pytest

from app import rag_engine
from app.config import Settings
from app.local_embeddings import LocalEmbeddings
from app.provider_errors import ProviderError
from app.rag_engine import RagEngine, RagError


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "AION_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "")


def test_lazy_cpu_model_cache_and_passage_query_inputs(tmp_path, monkeypatch):
    model = Mock()
    model.passage_embed.return_value = iter([np.array([1, 2]), np.array([3, 4])])
    model.query_embed.return_value = iter([np.array([5, 6])])
    factory = Mock(return_value=model)
    monkeypatch.setattr(fastembed, "TextEmbedding", factory)
    settings = Settings()
    adapter = LocalEmbeddings(settings.embedding_model, tmp_path)
    assert adapter.embed_documents([]) == []
    factory.assert_not_called()
    assert adapter.embed_documents(["Wasser", "Feuer"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert adapter.embed_query("Selbstbehalt?") == [5.0, 6.0]
    factory.assert_called_once_with(model_name=settings.embedding_model, cache_dir=str(tmp_path),
                                    threads=2, providers=["CPUExecutionProvider"])
    model.passage_embed.assert_called_once_with(["Wasser", "Feuer"], batch_size=16)
    model.query_embed.assert_called_once_with(["Selbstbehalt?"])


@pytest.mark.parametrize("vectors", [[], [np.array([])], [np.array([np.nan])],
                                     [np.array([np.inf])], [np.zeros(2)], [np.ones((2, 2))],
                                     [np.ones(2), np.ones(2)]])
def test_invalid_local_vectors_are_rejected(tmp_path, monkeypatch, vectors):
    model = Mock()
    model.passage_embed.return_value = iter(vectors)
    monkeypatch.setattr(fastembed, "TextEmbedding", Mock(return_value=model))
    with pytest.raises(ProviderError) as error:
        LocalEmbeddings(Settings().embedding_model, tmp_path).embed_documents(["Dokument"])
    assert error.value.code == "local_embeddings_failed"


def test_cli_initial_download_failure_is_actionable(tmp_path, monkeypatch, capsys):
    factory = Mock(side_effect=RuntimeError("private-upstream-details"))
    monkeypatch.setattr(fastembed, "TextEmbedding", factory)
    engine = RagEngine(replace(Settings(), chroma_dir=tmp_path / "chroma", embedding_cache_dir=tmp_path / "cache"))
    monkeypatch.setattr(rag_engine, "RagEngine", lambda: engine)
    monkeypatch.setattr("sys.argv", ["rag_engine", "index"])
    with pytest.raises(SystemExit) as error:
        rag_engine.main()
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "Modelldownload" in message and "Internet" in message
    assert "private-upstream" not in message and "Traceback" not in message and "GOOGLE_API_KEY" not in message
    assert engine._store(create=False).get(include=[])["ids"] == []


def test_default_cli_indexes_and_searches_without_keys(tmp_path, monkeypatch, capsys, embeddings):
    # Actual PDFs, Chroma, local adapter and CLI; only neural inference is doubled.
    model = Mock()
    model.passage_embed.side_effect = lambda texts, **kwargs: iter(embeddings.embed_documents(texts))
    model.query_embed.side_effect = lambda texts: iter([embeddings.embed_query(t) for t in texts])
    factory = Mock(return_value=model)
    monkeypatch.setattr(fastembed, "TextEmbedding", factory)
    settings = replace(Settings(), chroma_dir=tmp_path / "chroma", embedding_cache_dir=tmp_path / "cache")
    engine = RagEngine(settings)
    monkeypatch.setattr(rag_engine, "RagEngine", lambda: engine)
    monkeypatch.setattr("sys.argv", ["rag_engine", "index"])
    rag_engine.main()
    assert json.loads(capsys.readouterr().out)["added"] > 0
    calls = model.passage_embed.call_count
    assert engine.index()["added"] == 0
    assert model.passage_embed.call_count == calls
    # Re-open with a fresh adapter to exercise persistent-index compatibility.
    reopened = RagEngine(settings)
    monkeypatch.setattr(rag_engine, "RagEngine", lambda: reopened)
    monkeypatch.setattr("sys.argv", ["rag_engine", "search", "Selbstbehalt Leitungswasser"])
    rag_engine.main()
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 3 and results[0]["source_id"] == "S1"
    assert isinstance(reopened.embeddings, LocalEmbeddings)


def test_gemini_index_is_preserved_and_rejected_for_local_vectors(tmp_path, embeddings):
    settings = replace(Settings(), chroma_dir=tmp_path / "chroma")
    client = chromadb.PersistentClient(path=str(settings.chroma_dir),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    old_name = "insurance_gemini_embedding2_v1"
    old = client.create_collection(old_name, metadata={
        "embedding_model": "gemini-embedding-2", "embedding_provider": "google",
        "embedding_format": "retrieval-prefix-v1", "schema_version": 2,
    }, configuration={"hnsw": {"space": "cosine"}})
    old.add(ids=["old-gemini"], embeddings=[[1.0, 0.0]], documents=["Original Gemini document"])
    before = old.get(include=["documents", "embeddings"])
    engine = RagEngine(settings, embeddings)
    assert engine.index()["added"] > 0
    new = client.get_collection(settings.collection_name)
    assert new.metadata["embedding_provider"] == "local" and new.metadata["schema_version"] == 3
    after = old.get(include=["documents", "embeddings"])
    assert before["documents"] == after["documents"] and before["ids"] == after["ids"]
    np.testing.assert_array_equal(before["embeddings"], after["embeddings"])
    with pytest.raises(RagError, match="passt nicht"):
        RagEngine(replace(settings, collection_name=old_name), embeddings).index()
