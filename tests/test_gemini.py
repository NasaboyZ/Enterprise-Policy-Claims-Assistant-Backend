"""Exercise the real Google/LangChain clients using an offline HTTP transport."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import chromadb
from chromadb.config import Settings as ChromaSettings
from google.genai.errors import ClientError
import httpx
import pytest

from app import agent as agent_module, config, gemini_embeddings, rag_engine
from app.agent import GeminiAnswerGenerator, InsuranceAgent
from app.config import Settings, require_google_key
from app.gemini_embeddings import GeminiEmbeddings
from app.provider_errors import ProviderError, translate_provider_error


def test_config_uses_google_names_and_dotenv_without_rewriting(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    original = "GOOGLE_API_KEY=fake-file-key\nGEMINI_CHAT_MODEL=gemini-file-model\nFRAUD_THRESHOLD=0.7\n"
    path.write_text(original)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "ignored")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "ignored")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("FRAUD_THRESHOLD", raising=False)
    settings = Settings.from_env()
    assert require_google_key() == "fake-file-key"
    assert settings.chat_model == "gemini-file-model" and settings.fraud_threshold == 0.7
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-shell-model")
    assert Settings.from_env().chat_model == "gemini-shell-model"
    assert path.read_text() == original


def test_missing_google_key_does_not_fall_back_to_other_keys(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-use")
    monkeypatch.setenv("GEMINI_API_KEY", "do-not-use-either")
    with pytest.raises(ProviderError, match="GOOGLE_API_KEY"):
        GeminiEmbeddings()
    with pytest.raises(ProviderError, match="GOOGLE_API_KEY"):
        GeminiAnswerGenerator(Settings()).generate("test", [])


def install_embedding_transport(monkeypatch, handler):
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")
    original_client = gemini_embeddings.genai.Client
    def client(**kwargs):
        kwargs["http_options"].client_args = {"transport": httpx.MockTransport(handler)}
        return original_client(**kwargs)
    monkeypatch.setattr(gemini_embeddings.genai, "Client", client)


def test_embedding2_one_vector_per_document_and_retrieval_prefixes(monkeypatch):
    requests = []
    def handler(request):
        assert request.url.host == "generativelanguage.googleapis.com"
        assert "gemini-embedding-2:embedContent" in request.url.path
        body = json.loads(request.content)
        assert "taskType" not in body
        assert request.headers["x-goog-api-key"] == "offline-test-key"
        requests.append(body["content"]["parts"][0]["text"])
        return httpx.Response(200, json={"embedding": {"values": [1.0, float(len(requests))]}})
    install_embedding_transport(monkeypatch, handler)
    embeddings = GeminiEmbeddings()
    try:
        assert embeddings.embed_documents(["First", "Second"]) == [[1.0, 1.0], [1.0, 2.0]]
        assert embeddings.embed_query("Question") == [1.0, 3.0]
        assert requests == ["title: none | text: First", "title: none | text: Second",
                            "task: search result | query: Question"]
    finally:
        embeddings.client.close()


def test_real_chat_client_parses_structured_answer_offline(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")
    captured = []
    def handler(request):
        assert request.url.host == "generativelanguage.googleapis.com"
        assert "gemini-2.5-flash-lite:generateContent" in request.url.path
        body = json.loads(request.content)
        captured.append(body)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        result = {"supported": True, "statements": [{"text": "CHF 200", "source_ids": ["S1"]}]}
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [
            {"text": json.dumps(result)}]}, "finishReason": "STOP"}]})
    original_model = agent_module.ChatGoogleGenerativeAI
    def model(**kwargs):
        assert kwargs["vertexai"] is False and kwargs["max_retries"] == 0
        return original_model(**kwargs, client_args={"transport": httpx.MockTransport(handler)})
    monkeypatch.setattr(agent_module, "ChatGoogleGenerativeAI", model)
    result = GeminiAnswerGenerator(Settings()).generate("Selbstbehalt?", [])
    assert result.supported and result.statements[0].source_ids == ["S1"]
    assert len(captured) == 1


@pytest.mark.parametrize("http_code,status,expected", [
    (400, "API_KEY_INVALID", "google_auth_failed"),
    (401, "UNAUTHENTICATED", "google_auth_failed"),
    (403, "PERMISSION_DENIED", "google_access_denied"),
    (429, "RESOURCE_EXHAUSTED", "google_quota_exceeded"),
    (404, "NOT_FOUND", "google_model_unavailable"),
    (400, "INVALID_ARGUMENT", "google_request_rejected"),
    (503, "UNAVAILABLE", "google_unavailable"),
])
def test_provider_errors_are_safe_and_survive_wrapping(http_code, status, expected):
    original = ClientError(http_code, {"error": {"code": http_code, "status": status,
                                                "message": "sensitive-test-key"}})
    wrapped = RuntimeError("do-not-print-this")
    wrapped.__cause__ = original
    translated = translate_provider_error(wrapped)
    assert translated.code == expected
    assert "sensitive" not in str(translated) and "do-not-print" not in str(translated)


def test_index_cli_shows_quota_message_without_traceback_or_secret(tmp_path, monkeypatch, capsys):
    def handler(request):
        return httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                   "message": "sensitive-test-key"}})
    install_embedding_transport(monkeypatch, handler)
    settings = replace(Settings(), chroma_dir=tmp_path / "chroma")
    embeddings = GeminiEmbeddings()
    monkeypatch.setattr(rag_engine, "RagEngine", lambda: original_engine)
    # Use the actual example PDFs, actual Chroma and actual SDK against MockTransport.
    from app.rag_engine import RagEngine
    # Class was replaced above; get it through the original bound class stored below.
    original_engine = _RAG_CLASS(settings, embeddings)
    monkeypatch.setattr("sys.argv", ["rag_engine", "index"])
    try:
        with pytest.raises(SystemExit) as exit_info:
            rag_engine.main()
        assert exit_info.value.code == 1
        output = capsys.readouterr().err
        assert "Kontingent" in output and "Traceback" not in output and "sensitive" not in output
        assert original_engine._store(create=False).get(include=[])["ids"] == []
    finally:
        embeddings.client.close()


_RAG_CLASS = rag_engine.RagEngine


def test_old_openai_collection_is_preserved(tmp_path, embeddings):
    settings = replace(Settings(), chroma_dir=tmp_path / "chroma")
    client = chromadb.PersistentClient(path=str(settings.chroma_dir),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    old = client.create_collection("insurance_documents_v1", metadata={"embedding_model": "text-embedding-3-small"})
    old.add(ids=["old"], embeddings=[[1.0, 0.0]], documents=["Old document"])
    rag = _RAG_CLASS(settings, embeddings)
    assert rag.index()["added"] > 0
    assert old.get(include=["documents"])["documents"] == ["Old document"]
    assert client.get_collection(settings.collection_name).metadata["embedding_provider"] == "google"
    assert rag.retrieve("Selbstbehalt")[0].chunk_id != "old"


@pytest.mark.parametrize("stage", ["retrieval", "generation"])
def test_agent_preserves_actionable_provider_error(stage):
    scorer = Mock(score=Mock(return_value=0.1))
    retriever = Mock()
    generator = Mock()
    failure = ProviderError("google_quota_exceeded", "Gemini-Kontingent erreicht.")
    if stage == "retrieval":
        retriever.retrieve.side_effect = failure
    else:
        retriever.retrieve.return_value = [SimpleNamespace(text="Source")]
        generator.generate.side_effect = failure
    result = InsuranceAgent(Settings(), scorer=scorer, retriever=retriever, generator=generator).run({
        "query": "Frage", "claim": {"customer_age": 40, "claim_amount": 500, "claim_type": "water"},
    })
    assert result.status == "error" and result.error_code == "google_quota_exceeded"
    assert "Kontingent" in result.final_answer and result.sources == []
