"""Exercise Groq's actual LangChain/OpenAI client without network or real keys."""

import json
from dataclasses import replace
from unittest.mock import Mock

import httpx
import pytest
from openai import RateLimitError

from app import agent as agent_module, config
from app.agent import GroqAnswerGenerator, GeminiAnswerGenerator, InsuranceAgent, create_answer_generator
from app.config import Settings, require_groq_key
from app.provider_errors import ProviderError, translate_groq_error
from app.rag_engine import RetrievedChunk


REQUEST = {"query": "Wie hoch ist der Selbstbehalt?", "claim": {
    "customer_age": 40, "claim_amount": 500, "claim_type": "water",
}}
CONTEXT = [RetrievedChunk("S1", "chunk1", "avb.pdf", 2, "Selbstbehalt CHF 200.", 0.9)]
ANSWER = {"supported": True, "statements": [{"text": "Der Selbstbehalt beträgt CHF 200.", "source_ids": ["S1"]}]}


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for name in ("CHAT_PROVIDER", "GROQ_CHAT_MODEL", "GROQ_API_KEY", "GEMINI_CHAT_MODEL", "FRAUD_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def install_transport(monkeypatch):
    original_model = agent_module.ChatOpenAI
    clients = []

    def install(handler):
        monkeypatch.setenv("GROQ_API_KEY", "offline-groq-key")
        # A different provider's credentials and endpoint must never be used.
        monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
        monkeypatch.setenv("OPENAI_API_BASE", "https://wrong-provider.invalid/v1")

        def model(**kwargs):
            assert kwargs["max_retries"] == 0 and kwargs["timeout"] == 60
            client = httpx.Client(transport=httpx.MockTransport(handler))
            clients.append(client)
            return original_model(**kwargs, http_client=client)

        monkeypatch.setattr(agent_module, "ChatOpenAI", model)

    yield install
    for client in clients:
        client.close()


def completion(content, finish_reason="stop"):
    return httpx.Response(200, json={
        "id": "offline-completion", "object": "chat.completion", "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    })


def run_agent(settings=None):
    return InsuranceAgent(settings or Settings(chat_provider="groq"), scorer=Mock(score=Mock(return_value=0.1)),
                          retriever=Mock(retrieve=Mock(return_value=CONTEXT))).run(REQUEST)


def test_config_defaults_dotenv_and_shell_precedence(tmp_path, monkeypatch):
    defaults = Settings.from_env()
    assert defaults.chat_provider == "groq"
    assert defaults.groq_chat_model == "openai/gpt-oss-120b"
    path = tmp_path / ".env"
    original = ("GROQ_API_KEY= fake-file-key \nGROQ_CHAT_MODEL=openai/gpt-oss-20b\n"
                "CHAT_PROVIDER=groq\nGEMINI_CHAT_MODEL=gemini-custom\n")
    path.write_text(original)
    settings = Settings.from_env()
    assert require_groq_key() == "fake-file-key"
    assert settings.groq_chat_model == "openai/gpt-oss-20b"
    assert settings.chat_model == "gemini-custom"
    monkeypatch.setenv("CHAT_PROVIDER", "gemini")
    monkeypatch.setenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
    assert Settings.from_env().chat_provider == "gemini"
    assert Settings.from_env().groq_chat_model == "openai/gpt-oss-120b"
    assert path.read_text() == original


@pytest.mark.parametrize("kwargs", [{"chat_provider": "unknown"}, {"groq_chat_model": " "}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_provider_selection_is_explicit():
    assert isinstance(create_answer_generator(Settings(chat_provider="groq")), GroqAnswerGenerator)
    assert isinstance(create_answer_generator(Settings(chat_provider="gemini")), GeminiAnswerGenerator)


@pytest.mark.parametrize("key", [None, "", " ", "YOUR_GROQ_KEY"])
def test_missing_key_never_uses_other_credentials(key, monkeypatch):
    if key is not None:
        monkeypatch.setenv("GROQ_API_KEY", key)
    monkeypatch.setenv("GOOGLE_API_KEY", "wrong-google-key")
    monkeypatch.setenv("AION_API_KEY", "wrong-aion-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")
    client = Mock(side_effect=AssertionError("No client should be created"))
    monkeypatch.setattr(agent_module, "ChatOpenAI", client)
    result = run_agent()
    assert result.error_code == "groq_key_missing"
    assert result.status == "error" and result.sources == []
    client.assert_not_called()


def test_real_client_routes_to_groq_and_returns_cited_answer(install_transport):
    calls = []

    def handler(request):
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer offline-groq-key"
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["max_completion_tokens"] == 4096
        assert body["reasoning_effort"] == "low"
        schema = body["response_format"]["json_schema"]
        assert body["response_format"]["type"] == "json_schema"
        assert schema["strict"] is True
        assert schema["schema"]["additionalProperties"] is False
        assert schema["schema"]["$defs"]["CitedStatement"]["additionalProperties"] is False
        assert body["stream"] is False
        message = json.loads(body["messages"][1]["content"])
        assert message["question"] == REQUEST["query"]
        assert message["sources"][0]["source_id"] == "S1"
        return completion(json.dumps(ANSWER))

    install_transport(handler)
    result = run_agent(replace(Settings(chat_provider="groq"), groq_chat_model="openai/gpt-oss-20b"))
    assert result.status == "answered" and result.error_code is None
    assert result.final_answer.endswith("[S1]") and result.sources[0]["filename"] == "avb.pdf"
    assert len(calls) == 1


@pytest.mark.parametrize("content,finish", [
    ("", "stop"), (None, "stop"), ("not JSON: sensitive-output", "stop"),
    ('{"supported": true', "length"), (json.dumps(ANSWER), "length"),
    (json.dumps(ANSWER), "content_filter"), (json.dumps(ANSWER), None),
    ('{"supported": true}', "stop"),
    ('{"supported": "true", "statements": []}', "stop"),
    ('{"supported": false, "statements": [], "extra": "sensitive-output"}', "stop"),
    ("```json\n" + json.dumps(ANSWER) + "\n```", "stop"),
])
def test_invalid_or_incomplete_answers_are_suppressed(install_transport, content, finish):
    install_transport(lambda request: completion(content, finish))
    result = run_agent()
    assert result.status == "error" and result.error_code == "groq_invalid_response"
    assert result.sources == [] and "sensitive-output" not in result.model_dump_json()


@pytest.mark.parametrize("answer,expected_status,expected_code", [
    ({"supported": False, "statements": []}, "insufficient_context", None),
    ({"supported": True, "statements": [{"text": "Erfunden", "source_ids": ["S99"]}]},
     "error", "invalid_citations"),
])
def test_abstention_and_source_validation(install_transport, answer, expected_status, expected_code):
    install_transport(lambda request: completion(json.dumps(answer)))
    result = run_agent()
    assert result.status == expected_status and result.error_code == expected_code
    assert result.sources == [] and "Erfunden" not in result.final_answer


@pytest.mark.parametrize("status,code", [
    (400, "groq_request_rejected"), (401, "groq_auth_failed"), (403, "groq_access_denied"),
    (404, "groq_model_unavailable"), (429, "groq_quota_exceeded"), (502, "groq_unavailable"),
])
def test_http_failures_are_safe_without_retry_or_fallback(install_transport, monkeypatch, status, code):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "sensitive-upstream-key", "type": "error"}})

    install_transport(handler)
    fallback = Mock(side_effect=AssertionError("No provider fallback"))
    monkeypatch.setattr(agent_module, "ChatGoogleGenerativeAI", fallback)
    result = run_agent()
    assert result.error_code == code and result.status == "error"
    assert result.sources == [] and "sensitive" not in result.model_dump_json()
    assert len(calls) == 1
    fallback.assert_not_called()


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError])
def test_transport_failures_without_retry(install_transport, error_type):
    calls = []

    def handler(request):
        calls.append(request)
        raise error_type("sensitive-network-details", request=request)

    install_transport(handler)
    result = run_agent()
    assert result.status == "error" and result.error_code == "groq_connection_failed"
    assert "sensitive" not in result.model_dump_json() and len(calls) == 1


def test_wrapped_provider_error_and_unknown_errors_are_safe():
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    original = RateLimitError("sensitive-key", response=response, body=None)
    wrapped = RuntimeError("sensitive-wrapper")
    wrapped.__cause__ = original
    translated = translate_groq_error(wrapped)
    assert translated.code == "groq_quota_exceeded" and "sensitive" not in str(translated)
    unknown = translate_groq_error(RuntimeError("sensitive"))
    assert unknown.code == "groq_request_failed" and "sensitive" not in str(unknown)
    own = ProviderError("groq_key_missing", "GROQ_API_KEY fehlt.")
    assert translate_groq_error(own) is own
