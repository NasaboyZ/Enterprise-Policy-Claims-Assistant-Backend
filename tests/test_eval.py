import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app.agent import AgentResult
from app.config import Settings
from app.eval import RagasJudge, run_evaluation
from app.provider_errors import ProviderError
from app.rag_engine import RetrievedChunk
from app.reports import CaseResult, EvaluationReport, finalize_report, save_report


def report_with_scores(faithfulness=0.8, relevance=0.7):
    return EvaluationReport(total_cases=5, selected_cases=5, config={}, corpus_sha256={}, dataset_sha256="test",
                            thresholds={"faithfulness": 0.8, "answer_relevancy": 0.7},
                            cases=[CaseResult(id=str(i), question="Frage", reference="Referenz", status="answered",
                                              faithfulness=faithfulness, answer_relevancy=relevance) for i in range(5)])


@pytest.mark.parametrize("faith,relevance,code,status", [(0.8, 0.7, 0, "passed"), (0.79, 0.7, 1, "failed"), (0.8, 0.69, 1, "failed")])
def test_quality_gate_boundaries(faith, relevance, code, status):
    report = report_with_scores(faith, relevance)
    assert finalize_report(report) == code and report.status == status


def test_partial_nonfinite_and_unanswered_never_pass():
    report = report_with_scores()
    report.cases.pop()
    assert finalize_report(report) == 2
    report = report_with_scores()
    report.cases[0].faithfulness = float("nan")
    assert finalize_report(report) == 2 and report.scores["faithfulness"] is None
    report = report_with_scores()
    report.cases[0].status = "manual_review"
    assert finalize_report(report) == 2


def test_atomic_report_roundtrip_and_no_temporary_files(tmp_path):
    report = report_with_scores()
    finalize_report(report)
    save_report(report, tmp_path)
    loaded = EvaluationReport.model_validate_json((tmp_path / "latest.json").read_text())
    assert loaded == report
    assert len(list(tmp_path.glob("*.json"))) == 2 and not list(tmp_path.glob(".report-*"))


@pytest.mark.parametrize("mode,expected", [("ok", 0), ("partial", 2), ("quota", 2), ("nan", 2), ("abstain", 2)])
def test_evaluation_captures_context_and_saves_failures(tmp_path, mode, expected):
    settings = replace(Settings(), data_dir=tmp_path, reports_dir=tmp_path / "reports")
    chunk = RetrievedChunk("S1", "id", "terms.pdf", 1, "Selbstbehalt CHF 200.", 0.9)
    recorder = SimpleNamespace(context=[])

    def answer(request):
        recorder.context = [chunk]
        return AgentResult(status="insufficient_context" if mode == "abstain" else "answered", ml_score=0.1,
                           final_answer="CHF 200 [S1]", sources=[])

    judge = SimpleNamespace(score=AsyncMock(return_value={"faithfulness": 0.9, "answer_relevancy": 0.8}))
    if mode == "quota":
        judge.score.side_effect = ProviderError("aion_quota_exceeded", "Safe quota message")
    if mode == "nan":
        judge.score.return_value["faithfulness"] = float("nan")
    report, code = asyncio.run(run_evaluation(settings, agent=Mock(run=answer), recorder=recorder, judge=judge,
                                              limit=1 if mode == "partial" else None))
    assert code == expected
    assert report.cases[0].contexts[0]["text"] == chunk.text
    persisted = json.loads((settings.reports_dir / "latest.json").read_text())
    assert persisted["status"] == report.status and "NaN" not in json.dumps(persisted)
    if mode in {"quota", "nan", "abstain"}:
        assert len(report.cases) == 1
    if mode == "abstain":
        judge.score.assert_not_called()


def test_real_ragas_metrics_use_only_aion_and_local_embeddings(embeddings):
    from langchain_openai import ChatOpenAI
    calls = []
    outputs = [
        {"statements": ["Der Selbstbehalt beträgt CHF 200."]},
        {"statements": [{"statement": "Der Selbstbehalt beträgt CHF 200.", "reason": "Steht im Kontext.", "verdict": 1}]},
        {"question": "Wie hoch ist der Selbstbehalt?", "noncommittal": 0},
    ]

    def handler(request):
        assert request.url.host == "api.aionlabs.ai"
        assert request.headers["authorization"] == "Bearer offline-key"
        payload = json.loads(request.content)
        assert "response_format" not in payload
        calls.append(payload)
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "aion-labs/aion-2.0", "choices": [{"index": 0, "message": {"role": "assistant",
            "content": json.dumps(outputs[len(calls)-1])}, "finish_reason": "stop"}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = ChatOpenAI(model="aion-labs/aion-2.0", api_key="offline-key", base_url="https://api.aionlabs.ai/v1",
                                max_retries=0, use_responses_api=False, http_async_client=transport)
            return await RagasJudge(client, embeddings).score("Wie hoch ist der Selbstbehalt?", "CHF 200.", ["Selbstbehalt CHF 200."], "CHF 200")

    scores = asyncio.run(run())
    assert scores["faithfulness"] == 1 and scores["answer_relevancy"] == pytest.approx(1)
    assert len(calls) == 3


def test_ragas_does_not_retry_invalid_json(embeddings):
    from langchain_openai import ChatOpenAI
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "aion-labs/aion-2.0", "choices": [{"index": 0, "message": {"role": "assistant",
            "content": "not json"}, "finish_reason": "stop"}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = ChatOpenAI(model="aion-labs/aion-2.0", api_key="offline-key", base_url="https://api.aionlabs.ai/v1",
                                max_retries=0, use_responses_api=False, http_async_client=transport)
            with pytest.raises(Exception):
                await RagasJudge(client, embeddings).score("Frage", "Antwort", ["Kontext"], "Referenz")

    asyncio.run(run())
    assert len(calls) == 1
