from dataclasses import replace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.agent import AgentRequest, GroundedAnswer, InsuranceAgent
from app.config import Settings
from app.rag_engine import RetrievedChunk

REQUEST = {"query": "Wie hoch ist der Selbstbehalt?", "claim": {
    "customer_age": 40, "claim_amount": 500, "claim_type": "water",
}}
CONTEXT = [RetrievedChunk("S1", "chunk1", "avb.pdf", 2, "Selbstbehalt CHF 200.", 0.9)]


def dependencies(score=0.1):
    scorer, retriever, generator = Mock(), Mock(), Mock()
    scorer.score.return_value = score
    retriever.retrieve.return_value = CONTEXT
    generator.generate.return_value = GroundedAnswer(supported=True, statements=[
        {"text": "Der Selbstbehalt beträgt CHF 200.", "source_ids": ["S1"]},
    ])
    return scorer, retriever, generator


def agent(deps, **settings):
    return InsuranceAgent(replace(Settings(), **settings), scorer=deps[0], retriever=deps[1], generator=deps[2])


@pytest.mark.parametrize("score", [0.5, 0.9, 1])
def test_high_risk_and_threshold_do_not_call_external_services(score):
    deps = dependencies(score)
    result = agent(deps).run(REQUEST)
    assert result.status == "manual_review" and result.ml_score == score
    assert result.sources == []
    deps[1].retrieve.assert_not_called()
    deps[2].generate.assert_not_called()


def test_low_risk_returns_only_cited_sources_and_validates_input():
    deps = dependencies(0.499)
    result = agent(deps).run(REQUEST)
    assert result.status == "answered" and result.error_code is None
    assert result.final_answer.endswith("[S1]")
    assert result.sources[0]["page"] == 2 and result.sources[0]["filename"] == "avb.pdf"
    with pytest.raises(ValidationError):
        agent(deps).run({**REQUEST, "query": " "})
    with pytest.raises(ValidationError):
        AgentRequest.model_validate({**REQUEST, "claim": {**REQUEST["claim"], "claim_amount": -1}})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_risk_fails_closed(score):
    deps = dependencies(score)
    result = agent(deps).run(REQUEST)
    assert result.status == "error" and result.ml_score is None
    deps[1].retrieve.assert_not_called()
    deps[2].generate.assert_not_called()


@pytest.mark.parametrize("failure,code", [(FileNotFoundError(), "model_missing"), (ValueError(), "risk_check_failed")])
def test_risk_errors_stop_workflow(failure, code):
    deps = dependencies()
    deps[0].score.side_effect = failure
    result = agent(deps).run(REQUEST)
    assert result.error_code == code and result.ml_score is None
    deps[1].retrieve.assert_not_called()


def test_empty_context_skips_generation():
    deps = dependencies()
    deps[1].retrieve.return_value = []
    result = agent(deps).run(REQUEST)
    assert result.status == "insufficient_context" and not result.sources
    deps[2].generate.assert_not_called()


@pytest.mark.parametrize("statements", [[], [{"text": "Unbelegt", "source_ids": []}],
                                      [{"text": "Erfunden", "source_ids": ["S99"]}],
                                      [{"text": "Behauptung [S99]", "source_ids": ["S1"]}],
                                      [{"text": " ", "source_ids": ["S1"]}]])
def test_invalid_citations_suppress_answer(statements):
    deps = dependencies()
    deps[2].generate.return_value = GroundedAnswer(supported=True, statements=statements)
    result = agent(deps).run(REQUEST)
    assert result.status == "error" and result.error_code == "invalid_citations"
    assert result.sources == [] and "Erfunden" not in result.final_answer


def test_generator_can_abstain():
    deps = dependencies()
    deps[2].generate.return_value = GroundedAnswer(supported=False, statements=[])
    assert agent(deps).run(REQUEST).status == "insufficient_context"


@pytest.mark.parametrize("position,method,code", [(1, "retrieve", "retrieval_failed"), (2, "generate", "generation_failed")])
def test_service_failure_is_not_success(position, method, code):
    deps = dependencies()
    getattr(deps[position], method).side_effect = RuntimeError("secret upstream details")
    result = agent(deps).run(REQUEST)
    assert result.status == "error" and result.error_code == code
    assert "secret" not in result.model_dump_json()


def test_threshold_is_configurable():
    deps = dependencies(0.6)
    assert agent(deps, fraud_threshold=0.7).run(REQUEST).status == "answered"
    with pytest.raises(ValueError):
        Settings(fraud_threshold=float("nan"))
