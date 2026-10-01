"""Run Ragas explicitly: python -m app.eval [--limit 1]. Never invoked by the API."""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import asdict

# Set before importing Ragas so local evaluations do not enable usage tracking.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import ValidationError
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import AnswerRelevancy, ResponseRelevancePrompt
from ragas.metrics._faithfulness import Faithfulness, NLIStatementPrompt, StatementGeneratorPrompt
from ragas.run_config import RunConfig

from app.agent import AionAnswerGenerator, InsuranceAgent, create_aion_chat_model
from app.config import ROOT, Settings
from app.provider_errors import ProviderError, translate_aion_error
from app.rag_engine import RagEngine
from app.reports import CaseResult, EvaluationReport, finalize_report, save_report

# Ragas parser exceptions may contain model output; publish only our error codes.
logging.getLogger("ragas").addHandler(logging.NullHandler())
logging.getLogger("ragas").propagate = False


class NoParserRetries:
    async def generate_multiple(self, *args, **kwargs):
        kwargs["retries_left"] = 0
        return await super().generate_multiple(*args, **kwargs)


class StatementPrompt(NoParserRetries, StatementGeneratorPrompt):
    examples = []


class VerdictPrompt(NoParserRetries, NLIStatementPrompt):
    examples = []


class RelevancePrompt(NoParserRetries, ResponseRelevancePrompt):
    examples = []


class RecordingRetriever:
    def __init__(self, rag):
        self.rag = rag
        self.context = []

    def retrieve(self, query):
        self.context = self.rag.retrieve(query)
        return self.context


class RagasJudge:
    def __init__(self, client, embeddings):
        config = RunConfig(timeout=180, max_retries=1, max_workers=1)
        llm = LangchainLLMWrapper(client, run_config=config, bypass_n=True, bypass_temperature=True,
                                 is_finished_parser=lambda result: all(
                                     generation.generation_info.get("finish_reason") == "stop"
                                     for row in result.generations for generation in row))
        self.metrics = [
            Faithfulness(llm=llm, statement_generator_prompt=StatementPrompt(), nli_statements_prompt=VerdictPrompt()),
            AnswerRelevancy(llm=llm, embeddings=LangchainEmbeddingsWrapper(embeddings),
                            strictness=1, question_generation=RelevancePrompt()),
        ]
        for metric in self.metrics:
            metric.init(config)

    async def score(self, question, response, contexts, reference):
        sample = SingleTurnSample(user_input=question, response=response,
                                  retrieved_contexts=contexts, reference=reference)
        scores = {}
        for metric in self.metrics:
            value = float(await metric.single_turn_ascore(sample, timeout=180))
            if not math.isfinite(value):
                raise ProviderError("evaluation_invalid_score", "Evaluation lieferte keinen endlichen Messwert.")
            scores[metric.name] = value
        return scores


async def run_evaluation(settings, *, limit=None, faithfulness=0.80, relevance=0.70,
                         agent=None, recorder=None, judge=None):
    dataset_path = ROOT / "data" / "eval_cases.json"
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    selected = cases[:limit] if limit is not None else cases
    report = EvaluationReport(
        total_cases=len(cases), selected_cases=len(selected),
        config={"provider": "aion", "answer_model": settings.aion_chat_model,
                "judge_model": settings.aion_chat_model, "embedding_model": settings.embedding_model,
                "ragas_version": "0.4.3", "relevancy_strictness": 1, "request_interval_seconds": 5,
                "judge_reasoning_effort": "none", "judge_max_tokens": 2048,
                "fraud_threshold": settings.fraud_threshold,
                "ml_model_sha256": hashlib.sha256(settings.model_path.read_bytes()).hexdigest() if settings.model_path.is_file() else None},
        corpus_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(settings.data_dir.glob("*.pdf"))},
        dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        thresholds={"faithfulness": faithfulness, "answer_relevancy": relevance},
    )
    try:
        if agent is None or judge is None:
            limiter = InMemoryRateLimiter(requests_per_second=0.2, check_every_n_seconds=0.1, max_bucket_size=1)
            client = create_aion_chat_model(settings, rate_limiter=limiter)
            rag = RagEngine(settings)
            recorder = RecordingRetriever(rag)
            generator = AionAnswerGenerator(settings)
            generator.model = client
            agent = InsuranceAgent(settings, retriever=recorder, generator=generator)
            # Short classification tasks do not need the model's default reasoning
            # budget. Keep the answer generator unchanged and share only the limiter.
            judge_client = client.model_copy(update={"reasoning_effort": "none", "extra_body": {"max_tokens": 2048}})
            judge = RagasJudge(judge_client, rag._embeddings())
        for case in selected:
            print(f"Evaluation: {case['id']}", file=sys.stderr, flush=True)
            recorder.context = []
            result = await asyncio.to_thread(agent.run, {
                "query": case["question"],
                "claim": {"customer_age": 40, "claim_amount": 100, "claim_type": "water"},
            })
            item = CaseResult(id=case["id"], question=case["question"], reference=case["reference"],
                              status=result.status, answer=result.final_answer,
                              contexts=[asdict(c) for c in recorder.context], error_code=result.error_code)
            report.cases.append(item)
            if result.status != "answered":
                report.error_code = result.error_code or "evaluation_not_answered"
                break
            scores = await judge.score(case["question"], result.final_answer,
                                       [c.text for c in recorder.context], case["reference"])
            for name in ("faithfulness", "answer_relevancy"):
                value = float(scores[name])
                if not math.isfinite(value):
                    raise ProviderError("evaluation_invalid_score", "Evaluation lieferte keinen endlichen Messwert.")
                setattr(item, name, value)
            save_report(report, settings.reports_dir)
    except Exception as exc:
        error = translate_aion_error(exc)
        report.error_code = error.code if error.code != "aion_request_failed" else "evaluation_failed"
        report.failure_type = type(exc).__name__
        if isinstance(exc, ValidationError):
            report.error_code = "evaluation_invalid_output"
        if report.cases:
            report.cases[-1].error_code = report.error_code
    code = finalize_report(report)
    save_report(report, settings.reports_dir)
    return report, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Anzahl Fälle; Teilberichte bestehen das Quality Gate nicht")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.80)
    parser.add_argument("--relevance-threshold", type=float, default=0.70)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit muss mindestens 1 sein")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (args.faithfulness_threshold, args.relevance_threshold)):
        parser.error("Schwellen müssen zwischen 0 und 1 liegen")
    try:
        report, code = asyncio.run(run_evaluation(Settings.from_env(), limit=args.limit,
                                                faithfulness=args.faithfulness_threshold,
                                                relevance=args.relevance_threshold))
    except Exception:
        parser.exit(2, "Evaluation konnte nicht gestartet oder Bericht nicht gespeichert werden. Konfiguration und Dateirechte prüfen.\n")
    print(json.dumps({"status": report.status, "scores": report.scores, "error_code": report.error_code,
                      "failure_type": report.failure_type,
                      "run_id": report.run_id}, ensure_ascii=False, allow_nan=False, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
