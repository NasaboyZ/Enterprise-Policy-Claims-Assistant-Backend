"""Validated, atomic evaluation artifacts shared by the CLI and read-only API."""

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CaseResult(ReportModel):
    id: str
    question: str
    reference: str
    status: str
    answer: str = ""
    contexts: list[dict] = Field(default_factory=list)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    error_code: str | None = None


class EvaluationReport(ReportModel):
    schema_version: int = 1
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["incomplete", "passed", "failed"] = "incomplete"
    total_cases: int
    selected_cases: int
    config: dict
    corpus_sha256: dict[str, str]
    dataset_sha256: str
    thresholds: dict[str, float]
    scores: dict[str, float | None] = Field(default_factory=lambda: {"faithfulness": None, "answer_relevancy": None})
    cases: list[CaseResult] = Field(default_factory=list)
    error_code: str | None = None
    notes: list[str] = Field(default_factory=lambda: [
        "Synthetische Demo; keine Produktionsvalidierung.",
        "Aion bewertet Antworten desselben Modells; keine unabhängige Bewertung.",
        "AnswerRelevancy verwendet eine Vergleichsfrage (strictness=1).",
        "Referenzantworten dienen der Nachprüfung; diese Metriken prüfen nicht vollständig die fachliche Korrektheit.",
    ])
    failure_type: str | None = None


def finalize_report(report: EvaluationReport) -> int:
    for metric in ("faithfulness", "answer_relevancy"):
        values = [getattr(case, metric) for case in report.cases]
        report.scores[metric] = sum(values) / len(values) if values and all(
            value is not None and math.isfinite(value) for value in values) else None
    if (report.error_code or len(report.cases) != report.total_cases
            or any(case.status != "answered" or case.error_code for case in report.cases)
            or any(value is None for value in report.scores.values())):
        report.status = "incomplete"
        return 2
    passed = all(report.scores[name] >= threshold for name, threshold in report.thresholds.items())
    report.status = "passed" if passed else "failed"
    return 0 if passed else 1


def save_report(report: EvaluationReport, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for destination in (directory / f"{report.run_id}.json", directory / "latest.json"):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".report-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
