import shutil
from dataclasses import replace
from unittest.mock import Mock

from app.agent import GroundedAnswer, InsuranceAgent, ModelRiskScorer
from app.config import ROOT, Settings
from app.ml_model import FEATURES, load_claims, train
from app.rag_engine import RagEngine


def test_local_workflow_with_real_model_pdfs_and_chroma(tmp_path, embeddings):
    """Only the two remote services are doubled; all local components are real."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for path in (ROOT / "data").glob("*.pdf"):
        shutil.copy(path, data_dir / path.name)
    settings = replace(Settings(), data_dir=data_dir, chroma_dir=tmp_path / "chroma",
                       model_path=tmp_path / "model.pkl", embedding_model="test-hash")
    train(ROOT / "data" / "claims.csv", settings.model_path)
    scorer = ModelRiskScorer(settings)
    rag = RagEngine(settings, embeddings)
    assert rag.index()["chunks"] >= 3

    def cited_excerpt(query, context):
        assert context and all(c.page >= 1 for c in context)
        return GroundedAnswer(supported=True, statements=[
            {"text": context[0].text, "source_ids": [context[0].source_id]},
        ])

    generator = Mock()
    generator.generate.side_effect = cited_excerpt
    retriever = Mock(wraps=rag)
    agent = InsuranceAgent(settings, scorer=scorer, retriever=retriever, generator=generator)
    frame = load_claims(ROOT / "data" / "claims.csv").dropna()
    import joblib
    model = joblib.load(settings.model_path)
    probabilities = model.predict_proba(frame[FEATURES])[:, list(model.classes_).index(1)]
    for index, expected in [(probabilities.argmin(), "answered"), (probabilities.argmax(), "manual_review")]:
        row = frame.iloc[index]
        result = agent.run({"query": "Welcher Selbstbehalt gilt bei Leitungswasser?", "claim": {
            "customer_age": int(row.customer_age), "claim_amount": float(row.claim_amount),
            "claim_type": row.claim_type,
        }})
        assert result.status == expected
        if expected == "answered":
            assert result.sources and result.final_answer.endswith("[S1]")
        else:
            assert result.sources == []
    assert retriever.retrieve.call_count == generator.generate.call_count == 1
