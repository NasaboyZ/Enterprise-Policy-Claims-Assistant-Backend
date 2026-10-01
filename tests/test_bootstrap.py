from dataclasses import replace
from unittest.mock import Mock

from app import bootstrap
from app.config import Settings


def test_bootstrap_trains_only_missing_model_and_reconciles_index(tmp_path, monkeypatch):
    settings = replace(Settings(), model_path=tmp_path / "model.pkl", data_dir=tmp_path)

    def train(csv, model):
        assert csv == tmp_path / "claims.csv"
        model.write_bytes(b"test-artifact")

    trainer = Mock(side_effect=train)
    rag = Mock(index=Mock(return_value={"chunks": 9, "added": 0, "removed": 0, "unchanged": 9}))
    monkeypatch.setattr(bootstrap, "train", trainer)
    monkeypatch.setattr(bootstrap, "RagEngine", Mock(return_value=rag))
    assert bootstrap.prepare(settings)["chunks"] == 9
    assert bootstrap.prepare(settings)["unchanged"] == 9
    trainer.assert_called_once()
    assert rag.index.call_count == 2
