from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from app.agent import Claim, ModelRiskScorer
from app.config import Settings
from app.ml_model import FEATURES, SEED, TARGET, load_claims, train
from scripts.generate_demo_data import generate_claims


def test_training_round_trip_reproducibility_and_train_only_imputation(tmp_path):
    csv_path = tmp_path / "claims.csv"
    generate_claims(csv_path)
    model_path = tmp_path / "model.pkl"
    report = train(csv_path, model_path)
    model = joblib.load(model_path)
    frame = load_claims(csv_path)
    x_train, _, _, _ = train_test_split(frame[FEATURES], frame[TARGET], test_size=0.2,
                                       random_state=SEED, stratify=frame[TARGET])
    np.testing.assert_allclose(model.named_steps["preprocess"].named_transformers_["numeric"].statistics_,
                               x_train[FEATURES[:2]].median().to_numpy())
    again = tmp_path / "again.pkl"
    assert train(csv_path, again) == report
    np.testing.assert_array_equal(model.predict(frame[FEATURES]), joblib.load(again).predict(frame[FEATURES]))
    assert report["train_rows"] == 960 and report["test_rows"] == 240
    assert 0 <= report["accuracy"] <= 1 and 0 <= report["f1"] <= 1
    assert model_path.with_suffix(".metrics.json").is_file()
    scorer = ModelRiskScorer(replace(Settings(), model_path=model_path))
    for claim in [Claim(customer_age=None, claim_amount=None, claim_type=None),
                  Claim(customer_age=40, claim_amount=500, claim_type="new_category")]:
        assert 0 <= scorer.score(claim) <= 1


@pytest.mark.parametrize("mutation", ["missing_column", "missing_label", "single_class", "infinite_amount"])
def test_invalid_training_data_rejected(tmp_path, mutation):
    path = tmp_path / "claims.csv"
    generate_claims(path, 100)
    frame = pd.read_csv(path)
    if mutation == "missing_column":
        frame = frame.drop(columns="claim_type")
    elif mutation == "missing_label":
        frame.loc[0, TARGET] = np.nan
    elif mutation == "single_class":
        frame[TARGET] = 0
    else:
        frame.loc[0, "claim_amount"] = np.inf
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        train(path, tmp_path / "model.pkl")
    assert not (tmp_path / "model.pkl").exists()
