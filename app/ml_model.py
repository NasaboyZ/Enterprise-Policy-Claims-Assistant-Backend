"""Train and persist the full preprocessing/classifier pipeline."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import ROOT

FEATURES = ["customer_age", "claim_amount", "claim_type"]
TARGET = "is_fraud"
SEED = 42


def load_claims(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Fehlende CSV-Spalten: {', '.join(sorted(missing))}")
    for name in FEATURES[:2]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if frame[name].notna().sum() == 0 or np.isinf(frame[name]).any():
            raise ValueError(f"{name} muss endliche Werte enthalten.")
    if (frame.customer_age.dropna() < 18).any() or (frame.customer_age.dropna() > 100).any():
        raise ValueError("customer_age muss zwischen 18 und 100 liegen.")
    if (frame.claim_amount.dropna() < 0).any():
        raise ValueError("claim_amount darf nicht negativ sein.")
    frame["claim_type"] = frame.claim_type.str.strip().replace("", np.nan).astype(object)
    frame["claim_type"] = frame.claim_type.where(frame.claim_type.notna(), np.nan)
    if not frame[TARGET].isin([0, 1]).all() or set(frame[TARGET]) != {0, 1}:
        raise ValueError("is_fraud muss beide Klassen 0 und 1 ohne fehlende Labels enthalten.")
    if len(frame) < 20 or frame[TARGET].value_counts().min() < 5:
        raise ValueError("Mindestens 20 Zeilen und 5 Beispiele pro Klasse erforderlich.")
    return frame


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), FEATURES[:2]),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), ["claim_type"]),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=1,
        )),
    ])


def train(csv_path: Path = ROOT / "data" / "claims.csv",
          model_path: Path = ROOT / "models" / "fraud_model.pkl") -> dict:
    frame = load_claims(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.2,
        random_state=SEED, stratify=frame[TARGET],
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train).predict(x_test)
    report = {
        "data_kind": "synthetic_demo_not_production_validation",
        "random_seed": SEED, "train_rows": len(x_train), "test_rows": len(x_test),
        "features": FEATURES, "positive_label": 1,
        "fraud_rate": float(frame[TARGET].mean()),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "baseline_accuracy": float(accuracy_score(y_test, baseline)),
        "baseline_f1": float(f1_score(y_test, baseline, zero_division=0)),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    model_path.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "claims.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "fraud_model.pkl")
    args = parser.parse_args()
    print(json.dumps(train(args.csv, args.output), indent=2))
