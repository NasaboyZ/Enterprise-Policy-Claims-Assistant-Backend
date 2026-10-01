"""Prepare local artifacts before the API starts; never call the answer LLM."""

import json

from app.config import Settings
from app.ml_model import train
from app.rag_engine import RagEngine


def prepare(settings=None):
    settings = settings or Settings.from_env()
    if not settings.model_path.is_file():
        print("Trainiere lokales Demo-ML-Modell ...", flush=True)
        train(settings.data_dir / "claims.csv", settings.model_path)
    print("Bereite lokalen Dokumentenindex vor (erster Modelldownload kann einige Minuten dauern) ...", flush=True)
    result = RagEngine(settings).index()
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    try:
        prepare()
    except Exception:
        raise SystemExit("Lokale Vorbereitung fehlgeschlagen. PDF-Daten, Modellcache, Speicherplatz und Internet für den ersten Download prüfen.") from None
