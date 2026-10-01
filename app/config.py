"""Project-relative configuration; importing this module makes no API calls."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    model_path: Path = ROOT / "models" / "fraud_model.pkl"
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection_name: str = "insurance_documents_v1"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"
    fraud_threshold: float = 0.5

    def __post_init__(self):
        if not 0 <= self.fraud_threshold <= 1:
            raise ValueError("FRAUD_THRESHOLD muss zwischen 0 und 1 liegen.")

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            fraud_threshold=float(os.getenv("FRAUD_THRESHOLD", "0.5")),
        )


def require_openai_key():
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY fehlt. In .env oder der Umgebung setzen.")
