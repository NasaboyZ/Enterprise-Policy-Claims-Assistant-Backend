"""Project-relative configuration; importing this module makes no API calls."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.provider_errors import ProviderError

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    model_path: Path = ROOT / "models" / "fraud_model.pkl"
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection_name: str = "insurance_gemini_embedding2_v1"
    embedding_model: str = "gemini-embedding-2"
    chat_model: str = "gemini-2.5-flash-lite"
    fraud_threshold: float = 0.5

    def __post_init__(self):
        if not 0 <= self.fraud_threshold <= 1:
            raise ValueError("FRAUD_THRESHOLD muss zwischen 0 und 1 liegen.")

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite"),
            fraud_threshold=float(os.getenv("FRAUD_THRESHOLD", "0.5")),
        )


def require_google_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key.startswith("YOUR_"):
        raise ProviderError("google_key_missing", "GOOGLE_API_KEY fehlt. Den neuen Gemini-Key lokal in .env eintragen.")
    return key
