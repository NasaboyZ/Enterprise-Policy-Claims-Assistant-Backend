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
    collection_name: str = "insurance_local_jina_de_v1"
    embedding_model: str = "jinaai/jina-embeddings-v2-base-de"
    embedding_cache_dir: Path = ROOT / "models" / "embeddings"
    reports_dir: Path = ROOT / "reports"
    chat_model: str = "gemini-2.5-flash-lite"
    chat_provider: str = "aion"
    aion_chat_model: str = "aion-labs/aion-2.0"
    fraud_threshold: float = 0.5

    def __post_init__(self):
        if self.chat_provider not in {"aion", "gemini"}:
            raise ValueError("CHAT_PROVIDER muss aion oder gemini sein.")
        if not self.aion_chat_model.strip():
            raise ValueError("AION_CHAT_MODEL darf nicht leer sein.")
        if not 0 <= self.fraud_threshold <= 1:
            raise ValueError("FRAUD_THRESHOLD muss zwischen 0 und 1 liegen.")

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", str(ROOT / "data"))),
            model_path=Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "fraud_model.pkl"))),
            chroma_dir=Path(os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma"))),
            embedding_cache_dir=Path(os.getenv("EMBEDDING_CACHE_DIR", str(ROOT / "models" / "embeddings"))),
            reports_dir=Path(os.getenv("REPORTS_DIR", str(ROOT / "reports"))),
            chat_provider=os.getenv("CHAT_PROVIDER", "aion").strip().lower(),
            aion_chat_model=os.getenv("AION_CHAT_MODEL", "aion-labs/aion-2.0").strip(),
            chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite"),
            fraud_threshold=float(os.getenv("FRAUD_THRESHOLD", "0.5")),
        )


def require_google_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key.startswith("YOUR_"):
        raise ProviderError("google_key_missing", "GOOGLE_API_KEY fehlt. Den neuen Gemini-Key lokal in .env eintragen.")
    return key


def require_aion_key() -> str:
    key = os.getenv("AION_API_KEY", "").strip()
    if not key or key.startswith("YOUR_"):
        raise ProviderError("aion_key_missing", "AION_API_KEY fehlt. Den AionLabs-Key lokal in .env eintragen.")
    return key
