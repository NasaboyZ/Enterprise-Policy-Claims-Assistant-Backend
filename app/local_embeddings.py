"""CPU embeddings for German/English text; only the initial model download uses the network."""

from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from app.provider_errors import ProviderError


class LocalEmbeddings(Embeddings):
    def __init__(self, model: str, cache_dir: Path):
        self.model_name = model
        self.cache_dir = cache_dir
        self.model = None

    def _load(self):
        if self.model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise ProviderError("local_embeddings_missing", "Lokale Embeddings fehlen. Bitte python -m pip install -r requirements.txt ausführen.") from None
            try:
                self.model = TextEmbedding(model_name=self.model_name, cache_dir=str(self.cache_dir),
                                           threads=2, providers=["CPUExecutionProvider"])
            except Exception:
                raise ProviderError("local_embeddings_unavailable", "Das lokale Embedding-Modell konnte nicht geladen werden. Für den ersten Modelldownload ist Internet nötig; danach läuft die Suche lokal. Speicherplatz und models/embeddings prüfen.") from None
        return self.model

    def _embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        try:
            # Jina v2 German needs no query/passage prefixes. FastEmbed applies
            # the model's own pooling and normalization locally through ONNX.
            output = model.query_embed(texts) if query else model.passage_embed(texts, batch_size=16)
            vectors = [np.asarray(vector, dtype=float) for vector in output]
            if (len(vectors) != len(texts)
                    or any(v.ndim != 1 or not v.size or not np.isfinite(v).all()
                           or np.linalg.norm(v) == 0 for v in vectors)
                    or len({v.size for v in vectors}) != 1):
                raise ValueError("Invalid local embedding vectors")
            return [vector.tolist() for vector in vectors]
        except Exception:
            raise ProviderError("local_embeddings_failed", "Die lokale Textvektor-Berechnung ist fehlgeschlagen. Modellcache und verfügbaren Arbeitsspeicher prüfen.") from None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, query=False)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], query=True)[0]
