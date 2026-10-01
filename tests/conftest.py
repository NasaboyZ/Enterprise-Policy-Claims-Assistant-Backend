import hashlib
import math
import re

import pytest
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """Deterministic test double, never used by the production CLI."""
    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * 64
        for token in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 64] += 1
        norm = math.sqrt(sum(v * v for v in vector)) or 1
        return [v / norm for v in vector]


@pytest.fixture
def embeddings():
    return LocalEmbeddings()
