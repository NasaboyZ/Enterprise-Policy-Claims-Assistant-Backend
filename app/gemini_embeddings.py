"""LangChain adapter for Gemini Embedding 2 via the official Google SDK."""

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

from app.config import require_google_key
from app.provider_errors import translate_provider_error


class GeminiEmbeddings(Embeddings):
    """One vector per chunk: Embedding 2 would aggregate a list in one request.

    Unlike Embedding 001, Embedding 2 takes retrieval instructions in the text,
    not a task_type parameter. Keep these prefixes stable for existing indexes.
    """

    def __init__(self, model: str = "gemini-embedding-2"):
        self.model = model
        self.client = genai.Client(
            api_key=require_google_key(), vertexai=False,
            http_options=types.HttpOptions(timeout=30_000,
                                          retry_options=types.HttpRetryOptions(attempts=1)),
        )

    def _embed(self, text: str) -> list[float]:
        try:
            result = self.client.models.embed_content(model=self.model, contents=text)
            if not result.embeddings or len(result.embeddings) != 1 or not result.embeddings[0].values:
                raise ValueError("Expected one non-empty vector")
            return list(result.embeddings[0].values)
        except Exception as exc:
            raise translate_provider_error(exc) from None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(f"title: none | text: {text}") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(f"task: search result | query: {text}")
