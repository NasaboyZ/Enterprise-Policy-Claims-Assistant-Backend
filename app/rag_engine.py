"""Page-aware PDF indexing and persistent cosine retrieval (phase 3)."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.local_embeddings import LocalEmbeddings

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
TOP_K = 3


class RagError(RuntimeError):
    """An unavailable or inconsistent local knowledge index."""


@dataclass(frozen=True)
class RetrievedChunk:
    source_id: str
    chunk_id: str
    filename: str
    page: int
    text: str
    cosine_similarity: float


class RagEngine:
    def __init__(self, settings: Settings | None = None, embeddings: Embeddings | None = None):
        self.settings = settings or Settings.from_env()
        self.embeddings = embeddings

    def _store(self, *, create: bool) -> Chroma:
        if not create and not (self.settings.chroma_dir / "chroma.sqlite3").is_file():
            raise RagError("Index fehlt. Zuerst python -m app.rag_engine index ausführen.")
        client = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        expected = {"embedding_model": self.settings.embedding_model, "schema_version": 3,
                    "embedding_provider": "local", "embedding_format": "fastembed-passage-query-v1"}
        try:
            collection = client.get_collection(self.settings.collection_name)
        except NotFoundError:
            if not create:
                raise RagError("Dokument-Collection fehlt. Bitte zuerst indexieren.") from None
            collection = client.create_collection(
                self.settings.collection_name, metadata=expected,
                configuration={"hnsw": {"space": "cosine"}},
            )
        if any((collection.metadata or {}).get(k) != v for k, v in expected.items()):
            raise RagError("Index passt nicht zu Embedding-Modell oder Schema. Separaten Index verwenden.")
        if collection.configuration.get("hnsw", {}).get("space") != "cosine":
            raise RagError("Der vorhandene Index verwendet keine Cosine Similarity.")
        return Chroma(client=client, collection_name=self.settings.collection_name,
                      embedding_function=self.embeddings, create_collection_if_not_exists=False)

    def _embeddings(self) -> Embeddings:
        if self.embeddings is None:
            self.embeddings = LocalEmbeddings(model=self.settings.embedding_model,
                                              cache_dir=self.settings.embedding_cache_dir)
        return self.embeddings

    def load_chunks(self):
        """Read all inputs before touching the index; fail on unreadable/scanned PDFs."""
        if not self.settings.data_dir.is_dir():
            raise RagError(f"PDF-Verzeichnis fehlt: {self.settings.data_dir}")
        chunks = []
        paths = sorted(p for p in self.settings.data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
        for path in paths:
            chunks.extend(self.load_pdf_chunks(path))
        return chunks

    @staticmethod
    def load_pdf_chunks(path, filename=None):
        filename = filename or path.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            pages = PyPDFLoader(str(path), mode="page").load()
        except Exception as exc:
            raise RagError(f"PDF kann nicht gelesen werden: {filename}") from exc
        if not pages or not any(page.page_content.strip() for page in pages):
            raise RagError(f"PDF enthält keinen extrahierbaren Text: {filename} (OCR nicht implementiert).")
        for page in pages:
            page.metadata = {"filename": filename, "page": int(page.metadata["page"]) + 1,
                             "file_sha256": digest}
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(pages)
        for index, chunk in enumerate(chunks):
            identity = f"v1:{filename}:{digest}:{index}:{chunk.page_content}"
            chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode()).hexdigest()
        return chunks

    def index(self) -> dict:
        chunks = self.load_chunks()
        if not chunks and not (self.settings.chroma_dir / "chroma.sqlite3").is_file():
            raise RagError("Keine PDFs in data/ gefunden. Bitte zuerst Beispieldaten erzeugen.")
        store = self._store(create=True)
        existing = set(store.get(include=[])["ids"])
        desired = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}
        additions = sorted(set(desired) - existing)
        removals = sorted(existing - set(desired))
        # Obtain all new embeddings first. An embedding failure leaves existing content intact.
        if additions:
            documents = [desired[identifier] for identifier in additions]
            vectors = self._embeddings().embed_documents([d.page_content for d in documents])
            client = chromadb.PersistentClient(path=str(self.settings.chroma_dir),
                                              settings=ChromaSettings(anonymized_telemetry=False))
            collection = client.get_collection(self.settings.collection_name)
            # Small batches also support larger future PDF collections.
            for start in range(0, len(documents), 100):
                batch = documents[start:start + 100]
                collection.upsert(
                    ids=additions[start:start + 100], embeddings=vectors[start:start + 100],
                    documents=[d.page_content for d in batch], metadatas=[d.metadata for d in batch],
                )
        for start in range(0, len(removals), 100):
            store.delete(ids=removals[start:start + 100])
        return {"chunks": len(desired), "added": len(additions), "removed": len(removals),
                "unchanged": len(existing & set(desired))}

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Die Suchanfrage darf nicht leer sein.")
        store = self._store(create=False)
        if not store.get(limit=1, include=[])["ids"]:
            raise RagError("Index ist leer. PDFs hinzufügen und erneut indexieren.")
        vector = self._embeddings().embed_query(query)
        results = store.similarity_search_by_vector_with_relevance_scores(vector, k=TOP_K)
        # Chroma's vector method returns cosine DISTANCE, despite the method name.
        return [RetrievedChunk(
            source_id=f"S{index}", chunk_id=document.metadata["chunk_id"],
            filename=document.metadata["filename"], page=int(document.metadata["page"]),
            text=document.page_content, cosine_similarity=max(-1.0, min(1.0, 1.0 - distance)),
        ) for index, (document, distance) in enumerate(results, start=1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="Alle PDFs mit dem lokalen Index abgleichen")
    search = sub.add_parser("search", help="Drei ähnliche Abschnitte suchen")
    search.add_argument("query")
    args = parser.parse_args()
    try:
        engine = RagEngine()
        result = engine.index() if args.command == "index" else [asdict(c) for c in engine.retrieve(args.query)]
    except (RagError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
