"""Validate and index one uploaded PDF without replacing unrelated documents."""

import hashlib
import io
import os
import re
import tempfile
from pathlib import Path

from pypdf import PdfReader

from app.rag_engine import RagError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PAGES = 100


class UploadError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def upload_pdf(engine, filename: str, content: bytes) -> dict:
    if (not filename or any(c in filename for c in ("/", "\\", ".."))
            or any(ord(c) < 32 for c in filename) or Path(filename).suffix.lower() != ".pdf"):
        raise UploadError(422, "invalid_filename", "Einen einfachen Dateinamen mit Endung .pdf verwenden.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadError(413, "upload_too_large", "PDF darf höchstens 10 MiB gross sein.")
    if not content.startswith(b"%PDF-"):
        raise UploadError(422, "invalid_pdf", "Die Datei ist keine lesbare PDF.")
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise UploadError(422, "encrypted_pdf", "Verschlüsselte PDFs werden nicht unterstützt.")
        if len(reader.pages) > MAX_PAGES:
            raise UploadError(422, "too_many_pages", "PDF darf höchstens 100 Seiten enthalten.")
    except UploadError:
        raise
    except Exception:
        raise UploadError(422, "invalid_pdf", "Die Datei ist keine lesbare PDF.") from None

    digest = hashlib.sha256(content).hexdigest()
    directory = engine.settings.data_dir
    directory.mkdir(parents=True, exist_ok=True)
    # Content deduplication also covers PDFs copied into data/ before the API existed.
    matching = next((p for p in sorted(directory.glob("*.pdf")) if p.is_file()
                     and hashlib.sha256(p.read_bytes()).hexdigest() == digest), None)
    stored_name = matching.name if matching else f"upload-{digest}.pdf"
    store = engine._store(create=True)
    existing = store.get(where={"file_sha256": digest}, include=["metadatas"])
    if matching and existing["ids"]:
        return {"status": "duplicate", "filename": stored_name, "sha256": digest,
                "chunks": len(existing["ids"])}

    temporary = None
    published = False
    new_ids = []
    destination = directory / stored_name
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".upload-", suffix=".tmp", delete=False) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        try:
            chunks = engine.load_pdf_chunks(temporary, filename=stored_name)
        except RagError:
            raise UploadError(422, "unreadable_pdf", "PDF enthält keinen lesbaren Text. OCR wird nicht unterstützt.") from None
        vectors = engine._embeddings().embed_documents([chunk.page_content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count mismatch")
        ids = [c.metadata["chunk_id"] for c in chunks]
        prior_ids = set(store.get(ids=ids, include=[])["ids"])
        new_ids = [identifier for identifier in ids if identifier not in prior_ids]
        # Hard link publishes without overwriting any existing file.
        if not matching:
            os.link(temporary, destination)
            published = True
        collection = store._collection
        for start in range(0, len(chunks), 100):
            indexes = [i for i in range(start, min(start + 100, len(chunks))) if ids[i] not in prior_ids]
            if indexes:
                collection.upsert(ids=[ids[i] for i in indexes], embeddings=[vectors[i] for i in indexes],
                                  documents=[chunks[i].page_content for i in indexes],
                                  metadatas=[chunks[i].metadata for i in indexes])
        return {"status": "indexed", "filename": stored_name, "sha256": digest, "chunks": len(chunks)}
    except Exception:
        try:
            if new_ids:
                store.delete(ids=new_ids)
            if published:
                destination.unlink()
        except Exception:
            raise UploadError(503, "upload_rollback_failed", "Upload-Rücknahme fehlgeschlagen. API stoppen, Index mit dem index-Befehl abgleichen und neu starten.") from None
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
