"""Local, single-worker REST API. Start with uvicorn app.main:app."""

import json
from threading import RLock

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.agent import AgentRequest, AgentResult, InsuranceAgent
from app.config import Settings
from app.provider_errors import ProviderError
from app.rag_engine import RagEngine
from app.uploads import MAX_UPLOAD_BYTES, UploadError, upload_pdf


def error_status(code: str | None) -> int:
    if code and "quota_exceeded" in code:
        return 429
    if code and any(part in code for part in ("invalid_response", "request_rejected", "invalid_citations", "generation_failed")):
        return 502
    return 503


class UploadBodyLimit:
    """Bound the whole multipart body before Starlette parses it (also chunked uploads)."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/upload":
            return await self.app(scope, receive, send)
        messages, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > MAX_UPLOAD_BYTES + 65536:
                response = JSONResponse({"error_code": "upload_too_large", "message": "PDF darf höchstens 10 MiB gross sein."}, status_code=413)
                return await response(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay():
            try:
                return next(iterator)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, *, agent=None, rag=None) -> FastAPI:
    settings = settings or Settings.from_env()
    rag = rag if rag is not None else RagEngine(settings)
    agent = agent if agent is not None else InsuranceAgent(settings, retriever=rag)
    lock = RLock()
    app = FastAPI(title="Enterprise Policy & Claims Assistant", version="1.0.0")
    app.add_middleware(UploadBodyLimit)
    app.state.index_available = True

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/chat", response_model=AgentResult)
    def chat(payload: AgentRequest):
        with lock:
            if not app.state.index_available:
                return JSONResponse({"status": "error", "ml_score": None, "final_answer": "Index muss nach fehlgeschlagenem Upload repariert werden.",
                                     "sources": [], "error_code": "index_unavailable"}, status_code=503)
            try:
                result = agent.run(payload)
            except Exception:
                return JSONResponse({"status": "error", "ml_score": None, "final_answer": "Anfrage konnte nicht verarbeitet werden.",
                                     "sources": [], "error_code": "agent_failed"}, status_code=503)
        if result.status == "error":
            return JSONResponse(result.model_dump(mode="json"), status_code=error_status(result.error_code))
        return result

    def process_upload(filename, content):
        with lock:
            if not app.state.index_available:
                return JSONResponse({"error_code": "index_unavailable", "message": "Index zuerst reparieren und API neu starten."}, status_code=503)
            try:
                result = upload_pdf(rag, filename, content)
                return JSONResponse(result, status_code=201 if result["status"] == "indexed" else 200)
            except UploadError as exc:
                if exc.code == "upload_rollback_failed":
                    app.state.index_available = False
                return JSONResponse({"error_code": exc.code, "message": str(exc)}, status_code=exc.status)
            except ProviderError as exc:
                return JSONResponse({"error_code": exc.code, "message": str(exc)}, status_code=error_status(exc.code))
            except Exception:
                return JSONResponse({"error_code": "upload_failed", "message": "PDF konnte nicht indexiert werden. Index und lokalen Modellcache prüfen."}, status_code=503)

    @app.post("/api/upload")
    async def upload(request: Request, file: UploadFile = File(...)):
        form = await request.form()
        if sum(isinstance(value, StarletteUploadFile) for _, value in form.multi_items()) != 1:
            return JSONResponse({"error_code": "single_pdf_required", "message": "Genau eine PDF pro Anfrage hochladen."}, status_code=422)
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            return await run_in_threadpool(process_upload, file.filename or "", content)
        finally:
            await file.close()

    @app.get("/api/metrics")
    def metrics():
        from app.reports import EvaluationReport
        try:
            report = EvaluationReport.model_validate_json((settings.reports_dir / "latest.json").read_text())
            return report.model_dump(mode="json")
        except FileNotFoundError:
            return JSONResponse({"error_code": "metrics_missing", "message": "Noch keine Evaluation vorhanden. python -m app.eval ausführen."}, status_code=404)
        except Exception:
            return JSONResponse({"error_code": "metrics_unreadable", "message": "Evaluationsbericht ist nicht lesbar."}, status_code=503)

    return app


app = create_app()
