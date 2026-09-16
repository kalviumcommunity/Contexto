"""HTTP API for the Contexto grounded-answer pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from streaming import format_sse, rag_pipeline_stream
from pydantic import BaseModel, Field, field_validator
from document_upload import process_uploaded_document, store_upload

from document_upload import process_uploaded_document, store_upload


logger = logging.getLogger(__name__)
RagPipeline = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    openai_api_key: str
    embedding_model: str
    vector_db_url: str
    collection_name: str


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Load required service configuration without hardcoding deployment values."""
    values = os.environ if environ is None else environ
    missing = [name for name in ("OPENAI_API_KEY", "VECTOR_DB_URL") if not values.get(name)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        openai_api_key=values["OPENAI_API_KEY"],
        embedding_model=values.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        vector_db_url=values["VECTOR_DB_URL"],
        collection_name=values.get("COLLECTION_NAME", "rag_chunks"),
    )


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        question = value.strip()
        if len(question) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return question


class Source(BaseModel):
    source: str
    chunk_id: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: str


def _source_models(result: Mapping[str, Any]) -> list[Source]:
    sources: list[Source] = []
    for item in result.get("sources", []):
        if isinstance(item, str):
            sources.append(Source(source=item))
            continue
        if not isinstance(item, Mapping) or not item.get("source"):
            continue
        sources.append(
            Source(
                source=str(item["source"]),
                chunk_id=item.get("chunk_id") or item.get("id"),
                score=item.get("score"),
            )
        )
    return sources


def _unconfigured_pipeline(_question: str) -> Mapping[str, Any]:
    raise RuntimeError("RAG pipeline is not configured")


def create_app(
    rag_pipeline: RagPipeline | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the API application around an injected RAG pipeline."""
    pipeline = rag_pipeline or _unconfigured_pipeline
    app = FastAPI(title="Contexto RAG API")

    @app.post("/query", response_model=QueryResponse)
    def query_rag(request: QueryRequest) -> QueryResponse:
        try:
            if settings is None:
                load_settings()
            result = pipeline(request.question)
            return QueryResponse(
                answer=str(result.get("answer", "")),
                sources=_source_models(result),
                status=str(result.get("status", "answered")),
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            logger.exception("RAG request failed: %s", error)
            raise HTTPException(status_code=500, detail="RAG service failed") from error

    return app

    @app.post("/query/stream")
    async def stream_query(request: QueryRequest):
        async def events():
            try:
                async for event in rag_pipeline_stream(request.question):
                    yield format_sse(event)
            except Exception:
                yield format_sse({
                    "type": "error",
                    "message": "The answer stopped streaming. Please retry."
                })

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/documents")
    async def upload_document(file: UploadFile) -> dict[str, Any]:
        try:
            path = await store_upload(file)
            summary = process_uploaded_document(path)

            return {
                "status": "indexed",
                "filename": file.filename,
                "summary": summary,
            }

        except HTTPException:
            raise

        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

        except Exception as error:
            logger.exception(
                "Document indexing failed: %s",
                error,
            )
            raise HTTPException(
                status_code=500,
                detail="Document indexing failed",
             ) from error

    return app

app = create_app()


