import os
from contextlib import asynccontextmanager
from typing import Any

import psycopg2
import requests
from fastapi import FastAPI, HTTPException
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel


DB_URL = os.getenv("DATABASE_URL")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2:3b"
VECTOR_DIMENSIONS = 768


class IngestPayload(BaseModel):
    text: str


class AskPayload(BaseModel):
    question: str


def get_db() -> Any:
    """Open a PostgreSQL connection configured for pgvector."""
    if not DB_URL:
        raise RuntimeError("DATABASE_URL n'est pas définie.")

    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    return conn


def call_ollama(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call Ollama and return its JSON response or raise an HTTP error."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}{endpoint}",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama est inaccessible: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama a renvoyé une réponse JSON invalide.",
        ) from exc


def get_embedding(text: str) -> list[float]:
    result = call_ollama(
        "/api/embeddings",
        {"model": EMBEDDING_MODEL, "prompt": text},
    )
    embedding = result.get("embedding")
    if (
        not isinstance(embedding, list)
        or len(embedding) != VECTOR_DIMENSIONS
        or not all(isinstance(value, (int, float)) for value in embedding)
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "Ollama a renvoyé un embedding invalide : "
                f"{VECTOR_DIMENSIONS} nombres sont attendus."
            ),
        )
    return [float(value) for value in embedding]


def setup_db() -> None:
    """Create the pgvector extension and the documents table if necessary."""
    if not DB_URL:
        raise RuntimeError("DATABASE_URL n'est pas définie.")

    # L'extension doit exister avant d'enregistrer le type vector auprès de
    # psycopg2. Cette connexion est donc volontairement ouverte sans
    # register_vector().
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({VECTOR_DIMENSIONS}) NOT NULL
                );
                """
            )
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/ingest")
def ingest(payload: IngestPayload) -> dict[str, str]:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    vector = get_embedding(payload.text)
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s);",
                (payload.text, vector),
            )
        conn.commit()
    finally:
        conn.close()

    return {"status": "success"}


@app.post("/ask")
def ask(payload: AskPayload) -> dict[str, str]:
    if not payload.question.strip():
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide.",
        )

    query_vector = get_embedding(payload.question)
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT 1;
                """,
                (query_vector,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Aucun document n'est encore indexé.",
        )

    context = row[0]
    prompt = f"""Réponds à la question en utilisant uniquement le contexte.
Si le contexte ne contient pas la réponse, dis-le clairement.

Contexte:
{context}

Question:
{payload.question}
"""
    result = call_ollama(
        "/api/generate",
        {
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )
    response = result.get("response")
    if not isinstance(response, str) or not response.strip():
        raise HTTPException(
            status_code=502,
            detail="Ollama n'a pas renvoyé de réponse exploitable.",
        )

    return {"source_utilisee": context, "reponse": response}