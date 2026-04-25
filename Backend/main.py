"""
main.py
-------
FastAPI RAG backend — v4 (history + cleaned-up module API).

Endpoints:
  POST /upload          — index a .txt document
  GET  /search          — semantic search with LLM query enhancement
  POST /ask             — RAG: semantic search + Gemini-generated answer
  GET  /history         — persisted search history
  GET  /documents       — list indexed documents
  GET  /health          — liveness + LLM usage stats
  GET  /usage           — LLM call counter detail

Module responsibilities (strict separation):
  embedder.py  — local embeddings + chunking + heuristic title extraction
  store.py     — chunk-level FAISS index + grouped search
  topic.py     — document-level topic FAISS index
  llm.py       — Gemini API: enhance_query / generate_answer / generate_title
  limiter.py   — LLM budget: can_call_llm() / increment_usage() / status()
  history.py   — persistent search history: add() / get()

LLM call budget per request:
  /upload  → 1 call  (generate_title)
  /search  → 1 call  (enhance_query)
  /ask     → 2 calls (enhance_query + generate_answer)

All LLM calls are guarded:
    if limiter.can_call_llm():
        limiter.increment_usage()
        result = llm_function(...)
    else:
        result = fallback
        llm_warning = "LLM limit reached, using basic search"
"""

import uuid
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embedder import Embedder, chunk_text, extract_title, EMBEDDING_DIM
from store import VectorStore
from topic import TopicStore
from llm import enhance_query, generate_answer, generate_title
from limiter import UsageLimiter
from history import SearchHistory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRELOAD_DIR = Path("data/preload_docs")

# ---------------------------------------------------------------------------
# Module singletons — created once at startup, shared across all requests
# ---------------------------------------------------------------------------

embedder    = Embedder()
chunk_store = VectorStore(dim=EMBEDDING_DIM)
topic_store = TopicStore(dim=EMBEDDING_DIM)
limiter     = UsageLimiter()
history     = SearchHistory()

# ---------------------------------------------------------------------------
# Duplicate detection map: filename → doc_id
# Rebuilt from persisted chunk metadata so restarts don't re-index files.
# ---------------------------------------------------------------------------

_filename_to_doc: dict[str, str] = {
    meta["title"]: meta["doc_id"]
    for meta in chunk_store.metadata
    if "title" in meta
}


# ---------------------------------------------------------------------------
# Core indexing helper (shared by /upload and preload)
# ---------------------------------------------------------------------------

def _index_file(filepath: Path) -> dict:
    """
    Full indexing pipeline for a single .txt file.

    Returns a result dict with keys:
      doc_id, title, num_chunks, is_duplicate, llm_warning
      OR: error (str) if something went wrong

    Never raises — callers check for "error" key.

    Pipeline:
      1. Read + validate text
      2. Duplicate guard (filename-based, idempotent across restarts)
      3. LLM title generation (1 call if budget allows, else heuristic)
      4. Chunk → embed (local, no API call)
      5. Persist to chunk FAISS + topic FAISS
    """
    try:
        text = filepath.read_text(encoding="utf-8").strip()
    except Exception as exc:
        return {"error": f"Cannot read file: {exc}"}

    if not text:
        return {"error": "File is empty."}

    filename = filepath.name

    # --- Idempotent duplicate guard ---
    if filename in _filename_to_doc:
        doc_id = _filename_to_doc[filename]
        return {
            "doc_id":       doc_id,
            "title":        topic_store.topics.get(doc_id, {}).get("title", filename),
            "num_chunks":   0,
            "is_duplicate": True,
            "llm_warning":  "",
        }

    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)
    if not chunks:
        return {"error": "No chunks could be extracted."}

    # --- Title: LLM if budget allows, heuristic otherwise ---
    llm_warning = ""
    if limiter.can_call_llm():
        limiter.increment_usage()
        llm_title = generate_title(chunks[0])       # generate_title returns "" on failure
        title = llm_title or extract_title(text, filename=filename)
    else:
        llm_warning = "LLM limit reached, using basic search"
        title = extract_title(text, filename=filename)

    # --- Embed (local SentenceTransformer — no API call) ---
    embeddings = embedder.embed(chunks)

    # --- Persist ---
    chunk_store.add_chunks(doc_id, chunks, embeddings.copy(), title=title)
    topic_store.add_document(doc_id, embeddings.copy(), title=title)
    _filename_to_doc[filename] = doc_id

    return {
        "doc_id":       doc_id,
        "title":        title,
        "num_chunks":   len(chunks),
        "is_duplicate": False,
        "llm_warning":  llm_warning,
    }


# ---------------------------------------------------------------------------
# Startup preload
# ---------------------------------------------------------------------------

def _preload_documents() -> None:
    """
    Auto-index every .txt file in PRELOAD_DIR before the server accepts requests.

    - Creates the directory if it doesn't exist
    - Skips already-indexed files (idempotent — safe to call on every restart)
    - Logs a clear line per file so startup output is easy to scan
    """
    PRELOAD_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PRELOAD_DIR.glob("*.txt"))

    if not files:
        print(f"[Preload] '{PRELOAD_DIR}' is empty. Drop .txt files here for auto-indexing.")
        return

    print(f"[Preload] Found {len(files)} file(s). Indexing new files...")
    for fp in files:
        r = _index_file(fp)
        if "error" in r:
            print(f"[Preload]   ✗ {fp.name}: {r['error']}")
        elif r["is_duplicate"]:
            print(f"[Preload]   ↷ {fp.name}: already indexed, skipped.")
        else:
            warn = f"  [{r['llm_warning']}]" if r["llm_warning"] else ""
            print(
                f"[Preload]   ✓ {fp.name} → '{r['title']}' "
                f"({r['num_chunks']} chunks){warn}"
            )

    print(
        f"[Preload] Complete. "
        f"{topic_store.total_documents} docs / {chunk_store.total_chunks} chunks total."
    )


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _preload_documents()
    yield  # server is live from here


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic Search + RAG API",
    description=(
        "Production-ready RAG backend. "
        "FAISS vector search + Google Gemini + persistent history. "
        "LLM features degrade gracefully to pure semantic search when the budget is exhausted. "
        "All similarity scores are cosine similarity ∈ [0, 1]."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChunkHit(BaseModel):
    chunk_id: int
    text:     str
    score:    float   # cosine similarity ∈ [0, 1], 3 d.p.


class DocumentMatch(BaseModel):
    doc_id:    str
    title:     str
    top_score: float   # best chunk score for this document
    chunks:    List[ChunkHit]


class TopicResult(BaseModel):
    doc_id:      str
    title:       str
    topic_score: float   # cosine similarity ∈ [0, 1], 3 d.p.


class SearchMeta(BaseModel):
    total_documents: int
    total_chunks:    int
    query_enhanced:  bool   # did LLM rewrite the query?
    enhanced_query:  str    # the query actually used for embedding
    llm_warning:     str    # non-empty when LLM was skipped


class SearchResponse(BaseModel):
    query:          str
    meta:           SearchMeta
    direct_matches: List[DocumentMatch]
    related_topics: List[TopicResult]


class UploadResponse(BaseModel):
    doc_id:          str
    title:           str
    num_chunks:      int
    is_duplicate:    bool
    total_documents: int
    total_chunks:    int
    llm_warning:     str
    message:         str


class AskRequest(BaseModel):
    query:   str
    k:       int = 8    # chunks to retrieve for context
    topic_k: int = 5    # topic docs for hybrid boost


class SourceChunk(BaseModel):
    doc_id: str
    title:  str
    text:   str
    score:  float


class AskResponse(BaseModel):
    query:          str
    enhanced_query: str
    answer:         str
    sources:        List[SourceChunk]
    llm_warning:    str
    usage:          dict   # limiter.status()


class HistoryEntry(BaseModel):
    query:      str
    timestamp:  str
    user:       str
    top_result: str


class HistoryResponse(BaseModel):
    total:   int
    history: List[HistoryEntry]


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------

def _build_context(chunks: List[dict], word_budget: int = 2000) -> str:
    """
    Concatenate chunk texts into a single context string for LLM consumption.
    Labels each chunk with its source document title.
    Stops when word_budget is exhausted to keep token cost predictable.
    """
    parts = []
    remaining = word_budget
    for i, chunk in enumerate(chunks):
        if remaining <= 0:
            break
        words = chunk["text"].split()
        trimmed = " ".join(words[:remaining])
        parts.append(f"[Source {i+1}: {chunk.get('title', chunk['doc_id'])}]\n{trimmed}")
        remaining -= len(words)
    return "\n\n".join(parts)


def _apply_hybrid_boost(chunks: List[dict], boost_map: dict) -> List[dict]:
    """
    Apply topic-score boost to flat chunk list and re-sort.
    Boost weight 0.15: topic signal is meaningful but chunk similarity stays dominant.
    """
    for c in chunks:
        boost = boost_map.get(c["doc_id"], 0.0) * 0.15
        c["score"] = round(min(c["score"] + boost, 1.0), 3)
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse, summary="Upload and index a .txt document")
async def upload_document(
    file: UploadFile = File(..., description="Plain UTF-8 .txt file"),
):
    """
    Indexing pipeline:
      1. Validate file (must be .txt, valid UTF-8, non-empty)
      2. Write to temp file so _index_file can read it uniformly
      3. LLM title generation (1 call if budget allows, else heuristic)
      4. Chunk → embed → store in chunk FAISS + topic FAISS
      5. Return doc metadata

    LLM budget: 1 call per upload.
    Duplicate filenames: skipped (returns is_duplicate=True).
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are accepted.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Write content to a temp file named after the uploaded file so _index_file
    # can apply the filename-based duplicate check correctly
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_file = tmp_dir / file.filename
    try:
        tmp_file.write_text(text, encoding="utf-8")
        result = _index_file(tmp_file)
    finally:
        # Always clean up — even if indexing fails
        if tmp_file.exists():
            tmp_file.unlink()
        tmp_dir.rmdir()

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return UploadResponse(
        doc_id=result["doc_id"],
        title=result["title"],
        num_chunks=result["num_chunks"],
        is_duplicate=result["is_duplicate"],
        total_documents=topic_store.total_documents,
        total_chunks=chunk_store.total_chunks,
        llm_warning=result["llm_warning"],
        message=(
            f"'{result['title']}' already indexed — skipped."
            if result["is_duplicate"]
            else (
                f"Indexed {result['num_chunks']} chunks as '{result['title']}'. "
                f"Total: {chunk_store.total_chunks} chunks / "
                f"{topic_store.total_documents} docs."
            )
        ),
    )


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search with optional LLM query enhancement",
)
def search(
    query:   str = Query(..., description="Natural language search query"),
    k:       int = Query(10, ge=1, le=50,  description="Chunks to retrieve before grouping"),
    topic_k: int = Query(5,  ge=1, le=20,  description="Related topic documents to consider"),
    user:    str = Query("guest",          description="User identifier for history tracking"),
):
    """
    Search pipeline:
      1. LLM query enhancement (1 call if budget allows)
      2. Embed the (possibly enhanced) query
      3. Topic-level FAISS search → boost map
      4. Chunk-level FAISS search + hybrid boost → grouped by document
      5. Save query to history
      6. Return grouped results + topic matches + meta

    LLM budget: 1 call per /search request.
    Degrades to pure semantic search when budget is exhausted.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Fast-path: nothing indexed
    if chunk_store.total_chunks == 0:
        history.add(query=query, user=user, top_result="")
        return SearchResponse(
            query=query,
            meta=SearchMeta(
                total_documents=0,
                total_chunks=0,
                query_enhanced=False,
                enhanced_query=query,
                llm_warning="No documents indexed yet.",
            ),
            direct_matches=[],
            related_topics=[],
        )

    # --- Step 1: LLM query enhancement (1 call, guarded) ---
    llm_warning = ""
    if limiter.can_call_llm():
        limiter.increment_usage()
        enhanced = enhance_query(query)
        query_enhanced = enhanced.strip() != query.strip()
    else:
        enhanced = query
        query_enhanced = False
        llm_warning = "LLM limit reached, using basic search"

    # --- Step 2: Embed ---
    query_vec = embedder.embed_one(enhanced)

    # --- Step 3: Topic search → boost map ---
    raw_topics = topic_store.find_related_docs(query_vec.copy(), k=topic_k)
    boost_map  = {t["doc_id"]: t["topic_score"] for t in raw_topics}

    # --- Step 4: Chunk search + grouping ---
    grouped = chunk_store.search_grouped(query_vec.copy(), k=k, boost_doc_ids=boost_map)

    # --- Step 5: History ---
    top_result = grouped[0]["chunks"][0]["text"] if grouped else ""
    history.add(query=query, user=user, top_result=top_result)

    direct_matches = [
        DocumentMatch(
            doc_id=g["doc_id"],
            title=g["title"],
            top_score=g["top_score"],
            chunks=[ChunkHit(**c) for c in g["chunks"]],
        )
        for g in grouped
    ]
    related_topics = [TopicResult(**t) for t in raw_topics]

    return SearchResponse(
        query=query,
        meta=SearchMeta(
            total_documents=topic_store.total_documents,
            total_chunks=chunk_store.total_chunks,
            query_enhanced=query_enhanced,
            enhanced_query=enhanced,
            llm_warning=llm_warning,
        ),
        direct_matches=direct_matches,
        related_topics=related_topics,
    )


@app.post("/ask", response_model=AskResponse, summary="RAG: answer a question from indexed docs")
def ask(body: AskRequest, user: str = Query("guest", description="User identifier")):
    """
    RAG pipeline:
      1. LLM query enhancement (1 call if budget allows)
      2. Embed the query
      3. Topic search → boost map
      4. Chunk retrieval + hybrid boost → context
      5. LLM answer generation from context (1 call if budget allows)
      6. Save query to history
      7. Return answer + sources + usage stats

    LLM budget: up to 2 calls per /ask request.
    Degrades gracefully: if budget is exhausted, sources are still returned.
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if chunk_store.total_chunks == 0:
        history.add(query=body.query, user=user, top_result="")
        return AskResponse(
            query=body.query,
            enhanced_query=body.query,
            answer="No documents indexed yet. Please upload documents first.",
            sources=[],
            llm_warning="No documents indexed yet.",
            usage=limiter.status(),
        )

    warnings: List[str] = []

    # --- Step 1: Query enhancement (LLM call #1) ---
    if limiter.can_call_llm():
        limiter.increment_usage()
        enhanced = enhance_query(body.query)
    else:
        enhanced = body.query
        warnings.append("LLM limit reached, using basic search")

    # --- Step 2: Retrieval ---
    query_vec  = embedder.embed_one(enhanced)
    boost_map  = topic_store.topic_score_map(query_vec.copy(), k=body.topic_k)
    raw_chunks = chunk_store.search(query_vec.copy(), k=body.k)
    raw_chunks = _apply_hybrid_boost(raw_chunks, boost_map)

    sources = [
        SourceChunk(
            doc_id=c["doc_id"],
            title=c.get("title", c["doc_id"]),
            text=c["text"],
            score=c["score"],
        )
        for c in raw_chunks
    ]

    # --- Step 3: Answer generation (LLM call #2) ---
    if limiter.can_call_llm():
        limiter.increment_usage()

        # Reduce context for stability
        context = _build_context(raw_chunks[:1])
        context = context[:800]  # hard cap

        answer = generate_answer(enhanced, context)

        # --- SMART CLEAN FALLBACK if LLM fails ---
        if not answer or "failed" in answer.lower():
            if raw_chunks:
                text = raw_chunks[0]["text"]
                text = text.strip().replace("\n", " ")

                # Split into sentences and remove weak ones
                sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

                best_sentence = sentences[0] if sentences else text[:200]

                answer = f"Answer based on documents: {best_sentence}."
            else:
                answer = "No relevant information found."

    else:
        warnings.append("LLM limit reached, using basic search")

        # Fallback when LLM not allowed
        if raw_chunks:
            text = raw_chunks[0]["text"]
            text = text.strip().replace("\n", " ")

            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
            best_sentence = sentences[0] if sentences else text[:200]

            answer = f"Answer based on documents: {best_sentence}."
        else:
            answer = "No relevant information found."

    # --- Step 4: History ---
    top_result = raw_chunks[0]["text"] if raw_chunks else ""
    history.add(query=body.query, user=user, top_result=top_result)

    return AskResponse(
        query=body.query,
        enhanced_query=enhanced,
        answer=answer,
        sources=sources,
        llm_warning=" | ".join(dict.fromkeys(warnings)),  # deduplicated
        usage=limiter.status(),
    )


@app.get("/history", response_model=HistoryResponse, summary="Retrieve search history")
def get_history(
    user: str = Query("", description="Filter by user. Leave empty to get all history."),
):
    """
    Return persisted search history in reverse chronological order (newest first).
    Each entry includes the query, timestamp, user, and the top result snippet.
    """
    entries = history.get(user=user)
    return HistoryResponse(
        total=len(entries),
        history=[HistoryEntry(**e) for e in entries],
    )


@app.get("/documents", summary="List all indexed documents")
def list_documents():
    return {
        "total_documents": topic_store.total_documents,
        "total_chunks":    chunk_store.total_chunks,
        "documents":       topic_store.list_documents(),
    }


@app.get("/health", summary="Liveness check + LLM usage stats")
def health():
    return {
        "status":          "ok",
        "total_documents": topic_store.total_documents,
        "total_chunks":    chunk_store.total_chunks,
        "llm":             limiter.status(),
    }


@app.get("/usage", summary="LLM call counter detail")
def usage_stats():
    """Expose limiter stats for monitoring or frontend usage banners."""
    return limiter.status()


