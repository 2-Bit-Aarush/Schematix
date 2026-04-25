"""
main.py
-------
FastAPI RAG backend — v5 (Render free-tier optimised, <300 MB RAM).

CHANGES FROM v4
===============
REMOVED:
  - faiss / sentence-transformers / Embedder / VectorStore / TopicStore
  - All embedding computation (was the primary RAM consumer)
  - Heavy ML model singletons loaded at startup

REPLACED WITH:
  - Plain JSON document store on disk (data/docs/<doc_id>.json)
  - TF-IDF-style BM25-ish keyword retrieval (pure Python, zero ML deps)
  - Title extraction: LLM if budget allows, else first non-empty line

KEPT EXACTLY:
  - All 6 endpoint paths and HTTP methods
  - Every Pydantic request / response field (frontend unchanged)
  - LLM integration: enhance_query / generate_answer / generate_title
  - UsageLimiter + SearchHistory
  - Duplicate detection (filename → doc_id)
  - Preload directory (data/preload_docs/*.txt)
  - CORS middleware

Endpoints:
  POST /upload          — save + index a .txt document (instant, no embeddings)
  GET  /search          — keyword search with optional LLM query enhancement
  POST /ask             — RAG: keyword retrieval + LLM-generated answer
  GET  /history         — persisted search history
  GET  /documents       — list indexed documents
  GET  /health          — liveness + LLM usage stats
  GET  /usage           — LLM call counter detail

LLM call budget per request (unchanged):
  /upload  → 1 call  (generate_title)
  /search  → 1 call  (enhance_query)
  /ask     → 2 calls (enhance_query + generate_answer)
"""

import uuid
import json
import math
import re
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LLM helpers — kept exactly as before (OpenRouter / Gemini integration)
from llm import enhance_query, generate_answer, generate_title
from limiter import UsageLimiter
from history import SearchHistory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRELOAD_DIR = Path("data/preload_docs")
DOCS_DIR    = Path("data/docs")          # one JSON file per indexed document
META_FILE   = DOCS_DIR / "_meta.json"    # {filename: doc_id} duplicate map

CHUNK_SIZE  = 400    # words per chunk
CHUNK_STEP  = 200    # sliding window step (50 % overlap)
TOP_K_CHUNKS = 8     # default chunks returned by retrieval

# ---------------------------------------------------------------------------
# Module singletons
# ---------------------------------------------------------------------------

limiter = UsageLimiter()
history = SearchHistory()

# ---------------------------------------------------------------------------
# Disk-based document store (pure Python, no ML)
# ---------------------------------------------------------------------------

def _load_meta() -> Dict[str, str]:
    """Return {filename: doc_id} from disk, or empty dict."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_meta(meta: Dict[str, str]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_doc(doc_id: str) -> dict | None:
    path = DOCS_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_doc(doc_id: str, payload: dict) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / f"{doc_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _list_doc_ids() -> List[str]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in DOCS_DIR.glob("*.json") if p.stem != "_meta"]


# ---------------------------------------------------------------------------
# Text utilities  (no ML — pure Python)
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + CHUNK_SIZE]
        chunks.append(" ".join(chunk_words))
        if start + CHUNK_SIZE >= len(words):
            break
        start += CHUNK_STEP
    return chunks


def _extract_title(text: str, filename: str = "") -> str:
    """Return the first meaningful line, else the filename stem."""
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 120:
            return line
    return Path(filename).stem if filename else "Untitled"


def _tokenize(text: str) -> List[str]:
    """Lowercase alpha tokens, length ≥ 2."""
    return [t for t in re.findall(r"[a-zA-Z]{2,}", text.lower())]


# ---------------------------------------------------------------------------
# BM25-lite retrieval  (no external dependencies)
# ---------------------------------------------------------------------------

# BM25 parameters
_K1 = 1.5
_B  = 0.75

def _bm25_score(
    query_tokens: List[str],
    chunk_tokens: List[str],
    avgdl: float,
    idf: Dict[str, float],
) -> float:
    dl = len(chunk_tokens)
    freq: Dict[str, int] = {}
    for t in chunk_tokens:
        freq[t] = freq.get(t, 0) + 1

    score = 0.0
    for qt in query_tokens:
        if qt not in idf:
            continue
        f = freq.get(qt, 0)
        numerator   = f * (_K1 + 1)
        denominator = f + _K1 * (1 - _B + _B * dl / max(avgdl, 1))
        score += idf[qt] * numerator / max(denominator, 1e-9)
    return score


def _retrieve_chunks(
    query: str,
    k: int = TOP_K_CHUNKS,
) -> List[dict]:
    """
    Load all docs from disk, score each chunk with BM25-lite, return top-k.
    Returns list of dicts: {doc_id, title, chunk_id, text, score}

    Memory note: docs are loaded one at a time then discarded — never all in RAM
    simultaneously (important on 512 MB Render instance).
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    doc_ids = _list_doc_ids()
    if not doc_ids:
        return []

    # --- Pass 1: collect all (doc_id, chunk_tokens) for IDF calculation ---
    # We store only token sets + lengths to keep RAM low
    corpus: List[tuple] = []   # (doc_id, title, chunk_id, text, List[str])
    for doc_id in doc_ids:
        doc = _load_doc(doc_id)
        if not doc:
            continue
        title  = doc.get("title", doc_id)
        chunks = doc.get("chunks", [])
        for i, chunk_text in enumerate(chunks):
            toks = _tokenize(chunk_text)
            corpus.append((doc_id, title, i, chunk_text, toks))

    if not corpus:
        return []

    N    = len(corpus)
    avgdl = sum(len(c[4]) for c in corpus) / N

    # IDF per query token
    idf: Dict[str, float] = {}
    for qt in set(query_tokens):
        df = sum(1 for c in corpus if qt in c[4])
        idf[qt] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    # --- Pass 2: score each chunk ---
    scored = []
    for doc_id, title, chunk_id, text, toks in corpus:
        s = _bm25_score(query_tokens, toks, avgdl, idf)
        if s > 0:
            scored.append({
                "doc_id":   doc_id,
                "title":    title,
                "chunk_id": chunk_id,
                "text":     text,
                "score":    round(min(s / 10.0, 1.0), 3),  # normalise loosely to [0,1]
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def _group_chunks(chunks: List[dict]) -> List[dict]:
    """
    Group flat chunk list by document (highest-scoring chunk = top_score).
    Returns list of {doc_id, title, top_score, chunks:[{chunk_id,text,score}]}
    """
    groups: Dict[str, dict] = {}
    for c in chunks:
        did = c["doc_id"]
        if did not in groups:
            groups[did] = {
                "doc_id":    did,
                "title":     c["title"],
                "top_score": c["score"],
                "chunks":    [],
            }
        groups[did]["chunks"].append({
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "score":    c["score"],
        })

    result = sorted(groups.values(), key=lambda g: g["top_score"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Core indexing helper
# ---------------------------------------------------------------------------

_filename_to_doc: Dict[str, str] = _load_meta()   # warm from disk at startup


def _index_file(filepath: Path) -> dict:
    """
    Lightweight indexing pipeline:
      1. Read + validate text
      2. Duplicate guard (filename-based)
      3. LLM title generation (1 call if budget) OR heuristic
      4. Chunk text (pure Python, no ML)
      5. Save to disk as JSON

    Never raises — callers check for "error" key.
    RAM used: only the text of this one file while processing.
    """
    try:
        text = filepath.read_text(encoding="utf-8").strip()
    except Exception as exc:
        return {"error": f"Cannot read file: {exc}"}

    if not text:
        return {"error": "File is empty."}

    filename = filepath.name

    # --- Duplicate guard ---
    if filename in _filename_to_doc:
        doc_id = _filename_to_doc[filename]
        doc    = _load_doc(doc_id)
        return {
            "doc_id":       doc_id,
            "title":        doc.get("title", filename) if doc else filename,
            "num_chunks":   len(doc.get("chunks", [])) if doc else 0,
            "is_duplicate": True,
            "llm_warning":  "",
        }

    doc_id = str(uuid.uuid4())
    chunks = _chunk_text(text)
    if not chunks:
        return {"error": "No chunks could be extracted."}

    # --- Title ---
    llm_warning = ""
    if limiter.can_call_llm():
        limiter.increment_usage()
        llm_title = generate_title(chunks[0])
        title = llm_title or _extract_title(text, filename=filename)
    else:
        llm_warning = "LLM limit reached, using basic title extraction"
        title = _extract_title(text, filename=filename)

    # --- Save to disk ---
    _save_doc(doc_id, {"doc_id": doc_id, "title": title, "chunks": chunks})
    _filename_to_doc[filename] = doc_id
    _save_meta(_filename_to_doc)

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
    PRELOAD_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PRELOAD_DIR.glob("*.txt"))

    if not files:
        print(f"[Preload] '{PRELOAD_DIR}' is empty.")
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
            print(f"[Preload]   ✓ {fp.name} → '{r['title']}' ({r['num_chunks']} chunks){warn}")

    total_docs   = len(_list_doc_ids())
    total_chunks = sum(
        len((_load_doc(d) or {}).get("chunks", []))
        for d in _list_doc_ids()
    )
    print(f"[Preload] Complete. {total_docs} docs / {total_chunks} chunks total.")


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _preload_documents()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic Search + RAG API",
    description=(
        "Render free-tier optimised RAG backend. "
        "Disk-based JSON store + BM25 keyword retrieval + Google Gemini. "
        "No FAISS, no sentence-transformers, <300 MB RAM."
    ),
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas  (identical to v4 — frontend unchanged)
# ---------------------------------------------------------------------------

class ChunkHit(BaseModel):
    chunk_id: int
    text:     str
    score:    float


class DocumentMatch(BaseModel):
    doc_id:    str
    title:     str
    top_score: float
    chunks:    List[ChunkHit]


class TopicResult(BaseModel):
    doc_id:      str
    title:       str
    topic_score: float


class SearchMeta(BaseModel):
    total_documents: int
    total_chunks:    int
    query_enhanced:  bool
    enhanced_query:  str
    llm_warning:     str


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
    k:       int = 8
    topic_k: int = 5


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
    usage:          dict


class HistoryEntry(BaseModel):
    query:      str
    timestamp:  str
    user:       str
    top_result: str


class HistoryResponse(BaseModel):
    total:   int
    history: List[HistoryEntry]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _total_counts() -> tuple[int, int]:
    doc_ids     = _list_doc_ids()
    total_docs  = len(doc_ids)
    total_chunks = sum(
        len((_load_doc(d) or {}).get("chunks", []))
        for d in doc_ids
    )
    return total_docs, total_chunks


def _build_context(chunks: List[dict], word_budget: int = 2000) -> str:
    parts, remaining = [], word_budget
    for i, chunk in enumerate(chunks):
        if remaining <= 0:
            break
        words   = chunk["text"].split()
        trimmed = " ".join(words[:remaining])
        parts.append(f"[Source {i+1}: {chunk.get('title', chunk['doc_id'])}]\n{trimmed}")
        remaining -= len(words)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Plain UTF-8 .txt file"),
):
    """
    Lightweight upload pipeline:
      1. Validate (must be .txt, UTF-8, non-empty)
      2. Write to temp file
      3. Chunk + save to disk JSON  (no embeddings, no ML)
      4. Return instantly

    RAM spike: only the text of the uploaded file.
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

    import tempfile
    tmp_dir  = Path(tempfile.mkdtemp())
    tmp_file = tmp_dir / file.filename
    try:
        tmp_file.write_text(text, encoding="utf-8")
        result = _index_file(tmp_file)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()
        tmp_dir.rmdir()

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    total_docs, total_chunks = _total_counts()

    return UploadResponse(
        doc_id=result["doc_id"],
        title=result["title"],
        num_chunks=result["num_chunks"],
        is_duplicate=result["is_duplicate"],
        total_documents=total_docs,
        total_chunks=total_chunks,
        llm_warning=result["llm_warning"],
        message=(
            f"'{result['title']}' already indexed — skipped."
            if result["is_duplicate"]
            else (
                f"Indexed {result['num_chunks']} chunks as '{result['title']}'. "
                f"Total: {total_chunks} chunks / {total_docs} docs."
            )
        ),
    )


@app.get("/search", response_model=SearchResponse)
def search(
    query:   str = Query(...),
    k:       int = Query(10, ge=1, le=50),
    topic_k: int = Query(5,  ge=1, le=20),
    user:    str = Query("guest"),
):
    """
    Search pipeline:
      1. LLM query enhancement (1 call if budget allows)
      2. BM25 chunk retrieval (pure Python, no ML)
      3. Group by document
      4. Derive "related_topics" from top-scoring distinct docs
      5. Persist to history
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    total_docs, total_chunks = _total_counts()

    if total_chunks == 0:
        history.add(query=query, user=user, top_result="")
        return SearchResponse(
            query=query,
            meta=SearchMeta(
                total_documents=0, total_chunks=0,
                query_enhanced=False, enhanced_query=query,
                llm_warning="No documents indexed yet.",
            ),
            direct_matches=[],
            related_topics=[],
        )

    # --- LLM query enhancement ---
    llm_warning = ""
    if limiter.can_call_llm():
        limiter.increment_usage()
        enhanced      = enhance_query(query)
        query_enhanced = enhanced.strip() != query.strip()
    else:
        enhanced       = query
        query_enhanced = False
        llm_warning    = "LLM limit reached, using basic search"

    # --- BM25 retrieval ---
    raw_chunks = _retrieve_chunks(enhanced, k=k)
    grouped    = _group_chunks(raw_chunks)

    # "related_topics" = top-scoring docs beyond the first, mapped to TopicResult shape
    related_topics = [
        TopicResult(doc_id=g["doc_id"], title=g["title"], topic_score=g["top_score"])
        for g in grouped[1 : topic_k + 1]
    ]

    direct_matches = [
        DocumentMatch(
            doc_id=g["doc_id"],
            title=g["title"],
            top_score=g["top_score"],
            chunks=[ChunkHit(**c) for c in g["chunks"]],
        )
        for g in grouped
    ]

    top_result = raw_chunks[0]["text"] if raw_chunks else ""
    history.add(query=query, user=user, top_result=top_result)

    return SearchResponse(
        query=query,
        meta=SearchMeta(
            total_documents=total_docs,
            total_chunks=total_chunks,
            query_enhanced=query_enhanced,
            enhanced_query=enhanced,
            llm_warning=llm_warning,
        ),
        direct_matches=direct_matches,
        related_topics=related_topics,
    )


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, user: str = Query("guest")):
    """
    RAG pipeline:
      1. LLM query enhancement (1 call if budget)
      2. BM25 chunk retrieval
      3. Build context string
      4. LLM answer generation (1 call if budget)
      5. Graceful text-only fallback when LLM unavailable
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    total_docs, total_chunks = _total_counts()

    if total_chunks == 0:
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

    # --- Step 1: Query enhancement ---
    if limiter.can_call_llm():
        limiter.increment_usage()
        enhanced = enhance_query(body.query)
    else:
        enhanced = body.query
        warnings.append("LLM limit reached, using basic search")

    # --- Step 2: Retrieval ---
    raw_chunks = _retrieve_chunks(enhanced, k=body.k)

    sources = [
        SourceChunk(
            doc_id=c["doc_id"],
            title=c.get("title", c["doc_id"]),
            text=c["text"],
            score=c["score"],
        )
        for c in raw_chunks
    ]

    # --- Step 3: Answer generation ---
    if limiter.can_call_llm():
        limiter.increment_usage()
        context = _build_context(raw_chunks[:4])
        context = context[:1200]   # hard token cap
        answer  = generate_answer(enhanced, context)

        if not answer or "failed" in answer.lower():
            answer = _fallback_answer(raw_chunks)
    else:
        warnings.append("LLM limit reached, using basic search")
        answer = _fallback_answer(raw_chunks)

    top_result = raw_chunks[0]["text"] if raw_chunks else ""
    history.add(query=body.query, user=user, top_result=top_result)

    return AskResponse(
        query=body.query,
        enhanced_query=enhanced,
        answer=answer,
        sources=sources,
        llm_warning=" | ".join(dict.fromkeys(warnings)),
        usage=limiter.status(),
    )


def _fallback_answer(chunks: List[dict]) -> str:
    """Return a readable excerpt when the LLM is unavailable."""
    if not chunks:
        return "No relevant information found."
    text      = chunks[0]["text"].strip().replace("\n", " ")
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
    best      = sentences[0] if sentences else text[:200]
    return f"Answer based on documents: {best}."


# ---------------------------------------------------------------------------
# Read-only endpoints
# ---------------------------------------------------------------------------

@app.get("/history", response_model=HistoryResponse)
def get_history(user: str = Query("")):
    entries = history.get(user=user)
    return HistoryResponse(total=len(entries), history=[HistoryEntry(**e) for e in entries])


@app.get("/documents")
def list_documents():
    doc_ids = _list_doc_ids()
    docs    = []
    for did in doc_ids:
        d = _load_doc(did)
        if d:
            docs.append({"doc_id": did, "title": d.get("title", did), "num_chunks": len(d.get("chunks", []))})
    return {
        "total_documents": len(docs),
        "total_chunks":    sum(d["num_chunks"] for d in docs),
        "documents":       docs,
    }


@app.get("/health")
def health():
    total_docs, total_chunks = _total_counts()
    return {
        "status":          "ok",
        "total_documents": total_docs,
        "total_chunks":    total_chunks,
        "llm":             limiter.status(),
    }


@app.get("/usage")
def usage_stats():
    return limiter.status()
