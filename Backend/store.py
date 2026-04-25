"""
store.py
--------
FAISS vector index + JSON metadata store.

Two parallel data structures stay in sync:
  1. FAISS index  — stores float32 embeddings for fast ANN search
  2. metadata[]   — stores chunk text, doc_id, chunk_id, title (same row order as FAISS)

FAISS doesn't store metadata, so we mirror its integer row index in our list.
Row N in FAISS  ↔  metadata[N]

IMPROVEMENTS vs v1:
  - FIX: Empty index guard — search returns [] safely when ntotal == 0
  - FIX: title stored per-chunk so grouped results can include it
  - IMPROVE: scores rounded to 3 decimal places (cosine similarity, range 0–1)
  - IMPROVE: search_grouped() returns results keyed by doc_id (no repeated doc headers)
  - IMPROVE: duplicate upload detection — warns caller if doc_id already exists
"""

import faiss
import numpy as np
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

# Paths for persisting state between server restarts
INDEX_PATH = "data/chunk_index.faiss"
META_PATH  = "data/chunk_meta.json"


class VectorStore:
    """
    Manages chunk-level FAISS index and accompanying metadata.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension (must match the embedder model output).
        """
        self.dim = dim
        self.metadata: List[Dict[str, Any]] = []  # parallel to FAISS rows

        # Try to restore a saved index; otherwise create fresh
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            print("[Store] Loading existing chunk index from disk...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r") as f:
                self.metadata = json.load(f)
            print(f"[Store] Loaded {self.index.ntotal} vectors.")
        else:
            print("[Store] Creating new FAISS index (IndexFlatIP).")
            # IndexFlatIP = Inner Product similarity (equiv. to cosine on normalized vecs)
            self.index = faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: np.ndarray,
        title: str = "",
    ) -> bool:
        """
        Add a batch of chunks from one document into the index.

        Steps:
          1. L2-normalize so inner product == cosine similarity
          2. Add vectors to FAISS
          3. Append matching metadata rows (includes title for grouping)

        Args:
            doc_id:     Unique document identifier.
            chunks:     Raw text of each chunk.
            embeddings: float32 array of shape (len(chunks), dim).
            title:      Human-readable document title (stored per chunk for grouping).

        Returns:
            True if added fresh; False if doc_id already existed (still added — caller decides).
        """
        # FIX: Warn about duplicates — check if this doc_id was already indexed
        existing_ids = {m["doc_id"] for m in self.metadata}
        is_duplicate = doc_id in existing_ids

        # Normalize each vector to unit length → cosine similarity via dot product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "doc_id":   doc_id,
                "chunk_id": i,
                "text":     chunk,
                "title":    title or doc_id,  # title stored here for grouped results
            })

        self._persist()
        return not is_duplicate

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find top-k most similar chunks to a query embedding.
        Returns a FLAT list — used internally and for hybrid ranking.

        Scores are cosine similarity values in [0, 1], rounded to 3 decimal places.

        Args:
            query_vec: float32 array of shape (1, dim).
            k:         Number of results to return.

        Returns:
            List of dicts: { doc_id, title, chunk_id, text, score }
        """
        # FIX: Guard against empty index — FAISS raises on ntotal == 0
        if self.index.ntotal == 0:
            return []

        faiss.normalize_L2(query_vec)

        # Clamp k to available vectors to avoid FAISS assertion errors
        effective_k = min(k, self.index.ntotal)

        # FAISS returns (distances, indices) — both shape (1, k)
        scores, indices = self.index.search(query_vec, effective_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 when fewer than k results exist
            meta = self.metadata[idx]
            results.append({
                "doc_id":   meta["doc_id"],
                "title":    meta.get("title", meta["doc_id"]),
                "chunk_id": meta["chunk_id"],
                "text":     meta["text"],
                # IMPROVE: Round scores to 3 d.p. for clean JSON output
                "score":    round(float(score), 3),  # cosine similarity ∈ [0, 1]
            })

        return results

    def search_grouped(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        boost_doc_ids: Dict[str, float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search and return results grouped by document — no repeated doc headers.

        Response shape per group:
          {
            "doc_id":  str,
            "title":   str,
            "chunks":  [ { "chunk_id": int, "text": str, "score": float } ],
            "top_score": float   # best chunk score for this doc (for sorting)
          }

        IMPROVE: Hybrid ranking — optionally boost scores for chunks whose
        document ranked highly in the topic search. This surfaces chunks
        from topically relevant docs even if their raw chunk score is lower.

        Args:
            query_vec:     float32 array of shape (1, dim).
            k:             Number of CHUNK results to retrieve before grouping.
            boost_doc_ids: Optional dict of { doc_id: topic_score } from TopicStore.
                           Chunks whose doc_id appears here get a small additive boost.

        Returns:
            List of grouped doc dicts, sorted by top boosted score descending.
        """
        flat = self.search(query_vec, k=k)
        if not flat:
            return []

        boost = boost_doc_ids or {}

        # Group chunks by doc_id, applying topic boost to each chunk score
        groups: Dict[str, Dict] = {}
        for hit in flat:
            doc_id = hit["doc_id"]

            # IMPROVE: Hybrid boost — weight 0.15 keeps topic signal meaningful
            # but chunk similarity remains dominant
            topic_boost = boost.get(doc_id, 0.0) * 0.15
            boosted_score = round(min(hit["score"] + topic_boost, 1.0), 3)

            if doc_id not in groups:
                groups[doc_id] = {
                    "doc_id":    doc_id,
                    "title":     hit["title"],
                    "chunks":    [],
                    "top_score": 0.0,
                }

            groups[doc_id]["chunks"].append({
                "chunk_id": hit["chunk_id"],
                "text":     hit["text"],
                "score":    boosted_score,
            })

            # Track the best score in this group for outer-level sorting
            if boosted_score > groups[doc_id]["top_score"]:
                groups[doc_id]["top_score"] = boosted_score

        # Sort groups by their best chunk score (descending)
        result = sorted(groups.values(), key=lambda g: g["top_score"], reverse=True)

        # Sort chunks within each group by score (descending)
        for group in result:
            group["chunks"].sort(key=lambda c: c["score"], reverse=True)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self):
        """Save index and metadata to disk after every write."""
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump(self.metadata, f, indent=2)

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal

    def known_doc_ids(self) -> set:
        """Return the set of all doc_ids currently in the chunk store."""
        return {m["doc_id"] for m in self.metadata}
