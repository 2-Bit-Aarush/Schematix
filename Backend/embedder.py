"""
embedder.py
-----------
Handles all embedding logic using a local Sentence Transformers model.
No external APIs — everything runs on-device.

IMPROVEMENTS:
  - Added extract_title() heuristic (first meaningful line or filename fallback)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import re
from typing import List

# --- Model config ---
# 'all-MiniLM-L6-v2' is fast, small (~80MB), and great for semantic search.
# Swap for 'all-mpnet-base-v2' if you want higher quality at cost of speed.
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2


class Embedder:
    """
    Singleton-style wrapper around SentenceTransformer.
    Load once, reuse across requests to avoid expensive model reloads.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[Embedder] Loading model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        print("[Embedder] Model loaded.")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.
        Returns a float32 numpy array of shape (len(texts), EMBEDDING_DIM).
        FAISS requires float32 — we enforce that here.
        """
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """
        Convenience method: embed a single string.
        Returns shape (1, EMBEDDING_DIM) — ready for FAISS search.
        """
        return self.embed([text])


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split document text into overlapping word-level chunks.

    Why overlapping?
    ----------------
    If a key idea spans a chunk boundary, overlap ensures neither chunk
    misses the full context. 50-word overlap is a balanced default.

    Args:
        text:       Full document text.
        chunk_size: Max words per chunk.
        overlap:    Words shared between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        # Advance by (chunk_size - overlap) so next chunk re-uses tail context
        start += chunk_size - overlap

    # Drop any empty trailing chunks
    return [c for c in chunks if c]


def extract_title(text: str, filename: str = "") -> str:
    """
    Extract a human-readable title from document text using simple heuristics.
    No LLM calls — pure string logic.

    Strategy (in priority order):
      1. First non-empty line that looks like a title (short, no trailing period)
      2. First sentence of meaningful length (4–12 words)
      3. Cleaned-up filename (strip extension, replace underscores/hyphens)
      4. Fallback: "Untitled Document"

    Args:
        text:     Full document text.
        filename: Original filename (used as fallback).

    Returns:
        A short, readable title string (max ~80 chars).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # --- Heuristic 1: First line looks like a heading ---
    # Criteria: short (≤ 12 words), doesn't end with sentence punctuation
    if lines:
        first_line = lines[0]
        word_count = len(first_line.split())
        if 1 <= word_count <= 12 and not first_line.endswith((".", "!", "?")):
            return first_line[:80]

    # --- Heuristic 2: First sentence of the body text ---
    # Split on sentence-ending punctuation; take first that's 4–12 words
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for sent in sentences:
        words = sent.split()
        if 4 <= len(words) <= 12:
            title = sent.strip().rstrip(".!?,;:")
            return title[:80]

    # --- Heuristic 3: Filename ---
    if filename:
        name = re.sub(r'\.[^.]+$', '', filename)      # strip extension
        name = re.sub(r'[_\-]+', ' ', name)           # underscores → spaces
        name = re.sub(r'\s+', ' ', name).strip()
        if name:
            return name.title()[:80]

    return "Untitled Document"
