"""RAG layer: markdown KB -> header-based chunks -> local ONNX embeddings (fastembed, bge-small) -> FAISS (exact cosine).
Groq has no embeddings endpoint; local embeddings also remove a network hop from every query."""
import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # numpy exact search is equivalent at this corpus size
    faiss = None

KB_DIR = Path(__file__).resolve().parent / "kb"
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge retrieval convention
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))


def load_chunks() -> list[dict]:
    """One chunk per '## ' section; each is a self-contained role or framework, so header splitting keeps semantics intact."""
    chunks = []
    for f in sorted(KB_DIR.glob("*.md")):
        for block in re.split(r"(?m)^## ", f.read_text(encoding="utf-8"))[1:]:
            title, _, body = block.partition("\n")
            chunks.append({"id": len(chunks), "source": f.name, "title": title.strip(), "text": body.strip()})
    return chunks


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


class KnowledgeBase:
    def __init__(self):
        self.chunks: list[dict] = []
        self.index = None
        self.mat = None
        self._model = None
        self._qcache: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(EMBED_MODEL)
        return _normalize(np.array(list(self._model.embed(texts)), dtype="float32"))

    def build(self) -> None:
        self.chunks = load_chunks()
        t = time.perf_counter()
        self.mat = self._embed([f"{c['title']}\n{c['text']}" for c in self.chunks])
        if faiss is not None:
            # Exact search: tiny corpus, flat inner-product is microseconds. Use HNSW/IVF past ~100k chunks.
            self.index = faiss.IndexFlatIP(self.mat.shape[1])
            self.index.add(self.mat)
        print(f"[rag] embedded {len(self.chunks)} chunks in {time.perf_counter()-t:.1f}s")

    def search(self, query: str, k: int = 4, min_score: float = MIN_SCORE) -> list[dict]:
        qv = self._qcache.get(query)
        if qv is None:
            qv = self._embed([QUERY_PREFIX + query])
            if len(self._qcache) > 256:
                self._qcache.clear()
            self._qcache[query] = qv
        if self.index is not None:
            scores, ids = self.index.search(qv, k)
            scores, ids = scores[0], ids[0]
        else:
            sims = (self.mat @ qv[0])
            ids = np.argsort(-sims)[:k]
            scores = sims[ids]
        hits = [{**self.chunks[i], "score": float(s)} for s, i in zip(scores, ids) if i >= 0 and s >= min_score]
        if not hits and ids[0] >= 0:  # never leave the LLM with nothing: keep the single best chunk
            hits = [{**self.chunks[ids[0]], "score": float(scores[0])}]
        return hits


def format_context(hits: list[dict]) -> str:
    if not hits:
        return "(no relevant entries found)"
    return "\n\n".join(f"[{h['title']}]\n{h['text']}" for h in hits)
