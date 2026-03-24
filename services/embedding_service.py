"""
Embedding service — dense + sparse embedding using fastembed.
Self-contained: no imports from scripts/.
Consolidates scripts/embed_chunks.Embedder into EmbeddingService.
"""

from typing import List, Tuple

from fastembed import SparseTextEmbedding, TextEmbedding

_DENSE_MODEL  = "BAAI/bge-small-en-v1.5"
_SPARSE_MODEL = "Qdrant/bm25"


class EmbeddingService:
    """
    Lazy-loaded dense + sparse embedder.
    Use EmbeddingService.from_config(col_cfg) to construct from qdrant.yaml config.
    """

    def __init__(self, dense_model: str = _DENSE_MODEL, sparse_model: str = _SPARSE_MODEL):
        self._dense_model_name  = dense_model
        self._sparse_model_name = sparse_model
        self._dense  = None
        self._sparse = None

    @classmethod
    def from_config(cls, col_cfg: dict) -> "EmbeddingService":
        dense_model = col_cfg["vectors"]["dense"].get("model", _DENSE_MODEL)
        return cls(dense_model=dense_model)

    # ── internal lazy loaders ──────────────────────────────────────────────

    def _get_dense(self):
        if self._dense is None:
            print(f"  Loading dense model: {self._dense_model_name}")
            self._dense = TextEmbedding(self._dense_model_name)
        return self._dense

    def _get_sparse(self):
        if self._sparse is None:
            print(f"  Loading sparse model: {self._sparse_model_name}")
            self._sparse = SparseTextEmbedding(self._sparse_model_name)
        return self._sparse

    # ── document embedding ─────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> Tuple[list, list]:
        """Return (dense_vecs, sparse_vecs) for a batch of document texts."""
        dense  = list(self._get_dense().embed(texts))
        sparse = list(self._get_sparse().embed(texts))
        return dense, sparse

    # ── query embedding ────────────────────────────────────────────────────

    def embed_query(self, query: str) -> Tuple[object, object]:
        """Return (dense_vec, sparse_emb) for a single query string."""
        dense_vec  = list(self._get_dense().query_embed([query]))[0]
        sparse_emb = list(self._get_sparse().query_embed([query]))[0]
        return dense_vec, sparse_emb
