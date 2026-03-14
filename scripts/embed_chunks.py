#!/usr/bin/env python3
"""
Dense + sparse embedding wrapper using fastembed.

Dense model : BAAI/bge-small-en-v1.5  (384-dim, cosine)
Sparse model: Qdrant/bm25             (BM25 sparse)

Usage:
    python scripts/embed_chunks.py "test sentence"
    python scripts/embed_chunks.py "query text" --query
"""

import argparse

from fastembed import TextEmbedding, SparseTextEmbedding

_DENSE_MODEL  = "BAAI/bge-small-en-v1.5"
_SPARSE_MODEL = "Qdrant/bm25"


class Embedder:
    """Lazy-loaded dense + sparse embedder."""

    def __init__(self, dense_model=_DENSE_MODEL, sparse_model=_SPARSE_MODEL):
        self._dense_model_name  = dense_model
        self._sparse_model_name = sparse_model
        self._dense  = None
        self._sparse = None

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

    def embed_dense(self, texts):
        """
        Embed a list of strings.
        Returns a list of numpy float32 arrays (each 384-dim).
        """
        model = self._get_dense()
        return list(model.embed(texts))

    def embed_sparse(self, texts):
        """
        Embed a list of strings with BM25 sparse model.
        Returns a list of SparseEmbedding objects (with .indices and .values).
        """
        model = self._get_sparse()
        return list(model.embed(texts))

    def embed_query_dense(self, query):
        """Single query dense embedding (uses query prefix if model supports it)."""
        model = self._get_dense()
        return list(model.query_embed([query]))[0]

    def embed_query_sparse(self, query):
        """Single query sparse embedding."""
        model = self._get_sparse()
        return list(model.query_embed([query]))[0]


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embed a text string and print vector info.")
    parser.add_argument("text", help="Text to embed")
    parser.add_argument("--query", action="store_true", help="Use query embedding mode")
    args = parser.parse_args()

    embedder = Embedder()
    if args.query:
        dv = embedder.embed_query_dense(args.text)
        sv = embedder.embed_query_sparse(args.text)
    else:
        dv = embedder.embed_dense([args.text])[0]
        sv = embedder.embed_sparse([args.text])[0]

    print(f"Dense  : shape={dv.shape}  norm={float((dv**2).sum()**0.5):.4f}")
    print(f"Sparse : nnz={len(sv.indices)}  top-5 indices={sv.indices[:5].tolist()}")


if __name__ == "__main__":
    main()
