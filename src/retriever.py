"""
retriever.py — Dense and HyDE retrieval over the ChromaDB 'eu-ai-act' collection.

Two retrieval strategies, identical return interface:

    list[{"text": str, "source": str, "page": int | str, "score": float}]

Classes:
    DenseRetriever  — embeds the raw query and searches ChromaDB directly.
    HyDERetriever   — asks the LLM for a hypothetical answer, embeds *that*,
                      then searches ChromaDB (Hypothetical Document Embeddings).

Usage:
    from src.retriever import DenseRetriever, HyDERetriever

    dense = DenseRetriever(k=4)
    hyde  = HyDERetriever(k=4)

    results = dense.retrieve("What are obligations for high-risk AI systems?")
    results = hyde.retrieve("What are obligations for high-risk AI systems?")

    # Both return the same shape:
    # [{"text": "...", "source": "eu_ai_act.pdf", "page": 42, "score": 0.87}, ...]
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Collection
from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_DIR    = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
EMBED_MODEL   = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION    = "eu-ai-act"
GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Shared result type alias (for documentation clarity)
ResultList = list[dict[str, Any]]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_embedder(model_name: str = EMBED_MODEL) -> HuggingFaceEmbeddings:
    """Load the MiniLM sentence-transformer (CPU, L2-normalised)."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_collection(
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION,
) -> Collection:
    """Open the persisted ChromaDB client and return the named collection."""
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at '{chroma_dir}'. "
            "Run `python src/ingest.py` first to build the index."
        )
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(name=collection_name)


def _query_chroma(
    collection: Collection,
    query_vector: list[float],
    k: int,
) -> ResultList:
    """
    Run a nearest-neighbour search and return a normalised result list.

    ChromaDB returns squared L2 distances. Because our vectors are
    L2-normalised, cosine similarity = 1 − distance / 2  ∈ [0, 1].
    """
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    texts     = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    results: ResultList = [
        {
            "text":   text,
            "source": meta.get("source_file", meta.get("source", "unknown")),
            "page":   meta.get("page", "?"),
            "score":  round(1.0 - dist / 2.0, 6),
        }
        for text, meta, dist in zip(texts, metadatas, distances)
    ]

    # Highest similarity first (ChromaDB already orders by distance,
    # but explicit sort keeps behaviour correct after score conversion).
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Shared initialisation for both retrieval strategies.
    Subclasses only need to implement retrieve().
    """

    def __init__(
        self,
        k: int = 4,
        collection: str = COLLECTION,
        chroma_dir: Path = CHROMA_DIR,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        self.k           = k
        self._embedder   = _load_embedder(embed_model)
        self._collection = _load_collection(chroma_dir, collection)

    def _embed(self, text: str) -> list[float]:
        """Embed a single string with the shared MiniLM model."""
        return self._embedder.embed_query(text)

    @abstractmethod
    def retrieve(self, query: str, k: int | None = None) -> ResultList:
        """Return top-k results as list[{text, source, page, score}]."""

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"k={self.k}, vectors={self._collection.count()})"
        )


# ── Strategy 1: Dense (baseline) ──────────────────────────────────────────────

class DenseRetriever(BaseRetriever):
    """
    Baseline dense retriever.

    Embeds the raw query string and performs cosine-similarity search directly
    against the ChromaDB index.
    """

    def retrieve(self, query: str, k: int | None = None) -> ResultList:
        """
        Embed *query* and return the top-k most similar chunks.

        Args:
            query : natural-language question
            k     : number of results (overrides instance default)

        Returns:
            list[{"text", "source", "page", "score"}] sorted by descending score
        """
        top_k      = k if k is not None else self.k
        query_vec  = self._embed(query)
        return _query_chroma(self._collection, query_vec, top_k)


# ── Strategy 2: HyDE (Hypothetical Document Embeddings) ───────────────────────

_HYDE_PROMPT = """\
You are an expert on EU AI regulation.
Write a concise, factual answer (≈100 words) to the following question as if \
you were answering from the EU AI Act text itself.
Do NOT use phrases like "the EU AI Act says" — just state the facts directly.

Question: {query}

Hypothetical answer:"""


class HyDERetriever(BaseRetriever):
    """
    HyDE retriever (Gao et al., 2022).

    Instead of embedding the raw query, we:
      1. Ask the LLM to generate a plausible 100-word answer.
      2. Embed *that hypothetical answer* with MiniLM.
      3. Search ChromaDB.
    """

    def __init__(
        self,
        k: int = 4,
        collection: str = COLLECTION,
        chroma_dir: Path = CHROMA_DIR,
        embed_model: str = EMBED_MODEL,
        gemini_model: str = GEMINI_MODEL,
        temperature: float = 0.3,
        api_key: str | None = None,
    ) -> None:
        super().__init__(k=k, collection=collection,
                         chroma_dir=chroma_dir, embed_model=embed_model)

        resolved_key = api_key or GEMINI_API_KEY
        if not resolved_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")

        # New SDK Client
        self._client = genai.Client(api_key=resolved_key)
        self._model_name = gemini_model
        self._temperature = temperature

    def _generate_hypothesis(self, query: str) -> str:
        """Ask Gemini to produce a hypothetical answer for the query."""
        prompt = _HYDE_PROMPT.format(query=query)
        
        # New SDK Generation call
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config={"temperature": self._temperature, "max_output_tokens": 200}
        )
        return response.text.strip()

    def retrieve(self, query: str, k: int | None = None) -> ResultList:
        """
        Generate a hypothetical answer, embed it, and return top-k chunks.

        Args:
            query : natural-language question
            k     : number of results (overrides instance default)

        Returns:
            list[{"text", "source", "page", "score"}] sorted by descending score
        """
        top_k      = k if k is not None else self.k
        hypothesis = self._generate_hypothesis(query)
        hypo_vec   = self._embed(hypothesis)

        results = _query_chroma(self._collection, hypo_vec, top_k)

        # Attach the hypothesis to every result for transparency / evaluation
        for r in results:
            r["hypothesis"] = hypothesis

        return results


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    QUERY = "What are the obligations for high-risk AI systems?"

    print("=" * 70)
    print(f"Query: {QUERY}")
    print("=" * 70)

    for RetrieverCls in (DenseRetriever, HyDERetriever):
        retriever = RetrieverCls(k=3)
        print(f"\n── {retriever.__class__.__name__} ──")
        results = retriever.retrieve(QUERY)
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] score={r['score']:.4f}  "
                  f"source={r['source']}  page={r['page']}")
            print(f"       {r['text'][:180]}…")
            if "hypothesis" in r:
                print(f"\n  [HyDE hypothesis]\n  {r['hypothesis'][:300]}…")
