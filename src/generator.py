"""
generator.py — RAG answer generation using the Gemini API.

RAGGenerator takes a query and a list of retrieved chunks (from DenseRetriever
or HyDERetriever), builds a grounded prompt with numbered source citations,
calls Gemini, and returns a structured response containing both the answer text
and a deduplicated list of cited sources.

Usage:
    from src.retriever import DenseRetriever
    from src.generator import RAGGenerator

    retriever = DenseRetriever(k=5)
    generator = RAGGenerator()

    chunks = retriever.retrieve("What defines a high-risk AI system?")
    output = generator.generate(query="What defines a high-risk AI system?",
                                chunks=chunks)

    print(output["answer"])
    print(output["sources"])
"""

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── Prompt template ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a precise legal assistant specialising in the EU AI Act.
Answer the user's question using ONLY the numbered context passages provided.
Rules:
  1. Cite every claim with its passage number in square brackets, e.g. [1], [2].
  2. If multiple passages support a claim, cite all of them, e.g. [1][3].
  3. If the context does not contain enough information, say:
     "The provided excerpts do not cover this topic sufficiently."
  4. Do NOT add knowledge from outside the provided passages.
  5. Be concise but complete — aim for 150-250 words.\
"""

_USER_TEMPLATE = """\
Context passages:
{context}

Question: {query}

Answer (with inline citations [n]):"""


# ── Output container ───────────────────────────────────────────────────────────

class GenerationOutput:
    """
    Structured output from RAGGenerator.generate().

    Attributes:
        answer   : The generated answer text with inline [n] citations.
        sources  : Deduplicated list of source dicts cited in the answer.
                   Each entry: {"index": int, "source": str, "page": int|str,
                                "score": float, "text_preview": str}
        chunks   : The full list of input chunks passed to the generator.
        query    : The original query string.
    """

    def __init__(
        self,
        answer: str,
        sources: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        query: str,
    ) -> None:
        self.answer  = answer
        self.sources = sources
        self.chunks  = chunks
        self.query   = query

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (useful for the evaluator and API layer)."""
        return {
            "query":   self.query,
            "answer":  self.answer,
            "sources": self.sources,
        }

    def __repr__(self) -> str:
        preview = self.answer[:80].replace("\n", " ")
        return (
            f"GenerationOutput(query='{self.query[:50]}…', "
            f"sources={len(self.sources)}, answer='{preview}…')"
        )


# ── RAGGenerator ───────────────────────────────────────────────────────────────

class RAGGenerator:
    """
    Formats retrieved chunks into a grounded Gemini prompt and returns a
    structured answer with source attribution.

    Args:
        model       : Gemini model name (default: from GEMINI_MODEL env var)
        temperature : Sampling temperature — lower = more deterministic (default 0.2)
        max_tokens  : Maximum output tokens (default 1024)
    """

    def __init__(
        self,
        model: str = GEMINI_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or GEMINI_API_KEY
        if not resolved_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        self.client = genai.Client(api_key=resolved_key)

        self.model_name = model

        self.config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]]) -> str:
        """
        Format the retrieved chunks into a numbered context block.

        Each passage is numbered [1], [2], … so the LLM can cite them inline.
        Source and page are included so the model can reference provenance.
        """
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            source  = chunk.get("source", "unknown")
            page    = chunk.get("page", "?")
            text    = chunk.get("text", "").strip()
            lines.append(
                f"[{i}] (source: {source}, page {page})\n{text}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _extract_cited_sources(
        answer: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Parse [n] citation markers in the answer and return the corresponding
        chunk metadata deduplicated and sorted by citation index.
        """
        import re
        cited_indices = sorted(set(
            int(m) for m in re.findall(r"\[(\d+)\]", answer)
            if 1 <= int(m) <= len(chunks)
        ))

        sources = []
        for idx in cited_indices:
            chunk = chunks[idx - 1]          # convert 1-based → 0-based
            sources.append({
                "index":        idx,
                "source":       chunk.get("source", "unknown"),
                "page":         chunk.get("page", "?"),
                "score":        chunk.get("score", 0.0),
                "text_preview": chunk.get("text", "")[:120].replace("\n", " "),
            })

        # If the model cited nothing (it happens), fall back to the top chunk
        if not sources and chunks:
            top = chunks[0]
            sources.append({
                "index":        1,
                "source":       top.get("source", "unknown"),
                "page":         top.get("page", "?"),
                "score":        top.get("score", 0.0),
                "text_preview": top.get("text", "")[:120].replace("\n", " "),
            })

        return sources

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> GenerationOutput:
        """
        Generate a grounded answer from retrieved chunks.

        Args:
            query  : The original user question.
            chunks : list[{"text", "source", "page", "score"}] from a retriever.

        Returns:
            GenerationOutput with .answer, .sources, .chunks, .query
        """
        if not chunks:
            raise ValueError("chunks must not be empty — run the retriever first.")

        context = self._build_context(chunks)
        prompt  = _USER_TEMPLATE.format(context=context, query=query)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.config
        )
        answer   = response.text.strip()

        sources = self._extract_cited_sources(answer, chunks)

        return GenerationOutput(
            answer=answer,
            sources=sources,
            chunks=chunks,
            query=query,
        )

    def __repr__(self) -> str:
        return f"RAGGenerator(model='{GEMINI_MODEL}', temperature=0.2)"


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from retriever import DenseRetriever, HyDERetriever

    parser = argparse.ArgumentParser(description="Run the RAG generator.")
    parser.add_argument(
        "--query", 
        type=str, 
        default="What are the obligations for providers of high-risk AI systems?",
        help="The question to ask."
    )
    parser.add_argument(
        "--retriever", 
        choices=["dense", "hyde"], 
        default="dense",
        help="The retrieval strategy to use."
    )
    args = parser.parse_args()

    # Select and initialize the retriever
    if args.retriever == "hyde":
        print(f"Retrieving chunks using HyDE strategy for: {args.query}")
        retriever = HyDERetriever(k=5)
    else:
        print(f"Retrieving chunks using Dense strategy for: {args.query}")
        retriever = DenseRetriever(k=5)

    chunks = retriever.retrieve(args.query)

    print("Generating answer ...\n")
    generator = RAGGenerator()
    output    = generator.generate(query=args.query, chunks=chunks)

    print("=" * 70)
    print(f"Query:  {output.query}\n")
    print(f"Answer:\n{output.answer}\n")
    print("Sources cited:")
    for s in output.sources:
        print(f"  [{s['index']}] {s['source']} — page {s['page']} "
              f"(score={s['score']:.4f})")
        print(f"       \"{s['text_preview']}...\"")
