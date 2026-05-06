"""
main.py — Combined FastAPI + Gradio entry point for HF Spaces deployment.

FastAPI serves:
    GET  /api/health
    POST /api/query

Gradio is mounted at / (root) and calls the pipeline directly (no HTTP round-trip).

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860 --reload

Deploy: push to a HF Space with sdk: docker (see README.md and Dockerfile).
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import gradio as gr
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent / "src"))
from generator import RAGGenerator
from retriever import DenseRetriever, HyDERetriever

load_dotenv()

RETRIEVAL_K = 5

# ── Shared pipeline ────────────────────────────────────────────────────────────
# DenseRetriever uses only the local embedding model — no API key needed.
# It is loaded once at startup and shared by all users.
_dense: DenseRetriever | None = None

# HyDERetriever and RAGGenerator call Gemini, so they are created per user key
# and cached so the same key doesn’t trigger a re-initialisation on every request.
_gemini_cache: dict[str, dict] = {}


def _load_pipeline() -> None:
    global _dense
    print("[main] Loading embedding model and vector store …")
    _dense = DenseRetriever(k=RETRIEVAL_K)
    print("[main] ✅ Embedding pipeline ready.")


def _get_gemini_components(api_key: str) -> dict:
    """Return (and cache) HyDERetriever + RAGGenerator for the given API key."""
    if api_key not in _gemini_cache:
        _gemini_cache[api_key] = {
            "hyde": HyDERetriever(k=RETRIEVAL_K, api_key=api_key),
            "gen":  RAGGenerator(api_key=api_key),
        }
    return _gemini_cache[api_key]


def _run(query: str, strategy: str, api_key: str, top_k: int = RETRIEVAL_K) -> dict:
    """Core logic shared by both the API endpoint and the Gradio callback."""
    components = _get_gemini_components(api_key)
    retriever  = _dense if strategy == "dense" else components["hyde"]
    gen        = components["gen"]
    t0         = time.perf_counter()
    chunks     = retriever.retrieve(query, k=top_k)
    output     = gen.generate(query=query, chunks=chunks)
    latency    = (time.perf_counter() - t0) * 1000
    return {"answer": output.answer, "chunks": chunks, "latency_ms": round(latency, 1)}


# ── FastAPI ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_pipeline()   # loads dense embeddings only; no API key required
    yield


app = FastAPI(
    title="EU AI Act — RAG API",
    description="Retrieval-Augmented Generation over the EU AI Act corpus.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:    str                      = Field(..., min_length=5, max_length=1000)
    strategy: Literal["dense", "hyde"] = Field(default="dense")
    top_k:    int                      = Field(default=5, ge=1, le=20)


class SourceOut(BaseModel):
    rank:    int
    source:  str
    page:    int | str
    score:   float
    preview: str


class QueryResponse(BaseModel):
    answer:     str
    sources:    list[SourceOut]
    strategy:   str
    latency_ms: float


class HealthResponse(BaseModel):
    status:   str
    pipeline: bool


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Ops"],
    summary="Liveness check — returns ok when models are loaded.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", pipeline=all([_dense, _hyde, _gen]))


@app.post(
    "/api/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Ask a question about the EU AI Act.",
    description=(
        "Retrieves top-k relevant chunks from ChromaDB using the chosen strategy "
        "(dense cosine search or HyDE), then calls Gemini to produce a grounded "
        "answer with inline [n] source citations."
    ),
)
async def query(req: QueryRequest) -> QueryResponse:
    if not all([_dense, _hyde, _gen]):
        raise HTTPException(503, "Pipeline still loading — retry in a few seconds.")
    try:
        result = _run(req.query, req.strategy, req.top_k)
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    sources = [
        SourceOut(
            rank=i,
            source=c.get("source", "unknown"),
            page=c.get("page", "?"),
            score=round(c.get("score", 0.0), 4),
            preview=c.get("text", "")[:300].replace("\n", " "),
        )
        for i, c in enumerate(result["chunks"], 1)
    ]
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        strategy=req.strategy,
        latency_ms=result["latency_ms"],
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────────
def gradio_query(query: str, strategy: str, api_key: str):
    """Generator — yields two states: loading then result."""
    api_key = api_key.strip()

    if not api_key:
        yield (
            gr.update(value="🔍  Ask the EU AI Act", interactive=True),
            "⚠️ **No API key provided.** Please enter your Gemini API key above.", "", "",
        )
        return

    if not query.strip():
        yield (
            gr.update(value="🔍  Ask the EU AI Act", interactive=True),
            "⚠️ Please enter a question.", "", "",
        )
        return

    # ── State 1: disable button and show spinner ───────────────────────────
    yield (
        gr.update(value="⏳  Generating answer…", interactive=False),
        "*Retrieving relevant passages and generating answer — please wait…*",
        "",
        "",
    )

    # ── Do the actual work ────────────────────────────────────────────────
    try:
        result = _run(query.strip(), strategy.lower(), api_key)
    except Exception as e:
        yield (
            gr.update(value="🔍  Ask the EU AI Act", interactive=True),
            f"❌ **Error:** {e}", "", "",
        )
        return

    chunks = result["chunks"]

    sources_md = "\n\n---\n\n".join(
        f"### [{i}] `{c.get('source','?')}` — page {c.get('page','?')} "
        f"_(score: {c.get('score',0):.3f})_\n\n"
        f"> {c.get('text','')[:300].replace(chr(10),' ')}..."
        for i, c in enumerate(chunks, 1)
    )
    latency_str = f"⏱ **{result['latency_ms']:,.0f} ms** end-to-end"

    # ── State 2: re-enable button and return results ──────────────────────
    yield (
        gr.update(value="🔍  Ask the EU AI Act", interactive=True),
        result["answer"],
        sources_md,
        latency_str,
    )


EXAMPLES = [
    "What are the obligations for providers of high-risk AI systems?",
    "What AI practices are completely prohibited under the EU AI Act?",
    "What are the penalties for non-compliance with the EU AI Act?",
    "What transparency obligations apply to general-purpose AI models?",
    "How does the EU AI Act define a high-risk AI system?",
]

HOW_IT_WORKS = """
## How This System Works

This is a **Retrieval-Augmented Generation (RAG)** pipeline over the EU AI Act corpus.

| Stage | What happens |
|---|---|
| **Ingest** | PDFs are chunked, embedded with `BAAI/bge-base-en-v1.5`, stored in ChromaDB |
| **Retrieve** | Your query (or a Gemini-generated hypothesis) is embedded and top-k chunks are fetched |
| **Generate** | Gemini answers using *only* the retrieved passages, citing every claim with [n] |

### Retrieval Strategies
- **Dense** — embeds your raw query → fast, deterministic  
- **HyDE** — Gemini writes a hypothetical answer first, embeds that → slower, often more accurate for complex legal questions

### Evaluation Results (BGE-base + Gemini 1.5 Flash)
| Metric | Dense | HyDE |
|---|---|---|
| Faithfulness | 1.0000 | 0.9947 |
| Context Precision | 0.2303 | 0.4470 |
| Answer Relevancy | 0.7186 | 0.8227 |
| Answer Correctness | 0.4003 | 0.4870 |
| Mean Latency | 18.14s | 37.76s |
"""

CSS = """
body, .gradio-container { background-color: #11111b !important; font-family: 'Inter', sans-serif !important; }
.app-header { background: linear-gradient(135deg,#1e1e2e,#2a2a3e); border:1px solid #3b3b52; border-radius:16px; padding:28px 32px; margin-bottom:8px; }
.app-header h1 { font-size:1.9rem !important; font-weight:700 !important; background:linear-gradient(90deg,#818cf8,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 0 4px 0 !important; }
.app-header p { color:#7f849c !important; margin:0 !important; }
textarea { background:#2a2a3e !important; border:1px solid #3b3b52 !important; color:#cdd6f4 !important; border-radius:10px !important; }
.submit-btn { background:linear-gradient(135deg,#6366f1,#818cf8) !important; border:none !important; color:white !important; font-weight:600 !important; border-radius:10px !important; }
"""

with gr.Blocks(title="EU AI Act — RAG Q&A") as demo:
    gr.HTML(f"<style>{CSS}</style>")
    gr.HTML("""
    <div class="app-header">
        <h1>⚖️ EU AI Act — Regulatory Q&amp;A</h1>
        <p>Ask any question about the EU AI Act. Enter your Gemini API key to get started.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            api_key_in = gr.Textbox(
                label="🔑 Gemini API Key",
                placeholder="Paste your Gemini API key here (AIza…)",
                type="password",
                info="Your key is used only for this session and never stored. Get one free at aistudio.google.com.",
            )
            query_in   = gr.Textbox(label="Your Question", lines=3,
                                    placeholder="e.g. What are the obligations for high-risk AI providers?")
            strategy   = gr.Radio(["Dense", "HyDE"], value="Dense", label="Retrieval Strategy",
                                  info="Dense: fast. HyDE: more accurate for complex questions.")
            submit_btn = gr.Button("🔍  Ask the EU AI Act", variant="primary", elem_classes=["submit-btn"])
            latency    = gr.Markdown("")
            gr.Examples([[e] for e in EXAMPLES], inputs=[query_in], label="Example Questions")

        with gr.Column(scale=3):
            answer_out  = gr.Markdown("*Your answer will appear here…*", label="Answer")
            with gr.Accordion("📄 Retrieved Source Chunks", open=False):
                sources_out = gr.Markdown("*Retrieved chunks will appear here…*")

    with gr.Accordion("ℹ️ How It Works", open=False):
        gr.Markdown(HOW_IT_WORKS)

    submit_btn.click(
        gradio_query,
        inputs=[query_in, strategy, api_key_in],
        outputs=[submit_btn, answer_out, sources_out, latency],
    )
    query_in.submit(
        gradio_query,
        inputs=[query_in, strategy, api_key_in],
        outputs=[submit_btn, answer_out, sources_out, latency],
    )


# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
