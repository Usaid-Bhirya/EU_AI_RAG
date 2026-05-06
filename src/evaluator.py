"""
evaluator.py — Benchmark DenseRetriever vs HyDERetriever using RAGAS.

Metrics evaluated per strategy:
    - context_precision   : Are retrieved chunks relevant to the question?
    - faithfulness        : Is the answer grounded in the retrieved context?
    - answer_relevancy    : Does the answer actually address the question?
    - answer_correctness  : How correct is the answer vs. ground truth?

Also records mean end-to-end latency per question for each strategy.

Usage:
    # Run both retrievers (default)
    python src/evaluator.py

    # Run only one strategy
    python src/evaluator.py --strategy dense
    python src/evaluator.py --strategy hyde

    # Use a custom questions file
    python src/evaluator.py --questions data/eval_questions.json

Results are saved to data/results.json.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    faithfulness,
)

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
QUESTIONS_FILE   = Path("data/eval_questions.json")
RESULTS_FILE     = Path("data/results.json")
RETRIEVAL_K      = 5
# Delay between questions to respect API rate limits (15 req/min → 1 per 4s,
# but we use 10s to leave headroom for RAGAS scoring calls too).
API_DELAY_SECONDS = 10


# ── RAGAS evaluator LLM & embeddings setup ─────────────────────────────────────

def build_ragas_llm() -> LangchainLLMWrapper:
    """Wrap the Gemini chat model for RAGAS metric evaluation."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Wrap the same embedding model used by the retriever."""
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_single(
    question: str,
    ground_truth: str,
    retriever,
    generator,
) -> dict:
    """
    Run the full RAG pipeline for one question and return a result dict.

    Returns:
        {
            "question": str,
            "ground_truth": str,
            "answer": str,
            "contexts": list[str],   # list of retrieved chunk texts
            "latency_seconds": float,
        }
    """
    t0 = time.perf_counter()

    chunks  = retriever.retrieve(question, k=RETRIEVAL_K)
    output  = generator.generate(query=question, chunks=chunks)

    latency = time.perf_counter() - t0

    return {
        "question":        question,
        "ground_truth":    ground_truth,
        "answer":          output.answer,
        "contexts":        [c["text"] for c in chunks],
        "latency_seconds": round(latency, 3),
    }


# ── Strategy evaluator ─────────────────────────────────────────────────────────

def evaluate_strategy(
    strategy_name: str,
    retriever,
    generator,
    questions: list[dict],
    ragas_llm,
    ragas_embeddings,
) -> dict:
    """
    Run the full pipeline on all questions, compute RAGAS metrics, and
    return a structured results dict for one retrieval strategy.
    """
    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {strategy_name.upper()}")
    print(f"{'=' * 60}")

    per_question_results = []

    for i, item in enumerate(questions, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i:02d}/{len(questions)}] {q[:75]}...")

        try:
            result = run_single(q, gt, retriever, generator)
            per_question_results.append(result)
            print(f"         latency={result['latency_seconds']:.2f}s | "
                  f"chunks={len(result['contexts'])}")
        except Exception as e:
            print(f"         ERROR: {e}")
            # Keep the question in the dataset with empty answer so RAGAS
            # still receives a complete row and can score what it can.
            per_question_results.append({
                "question":        q,
                "ground_truth":    gt,
                "answer":          "",
                "contexts":        [],
                "latency_seconds": 0.0,
            })

        # Rate-limit throttle — skip the delay after the last question
        if i < len(questions):
            print(f"         Waiting {API_DELAY_SECONDS}s before next query (rate limit)...")
            time.sleep(API_DELAY_SECONDS)

    # ── Build RAGAS dataset (HF Dataset format for 0.2.x) ──────────────────────
    ragas_dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in per_question_results],
        "answer":       [r["answer"]       for r in per_question_results],
        "contexts":     [r["contexts"]     for r in per_question_results],
        "ground_truth": [r["ground_truth"] for r in per_question_results],
    })

    # ── Configure singleton metrics with our LLM & embeddings ──────────────────
    metrics = [context_precision, faithfulness, answer_relevancy, answer_correctness]
    for metric in metrics:
        metric.llm        = ragas_llm
        metric.embeddings = ragas_embeddings

    print(f"\n  Running RAGAS evaluation (this calls the LLM for scoring) ...")
    # Use a generous timeout — HyDE scoring is slow due to prior API calls
    run_config = RunConfig(timeout=180, max_workers=1, max_wait=300)
    ragas_result = evaluate(ragas_dataset, metrics=metrics, run_config=run_config)
    scores       = ragas_result.to_pandas().mean(numeric_only=True).to_dict()

    # ── Latency stats ──────────────────────────────────────────────────────────
    latencies         = [r["latency_seconds"] for r in per_question_results]
    mean_latency      = round(sum(latencies) / len(latencies), 3)

    print(f"\n  Results for {strategy_name.upper()}:")
    print(f"    context_precision  : {scores.get('context_precision', 'N/A'):.4f}")
    print(f"    faithfulness       : {scores.get('faithfulness', 'N/A'):.4f}")
    print(f"    answer_relevancy   : {scores.get('answer_relevancy', 'N/A'):.4f}")
    print(f"    answer_correctness : {scores.get('answer_correctness', 'N/A'):.4f}")
    print(f"    mean_latency       : {mean_latency:.3f}s")

    return {
        "metrics": {
            "context_precision":  round(scores.get("context_precision",  0.0), 4),
            "faithfulness":       round(scores.get("faithfulness",        0.0), 4),
            "answer_relevancy":   round(scores.get("answer_relevancy",    0.0), 4),
            "answer_correctness": round(scores.get("answer_correctness",  0.0), 4),
        },
        "mean_latency_seconds": mean_latency,
        "per_question": per_question_results,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Dense vs HyDE retrieval strategies with RAGAS."
    )
    parser.add_argument(
        "--strategy",
        choices=["dense", "hyde", "both"],
        default="both",
        help="Which retrieval strategy to evaluate (default: both).",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(QUESTIONS_FILE),
        help=f"Path to the JSON evaluation questions file (default: {QUESTIONS_FILE}).",
    )
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is not set in .env")

    # ── Load questions ─────────────────────────────────────────────────────────
    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(questions_path) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from '{questions_path}'.")

    # ── Lazy imports to avoid loading the models unless needed ─────────────────
    from generator import RAGGenerator
    from retriever import DenseRetriever, HyDERetriever

    generator        = RAGGenerator()
    ragas_llm        = build_ragas_llm()
    ragas_embeddings = build_ragas_embeddings()

    results = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "questions_file":  str(questions_path),
        "num_questions":   len(questions),
        "retrieval_k":     RETRIEVAL_K,
        "gemini_model":    GEMINI_MODEL,
    }

    # ── Run evaluations ────────────────────────────────────────────────────────
    if args.strategy in ("dense", "both"):
        dense_retriever      = DenseRetriever(k=RETRIEVAL_K)
        results["dense"]     = evaluate_strategy(
            "dense", dense_retriever, generator,
            questions, ragas_llm, ragas_embeddings,
        )

    if args.strategy in ("hyde", "both"):
        hyde_retriever       = HyDERetriever(k=RETRIEVAL_K)
        results["hyde"]      = evaluate_strategy(
            "hyde", hyde_retriever, generator,
            questions, ragas_llm, ragas_embeddings,
        )

    # ── Save results ───────────────────────────────────────────────────────────
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to '{RESULTS_FILE}'.")

    # ── Side-by-side summary (if both ran) ─────────────────────────────────────
    if "dense" in results and "hyde" in results:
        print(f"\n{'─' * 55}")
        print(f"  {'Metric':<26} {'Dense':>10} {'HyDE':>10}")
        print(f"{'─' * 55}")
        for metric in ["context_precision", "faithfulness",
                       "answer_relevancy", "answer_correctness"]:
            d = results["dense"]["metrics"][metric]
            h = results["hyde"]["metrics"][metric]
            print(f"  {metric:<26} {d:>10.4f} {h:>10.4f}")
        print(f"  {'mean_latency (s)':<26} "
              f"{results['dense']['mean_latency_seconds']:>10.3f} "
              f"{results['hyde']['mean_latency_seconds']:>10.3f}")
        print(f"{'─' * 55}")


if __name__ == "__main__":
    main()
