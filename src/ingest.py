"""
ingest.py — Load, chunk, embed, and persist EU AI Act PDFs into ChromaDB.

Usage:
    # First-time build
    python src/ingest.py

    # Wipe the existing collection and rebuild from scratch
    python src/ingest.py --reset
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_DIR        = Path(os.getenv("RAW_DIR", "data/raw"))
CHROMA_DIR     = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
EMBED_MODEL    = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION     = "eu-ai-act"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200


# ── Helpers ────────────────────────────────────────────────────────────────────

def reset_collection() -> None:
    """Delete the persisted ChromaDB directory so the collection is rebuilt fresh."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"[ingest] 🗑  Wiped existing ChromaDB at '{CHROMA_DIR}'.")
    else:
        print("[ingest] No existing ChromaDB found — nothing to wipe.")


def load_pdfs(raw_dir: Path) -> list:
    """
    Discover and load every PDF in raw_dir.
    Returns a flat list of LangChain Document objects (one per page).
    """
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[ingest] ⚠  No PDFs found in '{raw_dir}'. "
              "Download the EU AI Act PDF and place it there.")
        sys.exit(1)

    all_docs = []
    for pdf in pdf_files:
        print(f"[ingest] 📄 Loading '{pdf.name}' …")
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        # Stamp the source filename onto each page for traceability
        for doc in docs:
            doc.metadata["source_file"] = pdf.name
        all_docs.extend(docs)
        print(f"           └─ {len(docs)} pages loaded.")

    print(f"[ingest] Total pages across all PDFs: {len(all_docs)}")
    return all_docs


def chunk_documents(docs: list) -> list:
    """
    Split pages into overlapping chunks using RecursiveCharacterTextSplitter.
    Respects natural text boundaries (paragraphs → sentences → words).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] ✂  Split into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the all-MiniLM-L6-v2 sentence-transformer model.
    Downloads ~90 MB on first run, then cached in ~/.cache/huggingface/.
    """
    print(f"[ingest] 🤖 Loading embedding model: {EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embed all chunks and persist them to ChromaDB.
    Uses batch insertion to avoid memory spikes on large corpora.
    """
    print(f"[ingest] 💾 Embedding & storing {len(chunks)} chunks "
          f"→ collection '{COLLECTION}' at '{CHROMA_DIR}' …")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )

    count = vectorstore._collection.count()
    print(f"[ingest] ✅ Done — {count} vectors stored in ChromaDB.")
    return vectorstore


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest EU AI Act PDFs into a persistent ChromaDB collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing ChromaDB collection and rebuild it from scratch.",
    )
    args = parser.parse_args()

    if args.reset:
        reset_collection()

    docs   = load_pdfs(RAW_DIR)
    chunks = chunk_documents(docs)
    embeds = get_embeddings()
    build_vectorstore(chunks, embeds)


if __name__ == "__main__":
    main()
