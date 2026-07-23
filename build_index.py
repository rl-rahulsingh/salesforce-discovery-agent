"""
build_index.py — RUN ONCE (or whenever documents change).

What it does:
    1. Reads every .txt / .md file from the workspace folder.
    2. Chunks each file into overlapping windows (~500 chars, 100 char overlap).
    3. Runs each chunk through a local sentence-transformer to get an embedding vector.
    4. Saves everything (chunk text + metadata + vectors) to index.pkl on disk.

Run:
    py build_index.py ./docs

After it finishes, index.pkl sits in the workspace folder.
retrieve.py and the RAG-extended agent will read it at query time.
"""

import os
import pickle
import sys

from sentence_transformers import SentenceTransformer

# ---- Chunking parameters ----
# These are the design commitments Rahul was warned about.
CHUNK_SIZE = 500       # characters per chunk (~100 tokens rough estimate)
CHUNK_OVERLAP = 100    # characters of overlap between adjacent chunks (20%)

# ---- Embedding model ----
# "all-MiniLM-L6-v2" is small (~90MB), fast on CPU, and produces 384-dim vectors.
# Downloaded once, cached locally. Free.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Slide a fixed-size window across the text with overlap.
    Simplest version of recursive character chunking — no natural-boundary
    detection yet. That's a deliberate scope call: get the pipeline working
    end-to-end first, then upgrade the chunker without touching the rest.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap  # forward stride each iteration

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    return chunks


def build_index(workspace: str) -> None:
    """
    Walks the workspace, chunks every text file, embeds every chunk,
    saves the index. Prints progress so you can see what's happening.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    all_chunks = []          # list of dicts: {text, filename, chunk_index}
    all_vectors = []         # parallel list of embedding vectors

    files = [f for f in os.listdir(workspace) if f.endswith((".txt", ".md"))]
    # Skip the index file itself if it happens to be a .md/.txt (it won't be, but be safe)
    files = [f for f in files if f != "index.pkl"]

    print(f"Found {len(files)} document(s) in {workspace}")

    for filename in files:
        path = os.path.join(workspace, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  {filename}: {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {"text": chunk, "filename": filename, "chunk_index": i}
            )

    # Batch-embed everything in one call — much faster than one-by-one.
    print(f"Embedding {len(all_chunks)} chunk(s)...")
    all_vectors = model.encode([c["text"] for c in all_chunks], show_progress_bar=True)

    index = {
        "chunks": all_chunks,
        "vectors": all_vectors,
        "model_name": EMBEDDING_MODEL_NAME,  # so we know which model to use at query time
    }

    index_path = os.path.join(workspace, "index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    print(f"Wrote index: {index_path}")
    print(f"  {len(all_chunks)} chunks, {all_vectors.shape[1]}-dim vectors")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py build_index.py <workspace_folder>")
        sys.exit(1)
    build_index(sys.argv[1])
