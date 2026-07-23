"""
retrieve.py — standalone retrieval tester. NOT called by the agent yet.

Purpose: prove the index works before we wire it into the agent.
    py retrieve.py ./docs "what are the SLA requirements?"

Prints the top-k chunks ranked by cosine similarity to the query.
Read the output. Sanity-check: are the returned chunks actually about SLA?
If they're not, the chunking or embedding is bad and no agent will save you.
"""

import os
import pickle
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

TOP_K = 3  # how many chunks to return per query


def cosine_similarity(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    """
    Vectorized cosine similarity: query_vec against every row in chunk_vecs.
    Cosine similarity = dot product of unit vectors.
    Higher = more similar. Range [-1, 1]; in practice for embeddings it's ~[0, 1].
    """
    # Normalize to unit vectors so dot product == cosine similarity
    q_norm = query_vec / np.linalg.norm(query_vec)
    c_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    return c_norms @ q_norm  # matrix-vector product returns one score per chunk


def retrieve(workspace: str, query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Load index, embed the query with the SAME model used at index time,
    score every chunk, return the top-k with scores + metadata.
    """
    index_path = os.path.join(workspace, "index.pkl")
    with open(index_path, "rb") as f:
        index = pickle.load(f)

    # SAME MODEL AS INDEXING — this is the "same coordinate system" invariant.
    # If this line doesn't match build_index.py, similarity scores are meaningless.
    model = SentenceTransformer(index["model_name"])
    query_vec = model.encode(query)

    chunk_vecs = np.array(index["vectors"])
    scores = cosine_similarity(query_vec, chunk_vecs)

    # argsort returns ascending; we want the last top_k (highest scores) reversed
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in top_indices:
        chunk = index["chunks"][i]
        results.append(
            {
                "score": float(scores[i]),
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }
        )
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: py retrieve.py <workspace_folder> "<query>"')
        sys.exit(1)

    workspace = sys.argv[1]
    query = sys.argv[2]

    hits = retrieve(workspace, query)
    print(f'\nQuery: "{query}"\n')
    print(f"Top {len(hits)} chunks:\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"--- Rank {rank} | score={hit['score']:.3f} | "
              f"{hit['filename']} (chunk {hit['chunk_index']}) ---")
        print(hit["text"])
        print()
