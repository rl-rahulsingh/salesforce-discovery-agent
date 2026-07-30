"""
eval_retrieval.py — retrieval accuracy evaluation harness.

Runs each test case in eval_set.py through the retriever, then measures:

1. recall@k for each k in {1, 3, 5} — did any expected chunk appear in top-k?
2. Per-query breakdown so you can see which queries failed
3. Overall pass/fail summary

Run:
    py eval_retrieval.py ./docs

Prereqs: build_index.py must have been run on ./docs first (index.pkl exists).
"""

import os
import pickle
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from eval_set import TEST_CASES

# k values to measure — will show whether the answer shows up in top-1,
# top-3, top-5. This lets you see how far down the ranked list your target
# chunk sits when retrieval is "kind of right" but not perfect.
K_VALUES = [1, 3, 5]


def load_index(workspace: str):
    index_path = os.path.join(workspace, "index.pkl")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    model = SentenceTransformer(index["model_name"])
    return model, index


def retrieve_top_k(model, index, query: str, k: int):
    """Return top-k chunks as list of (filename, chunk_index, score)."""
    query_vec = model.encode(query)
    chunk_vecs = np.array(index["vectors"])
    q_norm = query_vec / np.linalg.norm(query_vec)
    c_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    scores = c_norms @ q_norm
    top_indices = np.argsort(scores)[-k:][::-1]
    results = []
    for i in top_indices:
        chunk = index["chunks"][int(i)]
        results.append((chunk["filename"], chunk["chunk_index"], float(scores[i])))
    return results


def is_hit(retrieved, expected):
    """
    True if any expected (filename, chunk_index) appears in retrieved list.
    This is 'recall' at the chunk level: did we find AT LEAST ONE right chunk?
    """
    retrieved_ids = {(f, ci) for f, ci, _ in retrieved}
    expected_ids = set(expected)
    return len(retrieved_ids & expected_ids) > 0


def run_eval(workspace: str):
    print(f"Loading index from {workspace}...")
    model, index = load_index(workspace)
    max_k = max(K_VALUES)

    # Per-query results: dict of query -> {k: True/False}
    per_query = []
    # Aggregate hits: dict of k -> hit count
    hits_at_k = {k: 0 for k in K_VALUES}

    print(f"\nRunning {len(TEST_CASES)} test cases...\n")
    print(f"{'#':>3} | {'k=1':>5} | {'k=3':>5} | {'k=5':>5} | Query")
    print("-" * 90)

    for i, (query, expected) in enumerate(TEST_CASES, start=1):
        retrieved = retrieve_top_k(model, index, query, max_k)

        # For each k, check if any expected chunk appears in top-k
        query_result = {}
        for k in K_VALUES:
            top_k = retrieved[:k]
            hit = is_hit(top_k, expected)
            query_result[k] = hit
            if hit:
                hits_at_k[k] += 1

        # One-line row per query
        row = f"{i:>3} | " + " | ".join(
            f"{'  ✓  ' if query_result[k] else '  ✗  '}" for k in K_VALUES
        )
        print(f"{row} | {query}")

        per_query.append({
            "query": query,
            "expected": expected,
            "retrieved": retrieved,
            "hits": query_result,
        })

    # Summary
    n = len(TEST_CASES)
    print("\n" + "=" * 90)
    print("SUMMARY — recall@k across all test cases")
    print("=" * 90)
    for k in K_VALUES:
        pct = 100 * hits_at_k[k] / n
        print(f"  recall@{k}: {hits_at_k[k]:>2}/{n} = {pct:>5.1f}%")

    # Detail on failures — this is the actionable output.
    # A failure at k=5 means top 5 chunks didn't contain the answer.
    print("\n" + "=" * 90)
    print("FAILURES at k=5 — queries where even top-5 missed the target")
    print("=" * 90)
    any_failures = False
    for entry in per_query:
        if not entry["hits"][max_k]:
            any_failures = True
            print(f"\n  Query: \"{entry['query']}\"")
            print(f"  Expected any of: {entry['expected']}")
            print(f"  Got (top {max_k}):")
            for fname, ci, score in entry["retrieved"]:
                print(f"    - {fname} chunk {ci}  (score={score:.3f})")
    if not any_failures:
        print("  None — every query retrieved at least one expected chunk in top-5.")

    return per_query, hits_at_k


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py eval_retrieval.py <workspace_folder>")
        sys.exit(1)
    run_eval(sys.argv[1])
