"""
eval_retrieval.py — measure retrieval accuracy against Postgres.

Runs every case in eval_set.py through match_chunks and reports
recall@k. This tests the SAME path the agent uses, including the HNSW
index — an eval against brute-force similarity would measure something
you do not ship.

Run:  py eval_retrieval.py [engagement-slug]
"""

import sys
from v3_agent import _get_db, _get_model, resolve_engagement, verify_embedding_model
from eval_set import TEST_CASES

K_VALUES = [1, 3, 5]
MAX_K = max(K_VALUES)


def retrieve(sb, model, engagement_id, query, k):
    """Return [(filename, chunk_index, similarity)] for the top-k."""
    vec = model.encode(query).tolist()
    res = sb.rpc("match_chunks", {
        "query_embedding":   vec,
        "target_engagement": engagement_id,
        "match_count":       k,
    }).execute()
    return [(r["chunk_file"], r["chunk_no"], r["similarity"]) for r in res.data]


def main(slug: str):
    sb = _get_db()
    engagement_id = resolve_engagement(slug)
    verify_embedding_model(engagement_id)   # fail fast on model drift
    model = _get_model()

    per_query = []
    hits = {k: 0 for k in K_VALUES}

    print(f"\nEngagement: {slug}   Cases: {len(TEST_CASES)}\n")
    print(f"{'#':>3} | {'k=1':^5} | {'k=3':^5} | {'k=5':^5} | {'top':>5} | Query")
    print("-" * 100)

    for i, (query, expected) in enumerate(TEST_CASES, start=1):
        results = retrieve(sb, model, engagement_id, query, MAX_K)
        expected_set = set(expected)

        row = {}
        for k in K_VALUES:
            found = {(f, c) for f, c, _ in results[:k]}
            hit = bool(found & expected_set)
            row[k] = hit
            if hit:
                hits[k] += 1

        marks = " | ".join("  ✓  " if row[k] else "  ✗  " for k in K_VALUES)
        top_score = results[0][2] if results else 0.0
        print(f"{i:>3} | {marks} | {top_score:>5.3f} | {query}")

        per_query.append({"query": query, "expected": expected,
                          "results": results, "hits": row})

    n = len(TEST_CASES)
    print("\n" + "=" * 100)
    print("RECALL@K")
    print("=" * 100)
    for k in K_VALUES:
        print(f"  recall@{k}: {hits[k]:>2}/{n} = {100*hits[k]/n:>5.1f}%")

    print("\n" + "=" * 100)
    print(f"FAILURES AT k={MAX_K} — the actionable output")
    print("=" * 100)
    any_fail = False
    for e in per_query:
        if not e["hits"][MAX_K]:
            any_fail = True
            print(f"\n  Query    : \"{e['query']}\"")
            print(f"  Expected : {e['expected']}")
            print(f"  Returned :")
            for f, c, s in e["results"]:
                print(f"      {s:>6.3f}  {f} #{c}")
    if not any_fail:
        print("  None. Every query recovered a correct chunk within the top 5.")

    # Low-confidence passes: correct, but only just. These are the ones a
    # similarity threshold would silently discard.
    print("\n" + "=" * 100)
    print("LOW-CONFIDENCE PASSES — correct chunk found, top score below 0.40")
    print("=" * 100)
    weak = [e for e in per_query if e["hits"][MAX_K] and e["results"][0][2] < 0.40]
    if not weak:
        print("  None.")
    for e in weak:
        print(f"  {e['results'][0][2]:.3f}  \"{e['query']}\"")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "acme-solar")