# Retrieval Eval — Run 1

**Date:** 2026-08-02
**Engagement:** acme-solar (12 chunks, 3 documents)
**Model:** all-MiniLM-L6-v2 · **Chunking:** 500 chars / 100 overlap
**Retrieval path:** Postgres pgvector, HNSW index, cosine

## Prediction (recorded before running)
- recall@5: 8/10
- Expected failure: Q7 (single valid chunk + topic dilution)
- Expected low confidence: Q2 (auto-close, previously 0.337)

## Result
| Metric | Random baseline | Measured |
|---|---|---|
| recall@1 | 24.2% | **80%** |
| recall@3 | ~58% | 100% |
| recall@5 | 78.6% | 100% |

Failures at k=1: Q6 (reverse logistics), Q7 (preventive maintenance).
Failures at k=5: none.

## What the prediction got right
Q7 failed at k=1 as expected — one valid chunk in the corpus, and that
chunk also carries escalation, TAT and reschedule content.

## What the prediction got wrong
Q2 scored 0.495, second highest, not low-confidence. Rewording the query
from "How are cases auto-closed?" (0.337 in an earlier run) to "When is a
case auto-closed if the customer gives no feedback?" raised the score 47%
against the same chunk.

**Revised understanding:** topic dilution is a mismatch between query
breadth and chunk breadth, not a chunk defect alone. A blended chunk is
matched well by a blended query. Query expansion is a cheaper lever than
re-chunking, and reversible.

## Caveats
- recall@5 on a 12-chunk corpus is near-meaningless: top-5 is 42% of the
  corpus and random selection scores 78.6%. recall@1 is the reportable metric.
- Ground truth authored by the system's builder. Acknowledged bias.
- Scores compress into 0.30–0.50. No threshold cutoff should be used —
  a 0.40 floor would discard five correct answers.
- n=1 run. Retrieval is deterministic, so repeat runs add nothing; but a
  larger corpus would change these numbers materially.

## Next
- Add corpus volume before trusting recall@5 again.
- Test query expansion as a retrieval improvement (cheap, reversible).
- Generation faithfulness is unmeasured — this eval covers retrieval only.