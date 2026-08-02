# Cost Analysis — Module 1 Discovery Agent

**Date:** 2026-08-02
**Model:** claude-sonnet-4-6 · $3/M input, $15/M output
**Corpus:** acme-solar — 3 documents, 12 chunks
**Task:** "Identify all conflicting requirements and open questions."

---

## Why measure this

Before instrumentation the agent "worked" — it produced findings, the trace
looked healthy, the UI rendered. Nothing indicated a problem. Measurement
revealed two things no functional test would have caught:

1. The run was **silently truncated** — it hit the 15-iteration cap mid-work
   and never wrote its summary.
2. One design decision (one tool call per finding) was the **dominant cost**
   in the entire run.

---

## Run 1 — baseline (one save_extraction call per finding)

| Iter | Input tok | Output tok | API s |
|---|---|---|---|
| 1 | 1,324 | 60 | 2.3 |
| 5 | 10,554 | 204 | 4.9 |
| 10 | 15,947 | 548 | 10.2 |
| 15 | 18,745 | 441 | 7.6 |

- Iterations: **15 — hit the safety cap, run incomplete**
- Input: 187,371 tokens → $0.5621 (87% of cost)
- Output: 5,690 tokens → $0.0853
- **Total: $0.6475** · 110.8s

---

## Run 2 — after batching (one save_extractions call with a list)

| Iter | Input tok | Output tok | API s |
|---|---|---|---|
| 1 | 1,341 | 60 | 2.5 |
| 4 | 8,989 | 194 | 3.8 |
| 5 | 11,691 | 5,122 | 76.2 |
| 7 | 19,781 | 1,960 | 30.8 |

- Iterations: **7 — completed normally**
- Input: 64,677 tokens → $0.1940
- Output: 11,252 tokens → $0.1688
- **Total: $0.3628** · 187.6s

---

## Delta

| Metric | Run 1 | Run 2 | Change |
|---|---|---|---|
| Iterations | 15 (capped) | 7 (finished) | −53% |
| Input tokens | 187,371 | 64,677 | **−65%** |
| Cost | $0.6475 | $0.3628 | **−44%** |
| Latency | 110.8s | 187.6s | +69% |
| Completed? | No | **Yes** | — |

More work done, for less money.

---

## Finding 1 — agent cost is quadratic in iteration count

The model is stateless. Every API call must resend the entire conversation
so far, because it has no memory between calls.

Run 1's final conversation was **18,745 tokens**. Billed input was
**187,371 tokens** — ten times the text that actually exists. The early
tokens were paid for 15 times, the middle ones 8 times, the last ones once.

**Consequence:** doubling iterations roughly quadruples cost — a longer
context, sent more times. Cost scales with the *square* of iteration count,
not linearly.

**Therefore the highest-leverage optimisation in any agent loop is doing
more per iteration**, not shortening prompts.

---

## Finding 2 — cost and latency have different drivers

| Driver | Governed by | Why |
|---|---|---|
| **Cost** | Input tokens | Full conversation resent every call; input was 87% of Run 1's bill |
| **Latency** | Output tokens | Generation is serial at a fixed rate |

Measured generation rate from Run 2:

| Output tokens | API seconds | tok/s |
|---|---|---|
| 5,122 | 76.2 | 67 |
| 3,423 | 57.5 | 60 |
| 1,960 | 30.8 | 63 |

**~65 tokens/second, consistently.** A run's duration is predictable from
its expected output length.

This explains Run 2's higher latency despite lower cost: batching
concentrated the same total output into fewer, longer generations — and
Run 2 actually finished, which means writing the report Run 1 never reached.

---

## Finding 3 — retrieval is not the bottleneck

Tool execution was 10% of Run 1 and 5% of Run 2. Postgres round trips to
Mumbai are a rounding error beside model inference.

**Consequence:** optimising vector index performance would be wasted effort.
To make runs faster, reduce iterations or use a faster model.

---

## Finding 4 — a clean design decision was the dominant cost

`save_extraction` taking one finding per call was better code: simple calls,
isolated errors, no large payloads to malform. It was also responsible for
roughly 10 of 15 iterations and the majority of the token spend.

**This is what observability buys** — not dashboards, but the ability to see
which design decision is expensive. Without measurement, one-call-per-item
looks like good engineering.

---

## Open items

- **Prompt caching not yet applied.** The system prompt and tool definitions
  (~1,300 tokens) are byte-identical on all 7 calls and paid for 7 times.
  Anthropic caches stable prefixes at ~10% of input price. Expected saving on
  an input-dominated workload: material.
- **n=1 per configuration.** Both runs are single samples. LLM output length
  varies between runs, so cost varies too. A cost *range* would need repeats.
- **Cost scales with corpus.** 12 chunks today. More documents means more
  searches, more retrieved text per call, and larger conversations.