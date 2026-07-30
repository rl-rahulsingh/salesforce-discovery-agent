"""
eval_set.py — ground-truth test cases for retrieval evaluation.

Design principle: each test case names a query and the chunk(s) that MUST
appear in the retriever's top-k for the retrieval to be considered correct.

The chunk identity is a tuple: (filename, chunk_index).

Ground truth is hand-authored by inspecting the source documents.
"""

# Format: (query, [list of (filename, chunk_index) that should be retrieved])
#
# A query passes retrieval eval if ANY of the expected chunks appears in
# the retriever's top-k results. This is called "recall@k" — did we recover
# at least one relevant chunk?
#
# Some queries have multiple valid answers (e.g., SLA info exists in both
# old_mom and new_transcript — either or both being retrieved is a pass).
#
# Chunks were determined by running build_index.py and inspecting the
# resulting chunk boundaries. Recompute if chunk parameters change.

TEST_CASES = [
    # Q1 — SLA response times. Answer sits in both MoM chunks 1 (SLA targets line).
    (
        "What are the SLA response times?",
        [("old_mom.txt", 1), ("new_transcript.txt", 1)],
    ),

    # Q2 — Auto-close cadence. Both MoMs have D+6/D+7 auto-close rules.
    # This is the query that FAILED in the Weekend 3 eval (topic dilution).
    # Keeping it here as the canonical hard case.
    (
        "How are cases auto-closed?",
        [("old_mom.txt", 1), ("new_transcript.txt", 1)],
    ),

    # Q3 — Escalation matrix. Both MoMs have escalation tier info.
    (
        "What is the escalation matrix?",
        [("old_mom.txt", 1), ("new_transcript.txt", 1)],
    ),

    # Q4 — SAP integration / inventory role. Old MoM chunk 2, new_transcript chunk 2,
    # and scope_note chunks 1-2 all discuss the SFDC-vs-SAP split.
    (
        "What is the integration with SAP?",
        [("old_mom.txt", 2), ("new_transcript.txt", 2), ("scope_note.txt", 1), ("scope_note.txt", 2)],
    ),

    # Q5 — Reverse logistics pickup model. Old MoM chunk 2 and new_transcript chunks 2-3.
    (
        "How does reverse logistics work for faulty parts?",
        [("old_mom.txt", 2), ("new_transcript.txt", 2), ("new_transcript.txt", 3)],
    ),

    # Q6 — Reschedule limits. Both MoMs have reschedule rules.
    (
        "How many times can a customer reschedule a visit?",
        [("old_mom.txt", 1), ("new_transcript.txt", 1), ("new_transcript.txt", 2)],
    ),

    # Q7 — Preventive maintenance. Only in scope_note.
    (
        "What is in scope for Preventive Maintenance?",
        [("scope_note.txt", 2)],
    ),

    # Q8 — Intake channels. Both MoMs, plus scope_note chunk 0 for the L1 layer.
    (
        "What are the case intake channels?",
        [("old_mom.txt", 0), ("new_transcript.txt", 0)],
    ),

    # Q9 — Dual-status closure model. Old MoM chunks 2-3, new_transcript chunk 3,
    # scope_note chunk 1.
    (
        "How does the dual-status closure model work?",
        [("old_mom.txt", 3), ("new_transcript.txt", 3), ("scope_note.txt", 1)],
    ),

    # Q10 — Engagement type (revamp vs greenfield). Old MoM chunk 0, new_transcript chunk 0,
    # scope_note chunk 0.
    (
        "Is this a revamp or a new build?",
        [("old_mom.txt", 0), ("new_transcript.txt", 0), ("scope_note.txt", 0)],
    ),
]
