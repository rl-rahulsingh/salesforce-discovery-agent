"""
eval_set.py — ground truth for retrieval evaluation.

Hand-authored on 2026-08-02 against the live 12-chunk corpus for the
acme-solar engagement. Verified by reading every chunk with
inspect_chunks.py — not written from memory.

IMPORTANT: chunk indices are tied to a specific chunking configuration
(500 chars, 100 overlap) and a specific corpus. Re-chunk or re-ingest
and this file must be rebuilt. It has already been invalidated once.

Format: (query, [list of (filename, chunk_index) that would each be a
correct retrieval]). A query passes at k if ANY expected chunk appears
in the top-k. That is recall@k: did we recover at least one chunk that
actually answers the question?
"""

TEST_CASES = [
    # Q1 — SLA targets. Stated explicitly in both MoMs, with different
    # resolution figures (48h vs 72h). Either is a correct retrieval.
    (
        "What are the SLA response and resolution targets?",
        [("new_transcript.txt", 1), ("old_mom.txt", 1)],
    ),

    # Q2 — Auto-close. The known-weak query. Scored 0.337 previously
    # because the chunks holding it also carry SLA, escalation,
    # reschedule and feedback content — classic topic dilution.
    (
        "When is a case auto-closed if the customer gives no feedback?",
        [("old_mom.txt", 1), ("new_transcript.txt", 1), ("new_transcript.txt", 2)],
    ),

    # Q3 — Escalation matrix. Full tier list in old_mom #1; the revised
    # 3-tier version in new_transcript #1.
    (
        "What is the escalation matrix and its tiers?",
        [("old_mom.txt", 1), ("new_transcript.txt", 1)],
    ),

    # Q4 — Reschedule limits. Spans a chunk boundary in old_mom
    # (#1 into #2) and in new_transcript (#1 into #2).
    (
        "How many times can a customer reschedule a visit?",
        [("old_mom.txt", 1), ("old_mom.txt", 2),
         ("new_transcript.txt", 1), ("new_transcript.txt", 2)],
    ),

    # Q5 — Inventory system of record. The SFDC-vs-SAP split appears
    # in all three documents.
    (
        "Is SFDC the system of record for inventory or only a visibility layer?",
        [("old_mom.txt", 2), ("new_transcript.txt", 2),
         ("scope_note.txt", 1), ("scope_note.txt", 2)],
    ),

    # Q6 — Reverse logistics pickup point. Directly contradicted
    # between the two MoMs; flagged as pending in the scope note.
    (
        "Where does the OEM collect faulty parts from?",
        [("old_mom.txt", 2), ("new_transcript.txt", 2),
         ("new_transcript.txt", 3), ("scope_note.txt", 3)],
    ),

    # Q7 — Preventive Maintenance. HARD CASE: exactly one chunk in the
    # whole corpus mentions it. No overlapping neighbour to fall back on.
    (
        "What is in scope for preventive maintenance?",
        [("scope_note.txt", 2)],
    ),

    # Q8 — Intake channels. Five in v1, four in v2 with Social deferred.
    (
        "What are the case intake channels?",
        [("old_mom.txt", 0), ("new_transcript.txt", 0)],
    ),

    # Q9 — Dual-status closure. Agreed in v1, elaborated in the scope
    # note, then eliminated in v2.
    (
        "How does the dual-status case closure model work?",
        [("old_mom.txt", 2), ("old_mom.txt", 3),
         ("new_transcript.txt", 3), ("scope_note.txt", 1)],
    ),

    # Q10 — Engagement approach. Stated in the opening chunk of all three.
    (
        "Is this a revamp of the live org or a greenfield rebuild?",
        [("old_mom.txt", 0), ("new_transcript.txt", 0), ("scope_note.txt", 0)],
    ),
]