"""
inspect_chunks.py — print every chunk in an engagement.

Ground truth for the eval set must be written against the chunks that
actually exist, not against what they were three corpus rebuilds ago.
"""

import sys
from v3_agent import _get_db, resolve_engagement

slug = sys.argv[1] if len(sys.argv) > 1 else "acme-solar"
sb = _get_db()
engagement_id = resolve_engagement(slug)

docs = sb.table("documents").select("id, filename") \
    .eq("engagement_id", engagement_id).execute().data
doc_names = {d["id"]: d["filename"] for d in docs}

chunks = sb.table("chunks") \
    .select("document_id, chunk_index, text") \
    .eq("engagement_id", engagement_id).execute().data

chunks.sort(key=lambda c: (doc_names[c["document_id"]], c["chunk_index"]))

print(f"Engagement: {slug}  —  {len(chunks)} chunks across {len(docs)} documents\n")

for c in chunks:
    fname = doc_names[c["document_id"]]
    print("=" * 78)
    print(f"{fname}  #{c['chunk_index']}")
    print("=" * 78)
    print(c["text"])
    print()