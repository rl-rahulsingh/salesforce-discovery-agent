"""
migrate_to_supabase.py — move index.pkl into Postgres/pgvector.

Reads the local pickle index and writes it into the engagements /
documents / chunks tables in Supabase.

Safe to re-run: it clears existing rows for the target engagement first.
"""

import os
import pickle
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
ENGAGEMENT_SLUG = "acme-solar"
INDEX_PATH = os.path.join("docs", "index.pkl")
BATCH_SIZE = 100

if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_SECRET_KEY in .env")

# The SECRET key bypasses Row Level Security. Correct for a trusted
# backend script; never use this key in anything a browser can see.
sb = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)


# ---------------------------------------------------------------
# 1. Find the engagement row we're migrating into.
# ---------------------------------------------------------------
res = sb.table("engagements").select("id, name").eq("slug", ENGAGEMENT_SLUG).execute()

if not res.data:
    raise SystemExit(f"No engagement with slug '{ENGAGEMENT_SLUG}'. Create it first.")

engagement_id = res.data[0]["id"]
print(f"Engagement : {res.data[0]['name']}")
print(f"  id       : {engagement_id}")


# ---------------------------------------------------------------
# 2. Clear any previous migration for this engagement.
#
# This makes the script IDEMPOTENT — running it twice gives the same
# result as running it once. Without this, a second run would double
# every chunk, and retrieval would return duplicates.
#
# Deleting documents is enough: the ON DELETE CASCADE on chunks
# removes their children automatically.
# ---------------------------------------------------------------
sb.table("documents").delete().eq("engagement_id", engagement_id).execute()
print("Cleared previous documents and chunks for this engagement.")


# ---------------------------------------------------------------
# 3. Load the local index.
# ---------------------------------------------------------------
with open(INDEX_PATH, "rb") as f:
    index = pickle.load(f)

chunks = index["chunks"]
vectors = index["vectors"]
print(f"Loaded {len(chunks)} chunks from {INDEX_PATH}")

if len(chunks) != len(vectors):
    raise SystemExit(f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")


# ---------------------------------------------------------------
# 4. Create one document row per distinct filename.
#
# Parents must exist before children — the foreign key on chunks
# would reject any row pointing at a document_id that isn't there.
# ---------------------------------------------------------------
filenames = sorted({c["filename"] for c in chunks})
doc_ids = {}

for filename in filenames:
    full_text = "\n".join(
        c["text"] for c in sorted(
            (c for c in chunks if c["filename"] == filename),
            key=lambda c: int(c["chunk_index"]),
        )
    )
    inserted = sb.table("documents").insert({
        "engagement_id": engagement_id,
        "filename": filename,
        "content": full_text,
    }).execute()

    doc_ids[filename] = inserted.data[0]["id"]
    print(f"  document created: {filename}")


# ---------------------------------------------------------------
# 5. Build chunk rows.
#
# .tolist() converts the NumPy array into plain Python floats.
# NumPy's float32 is not JSON-serialisable, so without this the
# insert fails with "Object of type ndarray is not JSON serializable".
# ---------------------------------------------------------------
rows = []
for chunk, vector in zip(chunks, vectors):
    rows.append({
        "document_id":   doc_ids[chunk["filename"]],
        "engagement_id": engagement_id,
        "chunk_index":   int(chunk["chunk_index"]),
        "text":          chunk["text"],
        "embedding":     vector.tolist(),
    })


# ---------------------------------------------------------------
# 6. Insert in batches.
#
# One row per request would mean 12 network round trips here, and
# 100,000 at real scale. Batching sends many rows per request.
# The batch is capped because a single request has a size limit.
# ---------------------------------------------------------------
for start in range(0, len(rows), BATCH_SIZE):
    batch = rows[start:start + BATCH_SIZE]
    sb.table("chunks").insert(batch).execute()
    print(f"  inserted chunks {start + 1}–{start + len(batch)}")


# ---------------------------------------------------------------
# 7. Verify by reading back from the database, not from memory.
# ---------------------------------------------------------------
check = sb.table("chunks").select("id", count="exact").eq(
    "engagement_id", engagement_id
).execute()

print()
print(f"Done. Rows in database: {check.count} (expected {len(rows)})")