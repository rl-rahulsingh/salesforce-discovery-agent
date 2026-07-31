"""
test_pg_retrieval.py — verify Postgres retrieval matches the old behaviour.

Runs the same queries the pickle-based retriever handled, so results
can be compared directly.
"""

import os
import pickle
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv()

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SECRET_KEY"))

# The embedding model MUST match the one used to build the stored vectors.
# Reading it from the index rather than hardcoding removes any chance of drift.
with open(os.path.join("docs", "index.pkl"), "rb") as f:
    model_name = pickle.load(f)["model_name"]

print(f"Embedding model: {model_name}")
model = SentenceTransformer(model_name)

engagement_id = (
    sb.table("engagements").select("id").eq("slug", "acme-solar").execute().data[0]["id"]
)

QUERIES = [
    "What are the SLA response times?",
    "How are cases auto-closed?",
    "What is the integration with SAP?",
]

for query in QUERIES:
    query_vector = model.encode(query).tolist()

    result = sb.rpc("match_chunks", {
        "query_embedding":   query_vector,
        "target_engagement": engagement_id,
        "match_count":       4,
    }).execute()

    print()
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    for row in result.data:
        preview = row["chunk_text"][:80].replace("\n", " ")
        print(f"  {row['similarity']:.3f}  {row['chunk_file']} #{row['chunk_no']}")
        print(f"         {preview}...")