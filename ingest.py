"""
ingest.py — server-side document ingestion.

Takes raw text, splits it into overlapping chunks, embeds each one,
and writes them to Postgres against an engagement.

This is build_index.py + migrate_to_supabase.py collapsed into one
server-side path, so ingestion no longer requires local files.
"""

from v3_agent import _get_db, _get_model, EMBEDDING_MODEL_NAME

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split text into overlapping windows.

    The overlap exists because a fixed cut can land mid-sentence and
    strand the second half of an idea in a chunk that no longer makes
    sense on its own. Overlapping windows mean any given sentence
    appears whole in at least one chunk.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def ingest_document(engagement_id: str, filename: str, content: str) -> dict:
    """
    Chunk, embed and store one document. Replaces any existing document
    with the same filename for this engagement, so re-uploading is safe
    rather than duplicating.
    """
    sb = _get_db()
    model = _get_model()

    # Idempotency: remove the previous version first. The cascade on
    # chunks means its chunks go with it.
    sb.table("documents").delete() \
        .eq("engagement_id", engagement_id) \
        .eq("filename", filename).execute()

    doc = sb.table("documents").insert({
        "engagement_id": engagement_id,
        "filename": filename,
        "content": content,
    }).execute()
    document_id = doc.data[0]["id"]

    pieces = chunk_text(content)

    # Embed all chunks in one call. The model batches internally, which
    # is far faster than one call per chunk.
    vectors = model.encode(pieces)

    rows = [
        {
            "document_id":   document_id,
            "engagement_id": engagement_id,
            "chunk_index":   i,
            "text":          piece,
            "embedding":     vec.tolist(),
            "model_name":    EMBEDDING_MODEL_NAME,
        }
        for i, (piece, vec) in enumerate(zip(pieces, vectors))
    ]

    sb.table("chunks").insert(rows).execute()

    return {
        "filename": filename,
        "document_id": document_id,
        "chunks_created": len(rows),
    }