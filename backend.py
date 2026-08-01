"""
backend.py — FastAPI wrapper around the Saras discovery agent.

Sprint 1: expose the agent over HTTP.
Phase 2: retrieval now comes from Postgres, scoped by engagement.
"""

import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from ingest import ingest_document

load_dotenv()

from v3_agent import run_agent, resolve_engagement

app = FastAPI(
    title="Saras — Discovery Intelligence",
    description="Module 1: extract structured discovery artifacts from engagement documents.",
    version="0.2.0",
)

DEFAULT_ENGAGEMENT = "acme-solar"


class RunRequest(BaseModel):
    task: str
    engagement: str = DEFAULT_ENGAGEMENT
    max_iterations: int = 15


class RunResponse(BaseModel):
    result: str
    engagement: str
    task: str


@app.get("/health")
def health():
    """Liveness check. Confirms secrets loaded, without exposing their values."""
    return {
        "status": "ok",
        "service": "saras-discovery-intelligence",
        "anthropic_key_loaded": bool(os.getenv("ANTHROPIC_API_KEY")),
        "supabase_url_loaded": bool(os.getenv("SUPABASE_URL")),
    }


@app.get("/engagements")
def list_engagements():
    """
    List available engagements.

    Useful for a caller that needs to know what slugs are valid before
    submitting a run — and a cheap way to confirm the database is reachable.
    """
    from v3_agent import _get_db
    sb = _get_db()
    res = sb.table("engagements").select("slug, name, created_at").execute()
    return {"engagements": res.data}

class EngagementRequest(BaseModel):
    slug: str
    name: str


@app.post("/engagements")
def create_engagement(request: EngagementRequest):
    """
    Create a new engagement.

    The slug is the human-readable handle used everywhere else, so it
    must be unique. A duplicate is the caller's error, not a server fault.
    """
    from v3_agent import _get_db
    sb = _get_db()

    existing = sb.table("engagements").select("id").eq("slug", request.slug).execute()
    if existing.data:
        raise HTTPException(
            status_code=400,
            detail=f"Engagement '{request.slug}' already exists.",
        )

    res = sb.table("engagements").insert({
        "slug": request.slug,
        "name": request.name,
    }).execute()
    return res.data[0]


@app.get("/documents")
def list_documents(engagement: str = DEFAULT_ENGAGEMENT):
    """List the documents uploaded to an engagement, with chunk counts."""
    from v3_agent import _get_db
    try:
        engagement_id = resolve_engagement(engagement)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sb = _get_db()
    docs = sb.table("documents").select("id, filename, created_at") \
        .eq("engagement_id", engagement_id).execute()

    out = []
    for d in docs.data:
        n = sb.table("chunks").select("id", count="exact") \
            .eq("document_id", d["id"]).execute()
        out.append({"filename": d["filename"], "chunks": n.count})
    return {"documents": out}

@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    engagement: str = Form("acme-solar"),
):
    """
    Upload a document to an engagement.

    The server chunks, embeds and stores it — no local build_index step.
    Re-uploading the same filename replaces the previous version.
    """
    try:
        engagement_id = resolve_engagement(engagement)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 text. Binary formats are not supported yet.",
        )

    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    result = ingest_document(engagement_id, file.filename, content)
    return result

@app.post("/run", response_model=RunResponse)
def run(request: RunRequest):
    """
    Run the discovery agent against one engagement.

    The agent searches that engagement's indexed documents in Postgres
    and returns its final report as text.
    """
    # Validate the engagement exists before starting a run that costs
    # 60-90 seconds and real API spend. Fail fast, and with a 400 (your
    # request was wrong) rather than a 500 (the server broke).
    try:
        resolve_engagement(request.engagement)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = run_agent(
            engagement=request.engagement,
            task=request.task,
            max_iterations=request.max_iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    return RunResponse(
        result=result,
        engagement=request.engagement,
        task=request.task,
    )