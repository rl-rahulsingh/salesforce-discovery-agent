"""
backend.py — FastAPI wrapper around the Saras discovery agent.

Sprint 1: expose the agent over HTTP.
Phase 2: retrieval now comes from Postgres, scoped by engagement.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

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
    from v3_agent import _lazy_load
    _, sb = _lazy_load()
    res = sb.table("engagements").select("slug, name, created_at").execute()
    return {"engagements": res.data}


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