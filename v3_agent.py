"""
v3_agent.py — V3 agent, backed by Postgres/pgvector instead of a local pickle.

Two ways to use this file:
1. CLI:  py v3_agent.py acme-solar "task"
2. Programmatic with callbacks (for Streamlit / FastAPI):
     from v3_agent import run_agent
     run_agent(engagement, task, on_event=my_callback)

The on_event callback receives structured dicts describing every step the
agent takes — searches, tool calls, results, exit. If on_event is None,
the agent prints to stdout.
"""

import json
import os
import sys

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

MODEL = "claude-sonnet-4-6"
TOP_K = 4
OUTPUT_DIR = "outputs"

# The embedding model must match the one used to build the stored vectors.
# Previously this was read from index.pkl; with the pickle gone it lives here.
# NOTE (Sprint 2): store this alongside the vectors so it cannot drift.
# The vector(384) column dimension gives partial protection — a model with a
# different output size would fail on insert — but two different 384-dim
# models would fail silently, returning nonsense rankings.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_EMBEDDING_MODEL = None
_SUPABASE = None


def _lazy_load():
    """
    Load the embedding model and Supabase client once, then reuse.

    The model is ~90 MB and takes seconds to initialise, so loading it per
    request would dominate response time. The corpus itself is no longer
    loaded at all — it stays in Postgres.
    """
    global _EMBEDDING_MODEL, _SUPABASE
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if _SUPABASE is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SECRET_KEY")
        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SECRET_KEY in .env")
        _SUPABASE = create_client(url, key)
    return _EMBEDDING_MODEL, _SUPABASE


def resolve_engagement(slug: str) -> str:
    """Turn a human-readable slug into the engagement's UUID."""
    _, sb = _lazy_load()
    res = sb.table("engagements").select("id, name").eq("slug", slug).execute()
    if not res.data:
        raise ValueError(f"No engagement with slug '{slug}'")
    return res.data[0]["id"]


TOOLS = [
    {
        "name": "list_documents",
        "description": (
            "List all source document filenames available for this engagement. "
            "Use this once at the start to know what documents exist. "
            "Note: you cannot read whole documents — use search_documents to query them."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_documents",
        "description": (
            "Search the document corpus for content relevant to a specific question or topic. "
            "Returns the top-k most semantically relevant chunks across all documents, "
            "with filename and chunk position for each result. "
            "Use targeted, specific queries — 'SLA response times' works better than 'requirements'. "
            "Call this multiple times with different queries to cover different aspects of the task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific question or topic to search for.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_output",
        "description": "Write the final reconciliation report (markdown) to the outputs folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
        },
    },
]

SYSTEM_PROMPT = """You are a discovery-to-scope agent for a Salesforce consulting engagement.

You have three tools:
1. list_documents — see what source documents exist for this engagement.
2. search_documents — retrieve relevant chunks by semantic query.
3. write_output — write the final report.

Approach:
- Start by listing documents to know the corpus.
- Then run several targeted searches to gather evidence — one per topic you need to compare
  (e.g., SLA targets, auto-close rules, escalation matrix, SAP integration, reverse logistics).
- For reconciliation, cross-check chunks from different source documents on the same topic.
- Cite the source filename + chunk index for every claim in your final report.
- Only state what the retrieved chunks support. If evidence is thin or absent, flag it as an open question.

Finally, call write_output with a markdown report:
  Sections: Confirmed Requirements | Conflicts | Open Questions | Scope Risks.

You MUST call write_output as your final action. Do not respond in prose.
"""


def _emit(on_event, event: dict, fallback_print: bool):
    """Send an event to the callback if provided, else print to stdout."""
    if on_event is not None:
        on_event(event)
    elif fallback_print:
        etype = event.get("type")
        if etype == "search":
            print(f"[iteration {event['iteration']}] search: \"{event['query']}\"")
        elif etype == "tool_call":
            print(f"[iteration {event['iteration']}] tool: {event['tool_name']} input: {event['tool_input']}")
        elif etype == "exit":
            print(f"[exit] stop_reason={event['stop_reason']}")


def run_tool(name: str, tool_input: dict, engagement_id: str) -> str:
    model, sb = _lazy_load()

    if name == "list_documents":
        res = (
            sb.table("documents")
            .select("filename")
            .eq("engagement_id", engagement_id)
            .execute()
        )
        return json.dumps([row["filename"] for row in res.data])

    if name == "search_documents":
        query = tool_input["query"]
        query_vec = model.encode(query).tolist()

        res = sb.rpc("match_chunks", {
            "query_embedding":   query_vec,
            "target_engagement": engagement_id,
            "match_count":       TOP_K,
        }).execute()

        # Map the function's column names back to the shape the rest of the
        # system already expects, so the UI and system prompt stay unchanged.
        results = [
            {
                "score":       round(float(row["similarity"]), 3),
                "filename":    row["chunk_file"],
                "chunk_index": row["chunk_no"],
                "text":        row["chunk_text"],
            }
            for row in res.data
        ]
        return json.dumps(results, indent=2)

    if name == "write_output":
        # Writes to outputs/, never to the folder that gets indexed.
        # basename() still strips any directory component the model might
        # supply — the agent should not choose where on disk to write.
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, os.path.basename(tool_input["filename"]))
        with open(path, "w", encoding="utf-8") as f:
            f.write(tool_input["content"])
        return f"Written to {path}"

    return f"ERROR: unknown tool {name}"


def run_agent(engagement: str, task: str, max_iterations: int = 15, on_event=None) -> str:
    """
    Run the agent against one engagement.

    `engagement` is the human-readable slug (e.g. "acme-solar"), resolved
    once to a UUID and then used to scope every retrieval.
    """
    engagement_id = resolve_engagement(engagement)

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": task}]

    for i in range(max_iterations):
        _emit(on_event, {"type": "iteration_start", "iteration": i + 1}, fallback_print=False)

        response = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            _emit(on_event, {"type": "exit", "stop_reason": response.stop_reason, "final_text": final_text}, fallback_print=True)
            return final_text

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "search_documents":
                    _emit(on_event, {
                        "type": "search",
                        "iteration": i + 1,
                        "query": block.input.get("query", ""),
                    }, fallback_print=True)
                else:
                    _emit(on_event, {
                        "type": "tool_call",
                        "iteration": i + 1,
                        "tool_name": block.name,
                        "tool_input": block.input,
                    }, fallback_print=True)

                output = run_tool(block.name, block.input, engagement_id)

                _emit(on_event, {
                    "type": "tool_result",
                    "iteration": i + 1,
                    "tool_name": block.name,
                    "output": output,
                }, fallback_print=False)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": tool_results})

    _emit(on_event, {"type": "exit", "stop_reason": "max_iterations", "final_text": "Stopped: hit max_iterations safety cap."}, fallback_print=True)
    return "Stopped: hit max_iterations safety cap."


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: py v3_agent.py <engagement-slug> "<task>"')
        sys.exit(1)
    print(run_agent(sys.argv[1], sys.argv[2]))