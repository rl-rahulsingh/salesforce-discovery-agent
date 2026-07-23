"""
v3_agent.py — V3 agent with optional streaming callbacks for UI use.

Two ways to use this file:
1. CLI (unchanged from before): py v3_agent.py ./docs "task"
2. Programmatic with callbacks (for Streamlit app):
     from v3_agent import run_agent
     run_agent(workspace, task, on_event=my_callback)

The on_event callback receives structured dicts describing every step the
agent takes — searches, tool calls, results, exit. UI consumes these to
render the live trace. If on_event is None, the agent prints to stdout
exactly like before, so existing CLI usage is unaffected.
"""

import json
import os
import pickle
import sys

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "claude-sonnet-4-6"
TOP_K = 4

_EMBEDDING_MODEL = None
_INDEX = None


def _lazy_load(workspace: str):
    global _EMBEDDING_MODEL, _INDEX
    if _INDEX is None:
        index_path = os.path.join(workspace, "index.pkl")
        with open(index_path, "rb") as f:
            _INDEX = pickle.load(f)
        _EMBEDDING_MODEL = SentenceTransformer(_INDEX["model_name"])
    return _EMBEDDING_MODEL, _INDEX


def _cosine_top_k(query_vec, chunk_vecs, k):
    q_norm = query_vec / np.linalg.norm(query_vec)
    c_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    scores = c_norms @ q_norm
    top_indices = np.argsort(scores)[-k:][::-1]
    return top_indices, scores[top_indices]


TOOLS = [
    {
        "name": "list_documents",
        "description": (
            "List all source document filenames available in the workspace. "
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
        "description": "Write the final reconciliation report (markdown) to a file in the workspace.",
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
1. list_documents — see what source documents exist.
2. search_documents — retrieve relevant chunks by semantic query.
3. write_output — write the final report to disk.

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


def run_tool(name: str, tool_input: dict, workspace: str) -> str:
    if name == "list_documents":
        files = [
            f for f in os.listdir(workspace)
            if f.endswith((".txt", ".md")) and f != "index.pkl"
        ]
        return json.dumps(files)

    if name == "search_documents":
        model, index = _lazy_load(workspace)
        query = tool_input["query"]
        query_vec = model.encode(query)
        chunk_vecs = np.array(index["vectors"])
        top_indices, top_scores = _cosine_top_k(query_vec, chunk_vecs, TOP_K)

        results = []
        for i, score in zip(top_indices, top_scores):
            chunk = index["chunks"][int(i)]
            results.append({
                "score": round(float(score), 3),
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            })
        return json.dumps(results, indent=2)

    if name == "write_output":
        path = os.path.join(workspace, os.path.basename(tool_input["filename"]))
        with open(path, "w", encoding="utf-8") as f:
            f.write(tool_input["content"])
        return f"Written to {path}"

    return f"ERROR: unknown tool {name}"


def run_agent(workspace: str, task: str, max_iterations: int = 15, on_event=None) -> str:
    """
    Run the agent. If on_event is provided, every step is streamed to that callback
    as a structured dict. Otherwise, prints trace to stdout (original CLI behaviour).

    Event shapes emitted:
      {"type": "iteration_start", "iteration": N}
      {"type": "search", "iteration": N, "query": "..."}
      {"type": "tool_call", "iteration": N, "tool_name": "...", "tool_input": {...}}
      {"type": "tool_result", "iteration": N, "tool_name": "...", "output": "..."}
      {"type": "exit", "stop_reason": "...", "final_text": "..."}
    """
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

                output = run_tool(block.name, block.input, workspace)

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
        print('Usage: py v3_agent.py <docs_folder> "<task>"')
        sys.exit(1)
    print(run_agent(sys.argv[1], sys.argv[2]))
