"""
app.py — Streamlit glass-box UI for the Discovery-to-Scope Agent.

Design: expose the agent's reasoning, not hide it.
- Left panel: task input + live trace (search queries streaming as they happen)
- Right panel: three tabs
    1. Final Report (rendered markdown)
    2. Retrieved Evidence (per-search: query, top-k chunks, scores)
    3. Raw Trace (full technical log)

Run:
    streamlit run app.py

Requires: streamlit installed in the same venv.
    pip install streamlit
"""

import json
import os
import queue
import threading
from datetime import datetime

import streamlit as st

# Import from the agent — same file, same defensible code.
from v3_agent import run_agent

# ---------- Page setup ----------

st.set_page_config(
    page_title="Discovery-to-Scope Agent",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Trace event cards */
    .trace-item {
        padding: 8px 12px;
        margin: 6px 0;
        border-left: 3px solid #0a3d62;
        background: #f7f9fb;
        font-family: 'Courier New', monospace;
        font-size: 13px;
    }
    .trace-search { border-left-color: #0a3d62; }
    .trace-tool { border-left-color: #7f8c8d; }
    .trace-iter { border-left-color: #16a085; font-weight: bold; }
    .trace-exit { border-left-color: #c0392b; font-weight: bold; }

    /* Chunk display */
    .chunk-box {
        padding: 10px;
        margin: 6px 0;
        background: #ffffff;
        border: 1px solid #dfe4e8;
        border-radius: 4px;
    }
    .chunk-meta {
        font-size: 12px;
        color: #7f8c8d;
        font-family: 'Courier New', monospace;
    }
    .chunk-score {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-weight: bold;
        color: white;
    }
    .score-high { background: #16a085; }
    .score-mid { background: #f39c12; }
    .score-low { background: #c0392b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session state ----------

def _init_state():
    if "events" not in st.session_state:
        st.session_state.events = []
    if "final_text" not in st.session_state:
        st.session_state.final_text = ""
    if "final_report_path" not in st.session_state:
        st.session_state.final_report_path = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

_init_state()

# ---------- Helpers ----------

def _score_class(score: float) -> str:
    if score >= 0.5:
        return "score-high"
    if score >= 0.35:
        return "score-mid"
    return "score-low"


def _render_trace_item(event: dict):
    etype = event.get("type")
    if etype == "iteration_start":
        return f'<div class="trace-item trace-iter">▸ Iteration {event["iteration"]}</div>'
    if etype == "search":
        return f'<div class="trace-item trace-search">🔎 search: "{event["query"]}"</div>'
    if etype == "tool_call":
        return f'<div class="trace-item trace-tool">🛠 {event["tool_name"]}({event.get("tool_input", "")})</div>'
    if etype == "exit":
        return f'<div class="trace-item trace-exit">■ exit: {event["stop_reason"]}</div>'
    return ""


def _run_in_thread(engagement: str, task: str, event_q: queue.Queue):
    """
    Worker thread that runs the agent and pushes every event onto the queue.
    Main thread polls the queue to update the UI.
    """
    def on_event(event: dict):
        event_q.put(event)

    try:
        final_text = run_agent(engagement, task, on_event=on_event)
        event_q.put({"type": "_done", "final_text": final_text})
    except Exception as e:
        event_q.put({"type": "_error", "error": str(e)})


# ---------- Header ----------

st.title("🔎 Discovery-to-Scope Agent")
st.markdown(
    "**Glass-box view** — the agent's reasoning is visible alongside its output. "
    "Every search query it decides to run, every chunk it retrieves, every scoring decision."
)

# ---------- Layout ----------

left, right = st.columns([0.38, 0.62], gap="large")

# ==================== LEFT PANEL ====================

with left:
    st.subheader("Task")

    engagement = st.text_input(
        "Engagement",
        value="acme-solar",
        help="Engagement slug to query, e.g. 'acme-solar'.",
    )
    task = st.text_area(
        "What should the agent do?",
        value="Reconcile the latest MoM against the prior scope. List all conflicts, open questions, and scope risks.",
        height=110,
    )

    run_btn = st.button(
        "▶ Run agent",
        type="primary",
        disabled=st.session_state.is_running,
        use_container_width=True,
    )

    if run_btn and not st.session_state.is_running:
        # Reset state for a new run
        st.session_state.events = []
        st.session_state.final_text = ""
        st.session_state.final_report_path = None
        st.session_state.is_running = True

        # Kick off the agent in a background thread; UI polls its event queue
        event_q: queue.Queue = queue.Queue()
        thread = threading.Thread(
            target=_run_in_thread,
            args=(engagement, task, event_q),
            daemon=True,
        )
        thread.start()

        # Live trace panel — placeholder we update as events arrive
        st.markdown("---")
        st.subheader("Live trace")
        trace_area = st.empty()

        while True:
            try:
                event = event_q.get(timeout=90)  # per-event timeout
            except queue.Empty:
                st.error("Timed out waiting for the agent. Check the terminal for errors.")
                break

            if event.get("type") == "_done":
                st.session_state.final_text = event["final_text"]
                # Look for a reconciliation report the agent wrote
                for candidate in ("reconciliation_report.md", "reconciliation_report.txt.md"):
                    p = os.path.join("outputs", candidate)
                    if os.path.exists(p):
                        st.session_state.final_report_path = p
                        break
                break

            if event.get("type") == "_error":
                st.error(f"Agent error: {event['error']}")
                break

            st.session_state.events.append(event)

            # Re-render the whole trace so far
            trace_html = "".join(_render_trace_item(e) for e in st.session_state.events)
            trace_area.markdown(trace_html, unsafe_allow_html=True)

        st.session_state.is_running = False
        st.rerun()

    elif st.session_state.events and not st.session_state.is_running:
        st.markdown("---")
        st.subheader("Live trace")
        trace_html = "".join(_render_trace_item(e) for e in st.session_state.events)
        st.markdown(trace_html, unsafe_allow_html=True)

# ==================== RIGHT PANEL ====================

with right:
    tab_report, tab_evidence, tab_trace = st.tabs(
        ["📄 Final Report", "🔍 Retrieved Evidence", "🧾 Raw Trace"]
    )

    # -------- Tab 1: Final Report --------
    with tab_report:
        if st.session_state.final_report_path and os.path.exists(st.session_state.final_report_path):
            with open(st.session_state.final_report_path, "r", encoding="utf-8") as f:
                report_md = f.read()
            st.markdown(f"*Source: `{st.session_state.final_report_path}`*")
            st.markdown(report_md)

            st.download_button(
                "⬇ Download report (markdown)",
                data=report_md,
                file_name=f"reconciliation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )
        elif st.session_state.final_text:
            st.info("Agent finished but did not write a report file. Showing the model's final text response:")
            st.markdown(st.session_state.final_text)
        else:
            st.info("Run the agent to see the reconciliation report here.")

    # -------- Tab 2: Retrieved Evidence --------
    with tab_evidence:
        # Pair each search with the tool_result that followed it, in order.
        search_results = []
        pending_search = None
        for e in st.session_state.events:
            if e.get("type") == "search":
                pending_search = e
            elif e.get("type") == "tool_result" and e.get("tool_name") == "search_documents":
                if pending_search is not None:
                    search_results.append((pending_search, e))
                    pending_search = None

        if not search_results:
            st.info("Run the agent to see which searches it decided to run and what came back.")
        else:
            st.markdown(f"**{len(search_results)} search(es) run.** Each shows the agent's query and the top chunks it retrieved.")

            for idx, (search_event, result_event) in enumerate(search_results, start=1):
                with st.expander(
                    f"Search {idx}  |  iteration {search_event['iteration']}  |  \"{search_event['query']}\"",
                    expanded=(idx == 1),
                ):
                    try:
                        chunks = json.loads(result_event["output"])
                    except Exception:
                        st.text(result_event["output"])
                        continue

                    if not isinstance(chunks, list):
                        st.text(result_event["output"])
                        continue

                    for rank, chunk in enumerate(chunks, start=1):
                        score = chunk.get("score", 0)
                        klass = _score_class(score)
                        st.markdown(
                            f"""
                            <div class="chunk-box">
                                <div class="chunk-meta">
                                    Rank {rank}
                                    &nbsp;·&nbsp;
                                    <span class="chunk-score {klass}">score {score:.3f}</span>
                                    &nbsp;·&nbsp;
                                    {chunk.get('filename', '?')}  (chunk {chunk.get('chunk_index', '?')})
                                </div>
                                <div style="margin-top: 8px; font-size: 13px; white-space: pre-wrap;">{chunk.get('text', '')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # -------- Tab 3: Raw Trace --------
    with tab_trace:
        if not st.session_state.events:
            st.info("Run the agent to see the raw event trace.")
        else:
            st.markdown(f"**{len(st.session_state.events)} events recorded.**")
            st.code(
                json.dumps(st.session_state.events, indent=2, default=str),
                language="json",
            )

# ---------- Footer ----------

st.markdown("---")
st.caption(
    "Same agent as `v3_agent.py` — this UI is a wrapper, not a rewrite. "
    "All reasoning happens in the agent; the UI only visualises what the agent already does."
)
