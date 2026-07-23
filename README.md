# Salesforce Discovery-to-Scope Agent

An AI agent that reads a folder of Salesforce consulting artifacts (meeting minutes, transcripts, scope notes) and produces a structured reconciliation report — highlighting confirmed requirements, conflicts, open questions, and scope risks — each claim traceable to a specific source chunk.

Built from scratch with the Anthropic API, retrieval-augmented generation (RAG), and a glass-box Streamlit UI.

<p align="center">
  <em>Live trace, final report, and retrieved evidence — every claim traceable to a specific chunk.</em>
</p>

---

## Why this exists

Discovery-phase reconciliation is one of the most time-consuming parts of Salesforce consulting. Multiple stakeholders. Multiple documents. Conflicting requirements buried in the second paragraph of MoM v2 that contradict a signed-off scope from three weeks ago. Reconciling them manually is a two-hour job that has to happen before any scope estimate can be locked.

This agent does the reconciliation, cites its evidence, and flags what needs a decision — in a form a consulting lead can review, challenge, and defend.

## What it does

Given a folder of source documents, the agent:

1. **Lists** the available documents.
2. **Runs targeted semantic searches** across all documents — one per topic it needs to reconcile (SLA targets, auto-close rules, escalation matrix, integration scope, etc.).
3. **Cross-checks** parallel chunks from different documents on the same topic.
4. **Writes a structured markdown report** with sections for:
   - Confirmed Requirements (consistent across sources)
   - Conflicts (direct contradictions — each with source citations)
   - Open Questions (ambiguities requiring stakeholder input)
   - Scope Risks (items likely to expand scope or invalidate estimates)

Every claim in the report cites `filename + chunk index` so anyone reading it can trace back to the source text.

## Architecture

```
                    ┌───────────────────────────┐
                    │   Streamlit UI (app.py)   │
                    │  ─ glass-box, 3-tab view  │
                    └────────────┬──────────────┘
                                 │
                                 │  on_event callback
                                 ▼
     ┌──────────────────────────────────────────────────┐
     │              Agent Loop (v3_agent.py)             │
     │                                                   │
     │  ┌──────────────────────┐                         │
     │  │   Anthropic Claude   │                         │
     │  │  (Sonnet, tool-use)  │                         │
     │  └──────────┬───────────┘                         │
     │             │  chooses tool                       │
     │             ▼                                     │
     │  ┌──────────────────────────────────────────┐     │
     │  │  Tools:                                  │     │
     │  │   • list_documents                       │     │
     │  │   • search_documents(query) ──► RAG      │     │
     │  │   • write_output(filename, content)      │     │
     │  └──────────────────────────────────────────┘     │
     └──────────────────────────────────────────────────┘
                                 │
                                 │ search_documents uses:
                                 ▼
     ┌──────────────────────────────────────────────────┐
     │  Retrieval (build_index.py + at runtime)          │
     │   • sentence-transformers (MiniLM-L6, 384-dim)    │
     │   • 500-char chunks, 100-char overlap             │
     │   • cosine similarity top-k                       │
     └──────────────────────────────────────────────────┘
```

The design principle behind every tool is **single responsibility**: `list_documents` only lists, `search_documents` only retrieves, `write_output` only writes. The model composes them. That's what makes it an *agent* rather than a workflow — the model controls the loop, not the code.

## Design decisions worth naming

**Why RAG replaces whole-document reading.** V1 of this agent read whole documents into context. That works for 3 documents. For 40, it breaks: context window limits, cost per call, and signal-to-noise degradation. Retrieval solves all three — the model asks targeted questions and receives semantically relevant chunks. Same agent architecture, smarter tool.

**Why single-purpose tools instead of one `do_everything()`.** Small tools let the model reason and compose them. One big tool moves reasoning back into hardcoded Python flow — and at that point it's a workflow again, not an agent. Small tools also make the trace auditable: you see exactly which sub-step the agent took.

**Why the UI is a glass box, not a black box.** In consulting, output trustworthiness matters more than output polish. The Streamlit UI exposes three levels of the agent's reasoning:
- **Live trace** — every search query the agent decides to run, streamed as it happens
- **Final report** — the markdown reconciliation, with citations
- **Retrieved evidence** — for each search: the query, top-4 chunks, similarity scores, source filenames

If a stakeholder challenges any claim, you can trace it back to a specific chunk with a specific similarity score. That's what makes an agent's output defensible in a consulting context.

**Why sandboxed file access.** The `read_document` and `write_output` tools use `os.path.basename()` to strip any directory paths the model provides — keeping filesystem access confined to the workspace folder. This is *path traversal defense* (OWASP top ten). Without it, a prompt-injected instruction in any document could redirect the agent to read SSH keys or overwrite config files.

## Repository structure

```
salesforce-discovery-agent/
├── README.md              — this file
├── requirements.txt       — Python dependencies
├── .gitignore
│
├── build_index.py         — chunk + embed documents, save index.pkl
├── retrieve.py            — standalone retrieval tester (CLI)
├── v3_agent.py            — the agent itself (CLI + callback interface)
├── app.py                 — Streamlit glass-box UI
│
└── docs/                  — sample synthetic consulting documents
    ├── old_mom.txt        — earlier MoM (baseline)
    ├── new_transcript.txt — later transcript (contains planted conflicts)
    └── scope_note.txt     — signed-off scope reference
```

The `docs/` folder ships with fully synthetic sample data (fictional company "Acme Solar Services") to make the agent immediately demoable. Drop your own `.txt` files in the same folder to run it against a real corpus.

## Quick start

**Prerequisites:** Python 3.11+, an [Anthropic API key](https://console.anthropic.com).

```bash
# 1. Clone and install
git clone https://github.com/YOUR-USERNAME/salesforce-discovery-agent.git
cd salesforce-discovery-agent
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 2. Set your API key
# Windows PowerShell:
$env:ANTHROPIC_API_KEY = "sk-ant-your-key"
# Mac/Linux:
export ANTHROPIC_API_KEY="sk-ant-your-key"

# 3. Build the retrieval index (one-time, ~1 minute)
python build_index.py ./docs

# 4a. Run the agent from the CLI
python v3_agent.py ./docs "Reconcile the latest MoM against the prior scope. List all conflicts."

# 4b. Or run the Streamlit UI
streamlit run app.py
```

## Try it — what to look for

When you run the agent against the sample data, you should see it:

1. Call `list_documents` once (iteration 1).
2. Run several targeted searches (iteration 2 onward) — SLA, auto-close rules, escalation, SAP integration, reverse logistics, etc.
3. Call `write_output` with a markdown report containing ~10 conflicts, several open questions, and scope risks.

The sample documents contain deliberately planted conflicts across topics like:
- Resolution SLA (48h vs 72h)
- Auto-close cadence (D+7 vs D+6)
- Intake channel count (5 vs 4)
- TAT measurement basis (calendar vs business hours)
- Dual-status closure model retained vs collapsed

If the agent catches these, retrieval + reconciliation are working.

## Known limitations

- **Fixed-size chunking** with overlap works for the sample data but is a naive choice. Real production RAG typically uses recursive character-based chunking with natural-boundary detection. See `build_index.py` for where this would be replaced.
- **No evaluation harness yet.** The agent's output quality is validated informally (planted-conflict test). A proper eval layer — measuring retrieval precision and generation faithfulness — is the natural next milestone.
- **Local embeddings only.** The `sentence-transformers/all-MiniLM-L6-v2` model is small and free but less nuanced than paid embedding APIs (OpenAI, Voyage). Fine for prototyping; production would benchmark against paid options.
- **No state persistence.** If the agent crashes mid-run, it starts over. A resumable version would require atomic state writes at iteration boundaries — see `run_agent` for where hooks would go.

## Roadmap

- [x] Layer 1 — Single-call extractor (V1)
- [x] Layer 2 — Agentic loop with tools (V2)
- [x] Layer 3 — Retrieval-augmented tool use (V3) + Streamlit UI
- [ ] Layer 4 — Evaluation harness (retrieval accuracy + generation faithfulness)
- [ ] Layer 5 — Production concerns (cost controls, observability, state persistence)

## About

Built by [Rahul Singh](https://linkedin.com/in/YOUR-LINKEDIN-URL) as a hands-on exploration of building agents from first principles rather than through frameworks. The goal was to internalize the tool-use pattern, retrieval design, and observability principles well enough to defend design choices in consulting engagements — not to build a production product.

Feedback and pushback welcome.
