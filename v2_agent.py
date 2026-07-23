import json
import os
import sys
import anthropic

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "list_documents",
        "description": "List all available document filenames in the workspace.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_document",
        "description": "Read the full text of one document by filename.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Exact filename from list_documents"}
            },
            "required": ["filename"],
        },
    },
    {
        "name": "write_output",
        "description": "Write the final deliverable (markdown) to a file.",
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
Use your tools to list and read the documents you need. Cross-check documents for:
- conflicting requirements (e.g., different SLA days in two documents)
- items in the newer document missing from the older scope
- unconfirmed assumptions
Then call write_output with a markdown report: Requirements | Conflicts | Open Questions | Draft Effort Line Items.
Only state what the documents support. Flag uncertainty explicitly.
"""


def run_tool(name: str, tool_input: dict, workspace: str) -> str:
    if name == "list_documents":
        files = [f for f in os.listdir(workspace) if f.endswith((".txt", ".md"))]
        return json.dumps(files)

    if name == "read_document":
        path = os.path.join(workspace, os.path.basename(tool_input["filename"]))
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"ERROR: {tool_input['filename']} not found."

    if name == "write_output":
        path = os.path.join(workspace, os.path.basename(tool_input["filename"]))
        with open(path, "w", encoding="utf-8") as f:
            f.write(tool_input["content"])
        return f"Written to {path}"

    return f"ERROR: unknown tool {name}"


def run_agent(workspace: str, task: str, max_iterations: int = 15) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": task}]

    for i in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            print(f"[exit] stop_reason={response.stop_reason}")
            return "".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[iteration {i+1}] tool: {block.name} input: {block.input}")
                output = run_tool(block.name, block.input, workspace)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: hit max_iterations safety cap."


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: py v2_agent.py <docs_folder> "<task>"')
        sys.exit(1)

    print(run_agent(sys.argv[1], sys.argv[2]))