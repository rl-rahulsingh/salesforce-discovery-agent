"""
cost_check.py — print token usage and cost for one agent run.

Reads the numbers straight from the Anthropic API responses rather
than the dashboard, so the arithmetic is visible.
"""

import sys
import time
import anthropic
from v3_agent import (
    MODEL, SYSTEM_PROMPT, TOOLS,
    resolve_engagement, verify_embedding_model, run_tool,
)

# Claude Sonnet pricing, USD per million tokens.
PRICE_IN = 3.00
PRICE_OUT = 15.00

engagement = sys.argv[1] if len(sys.argv) > 1 else "acme-solar"
task = sys.argv[2] if len(sys.argv) > 2 else "Identify all conflicting requirements and open questions."

engagement_id = resolve_engagement(engagement)
verify_embedding_model(engagement_id)

client = anthropic.Anthropic()
messages = [{"role": "user", "content": task}]

rows = []
t_start = time.time()

for i in range(15):
    t0 = time.time()
    response = client.messages.create(
        model=MODEL, max_tokens=8000,
        system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
    )
    api_secs = time.time() - t0

    rows.append({
        "iter": i + 1,
        "in": response.usage.input_tokens,
        "out": response.usage.output_tokens,
        "api_s": api_secs,
    })

    if response.stop_reason != "tool_use":
        break

    messages.append({"role": "assistant", "content": response.content})
    tool_results = []
    t1 = time.time()
    for block in response.content:
        if block.type == "tool_use":
            out = run_tool(block.name, block.input, engagement_id)
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id, "content": out,
            })
    rows[-1]["tool_s"] = time.time() - t1
    messages.append({"role": "user", "content": tool_results})

total_s = time.time() - t_start

print()
print(f"{'Iter':>4} | {'Input tok':>10} | {'Output tok':>10} | {'API s':>7} | {'Tool s':>7}")
print("-" * 55)
for r in rows:
    print(f"{r['iter']:>4} | {r['in']:>10,} | {r['out']:>10,} | "
          f"{r['api_s']:>7.1f} | {r.get('tool_s', 0):>7.1f}")

tin = sum(r["in"] for r in rows)
tout = sum(r["out"] for r in rows)
api_s = sum(r["api_s"] for r in rows)
tool_s = sum(r.get("tool_s", 0) for r in rows)

cost_in = tin / 1_000_000 * PRICE_IN
cost_out = tout / 1_000_000 * PRICE_OUT

print()
print(f"Iterations       : {len(rows)}")
print(f"Input tokens     : {tin:,}   ->  ${cost_in:.4f}")
print(f"Output tokens    : {tout:,}   ->  ${cost_out:.4f}")
print(f"TOTAL COST       : ${cost_in + cost_out:.4f}")
print()
print(f"Total time       : {total_s:.1f}s")
print(f"  waiting on API : {api_s:.1f}s  ({api_s/total_s*100:.0f}%)")
print(f"  running tools  : {tool_s:.1f}s  ({tool_s/total_s*100:.0f}%)")
print()
print(f"Input tokens iteration 1    : {rows[0]['in']:,}")
print(f"Input tokens final iteration: {rows[-1]['in']:,}")
print(f"Growth                      : {rows[-1]['in']/rows[0]['in']:.1f}x")