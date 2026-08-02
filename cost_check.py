"""
cost_check.py — token usage and cost for one agent run, with cache accounting.
"""

import sys
import time
import anthropic
from v3_agent import (
    MODEL, SYSTEM_PROMPT, TOOLS,
    resolve_engagement, verify_embedding_model, run_tool,
    _set_cache_breakpoint,
)

PRICE_IN = 3.00          # fresh input
PRICE_CACHE_WRITE = 3.75 # 1.25x — paid once to store
PRICE_CACHE_READ = 0.30  # 0.1x  — paid on every reuse
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
    _set_cache_breakpoint(messages)
    t0 = time.time()
    response = client.messages.create(
        model=MODEL, max_tokens=8000,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        tools=TOOLS, messages=messages,
    )
    api_secs = time.time() - t0
    u = response.usage

    rows.append({
        "iter": i + 1,
        "in": u.input_tokens,
        "cw": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cr": getattr(u, "cache_read_input_tokens", 0) or 0,
        "out": u.output_tokens,
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
print(f"{'Iter':>4} | {'Fresh in':>9} | {'Cache wr':>9} | {'Cache rd':>9} | {'Out':>7} | {'API s':>6}")
print("-" * 68)
for r in rows:
    print(f"{r['iter']:>4} | {r['in']:>9,} | {r['cw']:>9,} | {r['cr']:>9,} | "
          f"{r['out']:>7,} | {r['api_s']:>6.1f}")

tin = sum(r["in"] for r in rows)
tcw = sum(r["cw"] for r in rows)
tcr = sum(r["cr"] for r in rows)
tout = sum(r["out"] for r in rows)

c_in = tin / 1_000_000 * PRICE_IN
c_cw = tcw / 1_000_000 * PRICE_CACHE_WRITE
c_cr = tcr / 1_000_000 * PRICE_CACHE_READ
c_out = tout / 1_000_000 * PRICE_OUT

print()
print(f"Iterations        : {len(rows)}")
print(f"Fresh input       : {tin:>8,}  ->  ${c_in:.4f}")
print(f"Cache writes      : {tcw:>8,}  ->  ${c_cw:.4f}")
print(f"Cache reads       : {tcr:>8,}  ->  ${c_cr:.4f}")
print(f"Output            : {tout:>8,}  ->  ${c_out:.4f}")
print(f"TOTAL COST        : ${c_in + c_cw + c_cr + c_out:.4f}")

# What the same tokens would have cost with no caching at all.
without = (tin + tcw + tcr) / 1_000_000 * PRICE_IN + c_out
print(f"Without caching   : ${without:.4f}")
if without > 0:
    saved = without - (c_in + c_cw + c_cr + c_out)
    print(f"Saved             : ${saved:.4f}  ({saved/without*100:.0f}%)")

print()
print(f"Total time        : {total_s:.1f}s")
api_s = sum(r["api_s"] for r in rows)
print(f"  waiting on API  : {api_s:.1f}s  ({api_s/total_s*100:.0f}%)")