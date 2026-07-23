import json
import sys
import anthropic

SYSTEM_PROMPT = """You are a senior Salesforce consultant's analyst.
Given raw meeting minutes (MoM) from a discovery call, extract:

1. requirements: explicit functional requirements (verbatim intent, your words)
2. assumptions: things stated as assumed, not confirmed
3. open_questions: ambiguities the consultant must clarify with the client
4. scope_risks: anything that smells like scope creep or a conflict

Respond ONLY with valid JSON, no markdown fences, using keys:
requirements, assumptions, open_questions, scope_risks (each a list of strings).
"""

def extract(mom_text: str) -> dict:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": mom_text}],
    )
    raw = response.content[0].text
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py v1_extractor.py <mom_file.txt>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        mom = f.read()
    result = extract(mom)
    print(json.dumps(result, indent=2))