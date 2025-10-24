# verification_agent.py
import json, sys, re
from finbert_helper import get_finbert_sentiment
from groq_client import verify_with_groq

def parse_llm_json(raw_text):
    raw = raw_text.strip()
    try:
        return json.loads(raw)
    except Exception:
        # try to extract first JSON object substring
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return {"valid": False, "reason": "LLM returned unparseable JSON", "confidence": 0.0}
        return {"valid": False, "reason": "LLM returned non-JSON output", "confidence": 0.0}

def verification_agent(input_data):
    company = input_data.get("company")
    analyst_data = input_data.get("analyst_data", {})
    thesis_text = input_data.get("thesis", {}).get("claim", "")

    # 1) FinBERT sentiment on thesis
    finbert_res = get_finbert_sentiment(thesis_text)

    # 2) LLM verification via Groq
    raw = verify_with_groq(analyst_data, thesis_text)
    llm_json = parse_llm_json(raw)

    final = {
        "company": company,
        "thesis": thesis_text,
        "finbert_sentiment": finbert_res,
        "verification": llm_json,
        "raw_llm_output": raw
    }
    return final

if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "sample_input.json"
    with open(infile, "r") as f:
        data = json.load(f)
    out = verification_agent(data)
    print(json.dumps(out, indent=2))