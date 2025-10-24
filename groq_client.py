# groq_client.py
import os, json, requests
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env or environment variables.")

BASE_URL = "https://api.groq.com/openai/v1"
CHAT_ENDPOINT = f"{BASE_URL}/chat/completions"

def verify_with_groq(analyst_data: dict, thesis_text: str, model: str = "llama-3.1-8b-instant"):
    prompt = (
        "You are a financial verification agent. "
        "Analyst data (JSON):\n" + json.dumps(analyst_data, indent=2) + "\n\n"
        "Thesis claim (text):\n" + thesis_text + "\n\n"
        "Task: Check if the thesis is consistent with the analyst data. "
        "Provide your answer ONLY in JSON, with keys: valid (true/false), reason (string), confidence (float 0-1)."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a financial verification agent. Reply ONLY in JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(CHAT_ENDPOINT, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        print("Groq API Error:", resp.status_code, resp.text)  # 👈 detail print
        resp.raise_for_status()
    resp.raise_for_status()
    j = resp.json()

    # Response structure: choices[0].message.content (OpenAI-compatible)
    try:
        content = j["choices"][0]["message"]["content"]
    except Exception:
        # fallback: return full json
        content = json.dumps(j)
    return content