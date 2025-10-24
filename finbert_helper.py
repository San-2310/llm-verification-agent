# finbert_helper.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "ProsusAI/finbert"   # HF model

print("Loading FinBERT model (this may download weights)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
LABELS = ["negative", "neutral", "positive"]

def get_finbert_sentiment(text, max_length=512):
    if not text or len(text.strip()) == 0:
        return {"sentiment": "neutral", "confidence": 0.0}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return {"sentiment": LABELS[best_idx], "confidence": probs[best_idx], "scores": probs}