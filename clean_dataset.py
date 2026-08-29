import json
import re

texts = []
labels = []

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        text = clean_text(data["text"])
        label = data["sentiment"]

        texts.append(text)
        labels.append(label)