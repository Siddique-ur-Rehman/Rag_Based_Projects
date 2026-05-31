from datasets import load_dataset
import json

dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
samples = dataset.select(range(500))

data = []
for item in samples:
    data.append({
        "question": item["question"],
        "context": item["context"]["contexts"][0],  # first abstract
        "answer": item["final_decision"]
    })

with open("pubmed_data.json", "w") as f:
    json.dump(data, f)

print(f"Saved {len(data)} samples")