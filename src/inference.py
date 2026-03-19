import torch
import pickle
from pathlib import Path
from transformers import RobertaTokenizer

from config import RobertaConfig
from roberta_relevance_model import RobertaRelevanceScorer

def run_roberta_inference(model: RobertaRelevanceScorer, data: list[dict], tokenizer: RobertaTokenizer, config: RobertaConfig, device: torch.device, output_path: Path):
    model.eval()
    results = []

    with torch.no_grad():
        for i, row in enumerate(data):
            if i % 1000 == 0:
                print(f"Inference: {i}/{len(data)}", flush=True)

            claim = row["claim"]
            label = row["label"]
            claim_id = row["id"]

            # collect all sentences
            candidates = []
            for doc_id, sentences in row["articles"].items():
                for sent_id, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                    candidates.append({
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "sentence": sentence
                    })

            if not candidates:
                results.append({
                    "claim_id": claim_id,
                    "claim": claim,
                    "label": label,
                    "claim_embedding": None,
                    "candidates": []
                })
                continue

            # run claim and sentences through model to get relevance scores and embeddings
            claim_encoding = tokenizer(
                claim,
                max_length=config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            claim_embedding = model.get_embeddings(
                claim_encoding["input_ids"].to(device), #type: ignore
                claim_encoding["attention_mask"].to(device) #type: ignore
            )

            encodings = tokenizer(
                [claim] * len(candidates),
                [c["sentence"] for c in candidates],
                max_length=config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encodings["input_ids"].to(device) #type: ignore
            attention_mask = encodings["attention_mask"].to(device) #type: ignore

            # score relevance and get embeddings for each candidate sentence
            logits = model(input_ids, attention_mask).squeeze(-1)
            probs = torch.sigmoid(logits)
            embeddings = model.get_embeddings(input_ids, attention_mask)

            # store all candidates with their embeddings and relevance scores
            # top-k selection happens at Stage 2 time based on config.top_k
            candidates_with_embeddings = [
                {
                    "doc_id": c["doc_id"],
                    "sent_id": c["sent_id"],
                    "sentence": c["sentence"],
                    "embedding": embeddings[i].cpu(),
                    "prob": probs[i].item()
                }
                for i, c in enumerate(candidates)
            ]

            results.append({
                "claim_id": claim_id,
                "claim": claim,
                "label": label,
                "claim_embedding": claim_embedding.cpu(),
                "candidates": candidates_with_embeddings
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} inference results to {output_path}")
    return results

def load_roberta_model(config: RobertaConfig, device: torch.device):
    tokenizer = RobertaTokenizer.from_pretrained(config.tokenizer_name)
    model = RobertaRelevanceScorer(config).to(device)
    model.load_state_dict(torch.load(Path(config.output_dir) / "stage1.pt", map_location=device))
    model.eval()
    return model, tokenizer