import json
import pickle
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
from transformers import RobertaTokenizer

from config import MainConfig, RobertaConfig
from roberta_model import RobertaRelevanceScorer
from roberta_verifier_model import RobertaVerifier
from dataset import RobertaStage2Dataset


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    

def load_roberta_model(config: RobertaConfig, device: torch.device):
    tokenizer = RobertaTokenizer.from_pretrained(config.tokenizer_name)
    model = RobertaRelevanceScorer(config).to(device)
    model.load_state_dict(torch.load(Path(config.output_dir) / "stage1.pt", map_location=device))
    model.eval()
    return model, tokenizer


def evaluate_stage1(embeddings, test_data, config):
    gold_map = {}
    for row in test_data:
        claim_id = row["id"]
        gold = set()
        if row["label"] != "NOT ENOUGH INFO":
            for ev in row.get("evidence", []):
                if isinstance(ev, dict):
                    doc_id = ev.get("doc_id")
                    sent_id = ev.get("sentence_id")
                    if doc_id and sent_id is not None:
                        gold.add((doc_id, sent_id))
        gold_map[claim_id] = gold

    all_preds = []
    all_labels = []
    recall_hits = 0
    recall_total = 0

    for result in embeddings:
        claim_id = result["claim_id"]
        gold = gold_map.get(claim_id, set())
        candidates = result["candidates"]

        if not candidates:
            continue

        sorted_candidates = sorted(candidates, key=lambda x: x["prob"], reverse=True)
        top_k = sorted_candidates[:config.top_k]
        retrieved = {(c["doc_id"], c["sent_id"]) for c in top_k}

        for c in top_k:
            key = (c["doc_id"], c["sent_id"])
            all_preds.append(1)
            all_labels.append(1 if key in gold else 0)

        if gold:
            if gold & retrieved:
                recall_hits += 1
            recall_total += 1

    precision = precision_score(all_labels, all_preds, zero_division=0)
    retrieval_recall = recall_hits / recall_total if recall_total > 0 else 0

    print(f"Stage 1 Results (top-{config.top_k}):")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Retrieval Recall: {retrieval_recall:.4f} ({recall_hits}/{recall_total})")


def evaluate_stage2(model, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")


def main():
    main_config = MainConfig()
    roberta_config = RobertaConfig()

    if torch.cuda.is_available():
        main_config.device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        main_config.device = torch.device("cpu")
    device = main_config.device

    test_data = load_jsonl(main_config.test_path)
    test_embeddings_path = Path(roberta_config.output_dir) / "stage1_test_embeddings.pkl"
    test_embeddings = load_embeddings(test_embeddings_path)

    print("=== Stage 1 Evaluation ===")
    evaluate_stage1(test_embeddings, test_data, roberta_config)

    print("\n=== Stage 2 Evaluation ===")
    tokenizer = RobertaTokenizer.from_pretrained(roberta_config.tokenizer_name)
    test_dataset = RobertaStage2Dataset(test_embeddings, tokenizer, roberta_config)
    test_loader = DataLoader(test_dataset, batch_size=roberta_config.eval_batch_size, shuffle=False)

    model = RobertaVerifier(roberta_config).to(device)
    model.load_state_dict(torch.load(
        Path(roberta_config.output_dir) / "stage2.pt",
        map_location=device
    ))

    evaluate_stage2(model, test_loader, device)


if __name__ == "__main__":
    main()