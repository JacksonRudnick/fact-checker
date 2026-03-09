
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import pickle
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import (
        Dataset, 
        DataLoader,
        )
from transformers import (
        BertModel,
        BertTokenizer,
        )

@dataclass
class MainConfig:
    device: torch.device = torch.device("cpu")
    train_path: str = "data/fever/train_formatted_cleaned.jsonl"
    test_path: str = "data/fever/test_formatted_cleaned.jsonl"

@dataclass
class BertConfig:
    model_name: str = "bert-base-uncased"
    tokenizer_name: str = "bert-base-uncased"
    output_dir: str = "outputs/bert-fact-verifier"
    max_length: int = 512
    epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    top_k: int = 5
    threshold: float = 0.5
    dropout: float = 0.1

class FeverStage1Dataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: BertTokenizer, config: BertConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.samples = []

        for i, row in enumerate(data):
            if i % 1000 == 0:
                print(f"Building dataset: {i}/{len(data)} claims processed", flush=True)

            claim = row["claim"]
            label = row["label"]

            # collect gold sentence ids
            gold = set()
            if label != "NOT ENOUGH INFO":
                for evidence in row["evidence"]:
                    doc_id = evidence["doc_id"]
                    sent_id = evidence["sentence_id"]
                    gold.add((doc_id, sent_id))

            # flatten all sentences across all articles
            for doc_id, sentences in row["articles"].items():
                for sent_id, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue

                    is_gold = (doc_id, sent_id) in gold

                    self.samples.append({
                        "claim": claim,
                        "sentence": sentence,
                        "label": 1 if is_gold else 0,
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "claim_id": row["id"]
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
                sample["claim"],
                sample["sentence"],
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
                )

        return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "token_type_ids": encoding["token_type_ids"].squeeze(0),
                "label": torch.tensor(sample["label"], dtype=torch.float),
                }
    
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "token_type_ids": torch.stack([b["token_type_ids"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "claim_id": [b["claim_id"] for b in batch],
            "doc_id": [b["doc_id"] for b in batch],
            "sent_id": [b["sent_id"] for b in batch],
        }

class BertRelevanceScorer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(768, 1)

        # freeze all layers
        for param in self.bert.parameters():
            param.requires_grad = False

        # unfreeze last 2 encoder layers
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        # always unfreeze pooler
        for param in self.bert.pooler.parameters(): # type: ignore
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )

        cls_token = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        cls_token = self.dropout(cls_token)
        logit = self.classifier(cls_token)              # [batch_size, 1]
        return logit

    def get_embeddings(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )
        cls_token = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        return cls_token

def train_bert(main_config: MainConfig, bert_config: BertConfig, device: torch.device):
    # load data
    train_data = load_jsonl(Path(main_config.train_path))
    test_data = load_jsonl(Path(main_config.test_path))

    print(f"Train rows: {len(train_data)}", flush=True)
    print(f"Test rows: {len(test_data)}", flush=True)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_config.tokenizer_name)

    # datasets
    train_dataset = FeverStage1Dataset(train_data, tokenizer, bert_config)
    test_dataset = FeverStage1Dataset(test_data, tokenizer, bert_config)

    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Test samples: {len(test_dataset)}", flush=True)

    # dataloaders
    train_loader = DataLoader(
            train_dataset,
            batch_size=bert_config.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn
            )
    test_loader = DataLoader(
            test_dataset,
            batch_size=bert_config.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn
            )

    # model
    model = BertRelevanceScorer(bert_config).to(device)

    # weighted loss
    num_neg = sum(1 for s in train_dataset.samples if s["label"] == 0)
    num_pos = sum(1 for s in train_dataset.samples if s["label"] == 1)
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer — lower lr for bert, higher for classifier head
    optimizer = torch.optim.AdamW([
        {"params": model.bert.encoder.layer[-2:].parameters(), "lr": bert_config.learning_rate},
        {"params": model.bert.pooler.parameters(), "lr": bert_config.learning_rate}, # type: ignore
        {"params": model.classifier.parameters(), "lr": bert_config.learning_rate * 10},
        ])

    # training loop
    for epoch in range(bert_config.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits.squeeze(-1), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{bert_config.epochs} — Loss: {avg_loss:.4f}", flush=True)

        evaluate_bert(model, test_loader, criterion, device)

    # save model
    torch.save(model.state_dict(), Path(bert_config.output_dir) / "stage1.pt")

    embeddings_path = Path(bert_config.output_dir) / "stage1_embeddings.pkl"
    run_stage1_inference(model, test_data, tokenizer, bert_config, device, embeddings_path)

def evaluate_bert(model: BertRelevanceScorer, loader: DataLoader, criterion: nn.BCEWithLogitsLoss, device: torch.device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits.squeeze(-1), labels)
            total_loss += loss.item()

            # sigmoid to convert logits to probabilities, then threshold at 0.5
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().long().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"  Eval Loss: {avg_loss:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

def run_stage1_inference(model: BertRelevanceScorer, data: list[dict], tokenizer: BertTokenizer, config: BertConfig, device: torch.device, output_path: Path):
    model.eval()
    results = []

    with torch.no_grad():
        for row in data:
            claim = row["claim"]
            label = row["label"]
            claim_id = row["id"]

            # collect all sentences with their doc/sent ids
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
                    "top_k_embeddings": None,
                    "top_k_sentences": None
                    })
                continue

            # tokenize all candidates
            encodings = tokenizer(
                    [claim] * len(candidates),
                    [c["sentence"] for c in candidates],
                    max_length=config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                    )

            input_ids = encodings["input_ids"].to(device) # type: ignore
            attention_mask = encodings["attention_mask"].to(device) # type: ignore
            token_type_ids = encodings["token_type_ids"].to(device) # type: ignore

            # get relevance scores and embeddings
            logits = model(input_ids, attention_mask, token_type_ids).squeeze(-1)
            probs = torch.sigmoid(logits)
            embeddings = model.get_embeddings(input_ids, attention_mask, token_type_ids)

            # handle NEI — if no sentence clears threshold, skip
            if label == "NOT ENOUGH INFO" or probs.max().item() < config.threshold:
                results.append({
                    "claim_id": claim_id,
                    "claim": claim,
                    "label": label,
                    "top_k_embeddings": None,
                    "top_k_sentences": None
                    })
                continue

            # take top-k by probability
            k = min(config.top_k, len(candidates))
            top_k_indices = torch.topk(probs, k).indices

            top_k_embeddings = embeddings[top_k_indices].cpu()
            top_k_sentences = [candidates[i.item()] for i in top_k_indices] # type: ignore

            results.append({
                "claim_id": claim_id,
                "claim": claim,
                "label": label,
                "top_k_embeddings": top_k_embeddings,
                "top_k_sentences": top_k_sentences
                })

    # save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} inference results to {output_path}")
    return results


def load_stage1_embeddings(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)

def load_cuda():
    if torch.cuda.is_available():
        main_config.device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(main_config.device)}")
    else:
        print("CUDA is not available.")
        exit(-1)

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


main_config = MainConfig()
bert_config = BertConfig()

def main():
    load_cuda()

    train_bert(main_config, bert_config, device=main_config.device)

if __name__ == "__main__":
    main()
