import json
from dataclasses import dataclass
from pathlib import Path

import torch
import pickle
import spacy
import torch.nn as nn
import os
from multiprocessing import Pool
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import (Dataset, DataLoader)
from transformers import (BertModel, BertTokenizer)
from torch_geometric.nn.models import GAT
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader as PyGDataLoader

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
    train_batch_size: int = 128
    eval_batch_size: int = 128
    learning_rate: float = 2e-5
    top_k: int = 5
    threshold: float = 0.5
    dropout: float = 0.1

@dataclass
class GatConfig:
    output_dir: str = "outputs/gat-fact-verifier"
    train_batch_size: int = 128
    eval_batch_size: int = 128
    in_channels: int = 768
    hidden_channels: int = 512
    out_channels: int = 3
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 10
    nlp = spacy.load("en_core_web_sm")
    train_cache_path = "outputs/gat-fact-verifier/train_graphs.pt"
    eval_cache_path = "outputs/gat-fact-verifier/test_graphs.pt"

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
            "input_ids": encoding["input_ids"].squeeze(0), # type: ignore
            "attention_mask": encoding["attention_mask"].squeeze(0), # type: ignore
            "token_type_ids": encoding["token_type_ids"].squeeze(0), # type: ignore
            "label": torch.tensor(sample["label"], dtype=torch.float),
            "claim_id": sample["claim_id"],
            "doc_id": sample["doc_id"],
            "sent_id": sample["sent_id"]
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
    
class GATFactVerifier(nn.Module):
    def __init__(self, config: GatConfig):
        super().__init__()
        self.gat = GAT(
                in_channels=config.in_channels,
                hidden_channels=config.hidden_channels,
                out_channels=config.hidden_channels,
                heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout
                )
        
        self.classifier = nn.Linear(config.hidden_channels, config.out_channels)
        
    # cosine similarity between evidence and evidence embeddings as edge weights
    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.gat(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
        

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

        for i, batch in enumerate(train_loader):
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
            if i % 5000 == 0:
                print(f"  Batch {i}/{len(train_loader)} — Loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{bert_config.epochs} — Loss: {avg_loss:.4f}", flush=True)

        evaluate_bert(model, test_loader, criterion, device)

    Path(bert_config.output_dir).mkdir(parents=True, exist_ok=True)

    # save model
    torch.save(model.state_dict(), Path(bert_config.output_dir) / "stage1.pt")

    run_stage1_inference(model, test_data, tokenizer, bert_config, device, test_embeddings_path)

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
                    "top_k_sentences": None,
                    "claim_embedding": None
                    })
                continue

            # embed the claim itself
            claim_encoding = tokenizer(
                claim,
                max_length=config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            claim_input_ids = claim_encoding["input_ids"].to(device) # type: ignore
            claim_attention_mask = claim_encoding["attention_mask"].to(device) # type: ignore
            claim_token_type_ids = claim_encoding["token_type_ids"].to(device) # type: ignore
            claim_embedding = model.get_embeddings(claim_input_ids, claim_attention_mask, claim_token_type_ids)  # [1, 768]

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
                    "claim_embedding": claim_embedding.cpu(),
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
                "claim_embedding": claim_embedding.cpu(),
                "top_k_embeddings": top_k_embeddings,
                "top_k_sentences": top_k_sentences
                })

    # save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} inference results to {output_path}")
    return results


def train_gat(main_config, gat_config, device):
    train_embeddings = load_embeddings(train_embeddings_path)
    test_embeddings = load_embeddings(test_embeddings_path)

    tokenizer = BertTokenizer.from_pretrained(bert_config.tokenizer_name)
    bert_model = BertRelevanceScorer(bert_config).to(device)
    bert_model.load_state_dict(torch.load(Path(bert_config.output_dir) / "stage1.pt"))
    bert_model.eval()

    train_graphs = build_gat_dataset(train_embeddings, bert_model, tokenizer, device, gat_config.train_cache_path)
    test_graphs = build_gat_dataset(test_embeddings, bert_model, tokenizer, device, gat_config.eval_cache_path)

    model = GATFactVerifier(gat_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=gat_config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=gat_config.epochs)

    print(f"Train graphs: {len(train_graphs)}", flush=True)
    print(f"Test graphs: {len(test_graphs)}", flush=True)

    train_loader = PyGDataLoader(train_graphs, batch_size=gat_config.train_batch_size, shuffle=True)
    test_loader = PyGDataLoader(test_graphs, batch_size=gat_config.eval_batch_size, shuffle=False)

    for epoch in range(gat_config.epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
    
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.squeeze(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 1000 == 0:
                print(f"  Claim {i}/{len(train_loader)} — Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        print(f"Epoch {epoch+1}/{gat_config.epochs}, Loss: {total_loss/len(train_loader):.4f}")

        evaluate_gat(model, test_loader, device)

    Path(gat_config.output_dir).mkdir(parents=True, exist_ok=True)

    output_path = Path(gat_config.output_dir) / "gat_verifier.pt"
    torch.save(model.state_dict(), output_path)
    print(f"Saved GAT verifier to: {output_path}")
    
def evaluate_gat(model: GATFactVerifier, test_loader: PyGDataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = batch.y.squeeze(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

def result_to_graph(result, model: BertRelevanceScorer, tokenizer: BertTokenizer, device):
    label = LABEL_MAP[result["label"]]
    claim = result["claim"]
    top_k_sentences = result["top_k_sentences"]

    # sentences is a list of dicts with "sentence" key, or None for NEI
    sentences = [s["sentence"] for s in top_k_sentences] if top_k_sentences else []
    all_texts = [claim] + sentences

    # grab bert embeddings for each text
    all_node_embeddings = []  # list of tensors, one per text, each [num_words, 768]

    for text in all_texts:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=bert_config.max_length,
            return_offsets_mapping=False
        ).to(device)

        word_ids = encoding.word_ids()

        with torch.no_grad():
            outputs = model.bert(**encoding)
            hidden_states = outputs.last_hidden_state.squeeze(0).cpu()  # [seq_len, 768]

        # average wordpiece embeddings per word
        unique_word_ids = sorted(set(w for w in word_ids if w is not None))
        word_embeddings = []
        for word_idx in unique_word_ids:
            token_indices = [i for i, w in enumerate(word_ids) if w == word_idx]
            word_emb = hidden_states[token_indices].mean(dim=0)
            word_embeddings.append(word_emb)

        all_node_embeddings.append(torch.stack(word_embeddings))  # [num_words, 768]

    # spacy dependency edges
    all_docs = [gat_config.nlp(text) for text in all_texts]

    node_offsets = []  # starting node index for each text
    total_nodes = 0
    for emb in all_node_embeddings:
        node_offsets.append(total_nodes)
        total_nodes += emb.size(0)

    x = torch.cat(all_node_embeddings, dim=0).half()  # [total_nodes, 768]

    edge_src = []
    edge_dst = []

    # dependency edges within each text
    for text_idx, doc in enumerate(all_docs):
        offset = node_offsets[text_idx]
        for token in doc:
            if token.head != token:
                src = offset + token.head.i
                dst = offset + token.i
                edge_src.append(src)
                edge_dst.append(dst)

    # shared token edges between claim and each evidence sentence
    claim_doc = all_docs[0]
    claim_offset = node_offsets[0]
    for sent_idx, sent_doc in enumerate(all_docs[1:], start=1):
        sent_offset = node_offsets[sent_idx]
        for claim_token in claim_doc:
            for sent_token in sent_doc:
                if claim_token.text.lower() == sent_token.text.lower():
                    if not claim_token.is_stop:
                        edge_src.append(claim_offset + claim_token.i)
                        edge_dst.append(sent_offset + sent_token.i)

    if edge_src:

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        valid_mask = (edge_index[0] < total_nodes) & (edge_index[1] < total_nodes)
        edge_index = edge_index[:, valid_mask]
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long)
    )

def build_gat_dataset(embeddings, model, tokenizer, device, cache_path: Path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached graphs from {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)

    graphs = []
    for i, result in enumerate(embeddings):
        if i % 1000 == 0:
            print(f"Building graphs: {i}/{len(embeddings)}", flush=True)
        graph = result_to_graph(result, model, tokenizer, device)
        if graph is not None:
            graphs.append(graph)

        if i % 10000 == 0 and i > 0:
            torch.save(graphs, cache_path)
            print(f"Checkpoint saved at {i} graphs", flush=True)

    torch.save(graphs, cache_path)
    print(f"Saved {len(graphs)} graphs to {cache_path}", flush=True)
    return graphs

def load_embeddings(path: Path) -> list[dict]:
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

def rerun_stage1_inference():
    tokenizer = BertTokenizer.from_pretrained(bert_config.tokenizer_name)
    model = BertRelevanceScorer(bert_config).to(main_config.device)
    model.load_state_dict(torch.load(Path(bert_config.output_dir) / "stage1.pt"))
    
    train_data = load_jsonl(Path(main_config.train_path))
    test_data = load_jsonl(Path(main_config.test_path))
    
    print("Running inference on train data...", flush=True)
    run_stage1_inference(model, train_data, tokenizer, bert_config, main_config.device, train_embeddings_path)
    
    print("Running inference on test data...", flush=True)
    run_stage1_inference(model, test_data, tokenizer, bert_config, main_config.device, test_embeddings_path)

main_config = MainConfig()
bert_config = BertConfig()
gat_config = GatConfig()

train_embeddings_path = Path(bert_config.output_dir) / "stage1_train_embeddings.pkl"
test_embeddings_path = Path(bert_config.output_dir) / "stage1_test_embeddings.pkl"

LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

def main():
    load_cuda()

    # train_bert(main_config, bert_config, device=main_config.device)
    
    # rerun_stage1_inference()
    
    train_gat(main_config, gat_config, device=main_config.device)

if __name__ == "__main__":
    main()
