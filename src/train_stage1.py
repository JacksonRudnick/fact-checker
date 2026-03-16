import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score

from config import MainConfig, BertConfig, RobertaConfig
from dataset import FeverStage1BertDataset, FeverStage1RobertaDataset, bert_collate_fn, roberta_collate_fn
from bert_model import BertRelevanceScorer
from roberta_model import RobertaRelevanceScorer


def train_bert(main_config: MainConfig, bert_config: BertConfig, device: torch.device, train_data: list[dict], test_data: list[dict]):
    tokenizer = BertTokenizer.from_pretrained(bert_config.tokenizer_name)

    train_dataset = FeverStage1BertDataset(train_data, tokenizer, bert_config.max_length)
    test_dataset = FeverStage1BertDataset(test_data, tokenizer, bert_config.max_length)

    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Test samples: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=bert_config.train_batch_size, shuffle=True, collate_fn=bert_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=bert_config.eval_batch_size, shuffle=False, collate_fn=bert_collate_fn)

    model = BertRelevanceScorer(bert_config).to(device)

    num_neg = sum(1 for s in train_dataset.samples if s["label"] == 0)
    num_pos = sum(1 for s in train_dataset.samples if s["label"] == 1)
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW([
        {"params": model.bert.encoder.layer[-3:].parameters(), "lr": bert_config.learning_rate},
        {"params": model.bert.pooler.parameters(), "lr": bert_config.learning_rate},  # type: ignore
        {"params": model.classifier.parameters(), "lr": bert_config.learning_rate * 10},
    ])

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

        print(f"Epoch {epoch + 1}/{bert_config.epochs} — Loss: {total_loss / len(train_loader):.4f}", flush=True)
        evaluate_bert(model, test_loader, criterion, device)

    Path(bert_config.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(bert_config.output_dir) / "stage1.pt")
    print(f"Saved BERT model to {bert_config.output_dir}/stage1.pt")
    return model, tokenizer


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

            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().long().numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"  Eval Loss: {total_loss / len(loader):.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")


def train_roberta(main_config: MainConfig, roberta_config: RobertaConfig, device: torch.device, train_data: list[dict], test_data: list[dict]):
    tokenizer = RobertaTokenizer.from_pretrained(roberta_config.tokenizer_name)

    train_dataset = FeverStage1RobertaDataset(train_data, tokenizer, roberta_config.max_length)
    test_dataset = FeverStage1RobertaDataset(test_data, tokenizer, roberta_config.max_length)

    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Test samples: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=roberta_config.train_batch_size, shuffle=True, collate_fn=roberta_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=roberta_config.eval_batch_size, shuffle=False, collate_fn=roberta_collate_fn)

    model = RobertaRelevanceScorer(roberta_config).to(device)

    num_neg = sum(1 for s in train_dataset.samples if s["label"] == 0)
    num_pos = sum(1 for s in train_dataset.samples if s["label"] == 1)
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW([
        {"params": model.roberta.encoder.layer[-3:].parameters(), "lr": roberta_config.learning_rate},
        {"params": model.roberta.pooler.parameters(), "lr": roberta_config.learning_rate},  # type: ignore
        {"params": model.classifier.parameters(), "lr": roberta_config.learning_rate * 10},
    ])

    for epoch in range(roberta_config.epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if i % 5000 == 0:
                print(f"  Batch {i}/{len(train_loader)} — Loss: {loss.item():.4f}", flush=True)

        print(f"Epoch {epoch + 1}/{roberta_config.epochs} — Loss: {total_loss / len(train_loader):.4f}", flush=True)
        evaluate_roberta(model, test_loader, criterion, device)

    Path(roberta_config.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(roberta_config.output_dir) / "stage1.pt")
    print(f"Saved RoBERTa model to {roberta_config.output_dir}/stage1.pt")
    return model, tokenizer


def evaluate_roberta(model: RobertaRelevanceScorer, loader: DataLoader, criterion: nn.BCEWithLogitsLoss, device: torch.device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.squeeze(-1), labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().long().numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"  Eval Loss: {total_loss / len(loader):.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")