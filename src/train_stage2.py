import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import RobertaTokenizer

from config import RobertaConfig
from dataset import RobertaStage2Dataset
from roberta_verifier_model import RobertaVerifier

def train_roberta_stage2(roberta_config: RobertaConfig, device: torch.device,  train_embeddings: list[dict], test_embeddings: list[dict]):
    tokenizer = RobertaTokenizer.from_pretrained(roberta_config.model_name)

    # train on retrieved sentences
    train_dataset = RobertaStage2Dataset(train_embeddings, tokenizer, roberta_config)

    # test on retrieved sentences
    test_dataset = RobertaStage2Dataset(test_embeddings, tokenizer, roberta_config)

    train_loader = DataLoader(train_dataset, batch_size=roberta_config.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=roberta_config.eval_batch_size, shuffle=False)

    model = RobertaVerifier(roberta_config).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.roberta.encoder.layer[-roberta_config.stage2_unfreeze_layers:].parameters(), "lr": roberta_config.learning_rate},
        {"params": model.roberta.pooler.parameters(), "lr": roberta_config.learning_rate}, # type: ignore
        {"params": model.classifier.parameters(), "lr": roberta_config.learning_rate * 10},
    ])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=roberta_config.stage2_epochs)

    for epoch in range(roberta_config.stage2_epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 1000 == 0:
                print(f"  Batch {i}/{len(train_loader)} — Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        print(f"Epoch {epoch+1}/{roberta_config.stage2_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        evaluate_stage2(model, test_loader, device)

    Path(roberta_config.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(roberta_config.output_dir) / "stage2.pt")
    print(f"Saved Stage 2 to {roberta_config.output_dir}/stage2.pt")


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