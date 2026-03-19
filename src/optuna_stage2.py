import torch
import optuna
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.metrics import f1_score

from config import MainConfig, RobertaConfig
from roberta_verifier_model import RobertaVerifier
from dataset import RobertaStage2Dataset
from util import load_cuda, load_embeddings

def objective(trial, train_embeddings, test_embeddings, device):
    # hyperparameter search space
    lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    unfreeze_layers = trial.suggest_int("unfreeze_layers", 2, 4)
    top_k = trial.suggest_int("top_k", 3, 8)
    epochs = trial.suggest_int("epochs", 3, 5)

    # build config for this trial
    config = RobertaConfig()
    config.learning_rate = lr
    config.dropout = dropout
    config.stage2_unfreeze_layers = unfreeze_layers
    config.top_k = top_k
    config.stage2_epochs = epochs

    tokenizer = RobertaTokenizer.from_pretrained(config.tokenizer_name)

    train_dataset = RobertaStage2Dataset(train_embeddings, tokenizer, config)
    test_dataset = RobertaStage2Dataset(test_embeddings, tokenizer, config)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

    model = RobertaVerifier(config).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.roberta.encoder.layer[-unfreeze_layers:].parameters(), "lr": lr},
        {"params": model.roberta.pooler.parameters(), "lr": lr},  # type: ignore
        {"params": model.classifier.parameters(), "lr": lr * 10},
    ])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        print(f"  Trial {trial.number} — Epoch {epoch+1}/{epochs}", flush=True)

    # evaluate
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"Trial {trial.number} — Accuracy: {accuracy:.4f} | F1: {f1:.4f} | lr={lr:.2e} dropout={dropout:.2f} layers={unfreeze_layers} top_k={top_k} epochs={epochs}", flush=True)
    return f1


def main():
    main_config = MainConfig()
    load_cuda(main_config)

    roberta_config = RobertaConfig()
    train_embeddings_path = Path(roberta_config.output_dir) / "stage1_train_embeddings.pkl"
    test_embeddings_path = Path(roberta_config.output_dir) / "stage1_test_embeddings.pkl"

    print("Loading embeddings...", flush=True)
    train_embeddings = load_embeddings(train_embeddings_path)
    test_embeddings = load_embeddings(test_embeddings_path)

    study = optuna.create_study(
        direction="maximize",
        study_name="stage2_tuning",
        storage="sqlite:///optuna_stage2.db",  # saves progress to disk
        load_if_exists=True  # resume if interrupted
    )

    study.optimize(
        lambda trial: objective(trial, train_embeddings, test_embeddings, main_config.device),
        n_trials=20
    )

    print("\n---- Best Trial ----")
    best = study.best_trial
    print(f"  F1:     {best.value:.4f}")
    print(f"  Params: {best.params}")


if __name__ == "__main__":
    main()