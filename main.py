import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABEL_TO_ID = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass
class Config:
    model_name: str = "bert-base-uncased"
    output_dir: str = "outputs/bert-fever"
    train_path: str = "data/fever/train_formatted_cleaned.jsonl"
    test_path: str = "data/fever/test_formatted_cleaned.jsonl"
    max_length: int = 384
    max_evidence_sentences: int = 5
    epochs: float = 1.0
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_evidence_text(row: dict, max_evidence_sentences: int) -> str:
    evidence = row.get("evidence", [])
    articles = row.get("articles", {})

    selected = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue

        doc_id = ev.get("doc_id")
        sentence_id = ev.get("sentence_id")
        if doc_id is None or sentence_id is None:
            continue

        doc_sentences = articles.get(doc_id, [])
        if not isinstance(doc_sentences, list):
            continue

        if 0 <= sentence_id < len(doc_sentences):
            sentence_text = str(doc_sentences[sentence_id]).strip()
            if sentence_text:
                selected.append(sentence_text)

        if len(selected) >= max_evidence_sentences:
            break

    if not selected:
        return "NO_EVIDENCE"

    return " [EVIDENCE] ".join(selected)


def prepare_model_records(rows: list[dict], max_evidence_sentences: int) -> list[dict]:
    records = []

    for row in rows:
        label = row.get("label")
        if label not in LABEL_TO_ID:
            continue

        claim = str(row.get("claim", "")).strip()
        if not claim:
            continue

        records.append(
            {
                "text": claim,
                "text_pair": build_evidence_text(row, max_evidence_sentences=max_evidence_sentences),
                "label": LABEL_TO_ID[label],
            }
        )

    return records


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


def main() -> None:
    cfg = Config()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. A ROCm-compatible GPU is required. "
            "Install a ROCm-enabled PyTorch build and verify with: "
            "python -c 'import torch; print(torch.cuda.is_available())'"
        )

    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")

    train_path = Path(cfg.train_path)
    test_path = Path(cfg.test_path)

    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    print(f"Loaded train file: {train_path}")
    print(f"Loaded test file: {test_path}")
    print(f"Train rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")

    train_examples = prepare_model_records(train_data, max_evidence_sentences=cfg.max_evidence_sentences)
    test_examples = prepare_model_records(test_data, max_evidence_sentences=cfg.max_evidence_sentences)

    print(f"Train examples used: {len(train_examples)}")
    print(f"Test examples used: {len(test_examples)}")

    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            batch["text_pair"],
            truncation=True,
            max_length=cfg.max_length,
        )

    train_tokenized = train_dataset.map(tokenize_fn, batched=True)
    test_tokenized = test_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:")
    print(metrics)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()