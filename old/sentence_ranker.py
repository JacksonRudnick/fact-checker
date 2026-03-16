"""
Part 1: Sentence Ranker
Selects the most relevant sentence from an article for a given claim.
Outputs the sentence number (index) that is most likely to verify the claim.
"""
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
    "NOT_VERIFIABLE": 0,
    "VERIFIABLE": 1,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass
class SentenceRankerConfig:
    model_name: str = "bert-base-uncased"
    output_dir: str = "outputs/bert-sentence-ranker"
    train_path: str = "data/sentence_ranker/train.jsonl"
    test_path: str = "data/sentence_ranker/test.jsonl"
    max_length: int = 384
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


def train_sentence_ranker() -> None:
    """Train the sentence ranking model."""
    cfg = SentenceRankerConfig()

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

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["claim"],
            batch["sentence"],
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


def select_sentence(
    model_path: str,
    claim: str,
    sentences: list[str],
) -> int:
    """
    Select the most relevant sentence for a claim.
    
    Args:
        model_path: Path to trained sentence ranker model
        claim: The claim text
        sentences: List of sentence texts from article
        
    Returns:
        Index of the most relevant sentence
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(
                claim,
                sentence,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            # Get the verifiable score (class 1)
            score = logits[0][1].item()
            scores.append(score)

    # Return index of highest scoring sentence
    best_idx = int(np.argmax(scores))
    return best_idx


if __name__ == "__main__":
    train_sentence_ranker()
