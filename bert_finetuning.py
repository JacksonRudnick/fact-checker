import torch
from dataclasses import dataclass

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding


@dataclass
class cfg:
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 2e-5
    max_seq_length: int = 128
    device: str = "cuda" if torch.cuda.is_available() else exit(-25)

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=cfg.num_labels).to(cfg.device)


def main():
