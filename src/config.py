import torch
from dataclasses import dataclass

@dataclass
class MainConfig:
    device: torch.device = torch.device("cpu")
    train_path: str = "data/fever/train_formatted_cleaned.jsonl"
    test_path: str = "data/fever/test_formatted_cleaned.jsonl"

@dataclass
class RobertaConfig:
    model_name: str = "roberta-base"
    tokenizer_name: str = "roberta-base"
    output_dir: str = "outputs/roberta-fact-verifier"
    max_length: int = 512
    train_batch_size: int = 128
    eval_batch_size: int = 128
    learning_rate: float = 2e-5
    top_k: int = 5
    threshold: float = 0.3
    dropout: float = 0.1

    stage1_epochs: int = 3
    stage1_unfreeze_layers: int = 3

    stage2_epochs: int = 3
    stage2_unfreeze_layers: int = 2

LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}