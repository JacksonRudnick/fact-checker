import torch
from dataclasses import dataclass
import spacy

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
class RobertaConfig:
    model_name: str = "roberta-base"
    tokenizer_name: str = "roberta-base"
    output_dir: str = "outputs/roberta-fact-verifier"
    max_length: int = 512
    epochs: int = 6
    train_batch_size: int = 128
    eval_batch_size: int = 128
    learning_rate: float = 2e-5
    top_k: int = 10
    threshold: float = 0.3
    dropout: float = 0.1

@dataclass
class GatConfig:
    output_dir: str = "outputs/gat-fact-verifier"
    train_batch_size: int = 64
    eval_batch_size: int = 64
    in_channels: int = 768
    hidden_channels: int = 512
    out_channels: int = 3
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.3
    learning_rate: float = 3e-4
    epochs: int = 30
    nlp = spacy.load("en_core_web_sm")
    train_cache_path = "outputs/gat-fact-verifier/train_graphs.pt"
    eval_cache_path = "outputs/gat-fact-verifier/test_graphs.pt"

@dataclass
class TransformerConfig:
    output_dir: str = "outputs/transformer-fact-verifier"
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 8
    dim_feedforward: int = 3072
    dropout: float = 0.2
    learning_rate: float = 1e-4
    epochs: int = 20
    train_batch_size: int = 128
    eval_batch_size: int = 128
    num_classes: int = 3

LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}