"""
Part 2: Graph Attention Network Fact Verifier
Classifies whether a claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO
using a Graph Attention Network with syntactic dependencies as edges.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch import nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
from transformers import AutoTokenizer
import networkx as nx


LABEL_TO_ID = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass
class GATVerifierConfig:
    model_name: str = "bert-base-uncased"
    output_dir: str = "outputs/gat-fact-verifier"
    train_path: str = "data/fever/train_formatted_cleaned.jsonl"
    test_path: str = "data/fever/test_formatted_cleaned.jsonl"
    hidden_dim: int = 768
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 2e-5
    epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 8


class GraphAttentionVerifier(nn.Module):
    """Graph Attention Network for fact verification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GATConv(in_channels, hidden_dim, heads=num_heads, dropout=dropout)
            )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * num_heads, 3)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for pooling
            
        Returns:
            Logits [batch_size, 3]
        """
        for i, gat in enumerate(self.gat_layers):
            x = gat(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Global average pooling over nodes
        if batch is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            # Pool by batch
            batch_size = batch.max().item() + 1
            x_pooled = []
            for i in range(batch_size):
                mask = batch == i
                x_pooled.append(x[mask].mean(dim=0))
            x = torch.stack(x_pooled)
        
        logits = self.classifier(x)
        return logits


def load_nlp_model():
    """Load spaCy model for dependency parsing."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp


def build_dependency_graph(claim: str, sentence: str, nlp) -> Tuple[nx.DiGraph, list[str]]:
    """
    Build a directed graph from claim and sentence using syntactic dependencies.
    
    Args:
        claim: Claim text
        sentence: Sentence text
        nlp: spaCy language model
        
    Returns:
        Tuple of (networkx graph, list of tokens)
    """
    # Process both texts
    claim_doc = nlp(claim)
    sentence_doc = nlp(sentence)
    
    # Build graph
    graph = nx.DiGraph()
    tokens = []
    token_to_idx = {}
    
    # Add claim tokens
    for token in claim_doc:
        idx = len(tokens)
        token_to_idx[("claim", token.i)] = idx
        tokens.append(token.text)
        graph.add_node(idx, token=token.text, pos=token.pos_)
    
    # Add sentence tokens with offset
    for token in sentence_doc:
        idx = len(tokens)
        token_to_idx[("sentence", token.i)] = idx
        tokens.append(token.text)
        graph.add_node(idx, token=token.text, pos=token.pos_)
    
    # Add dependency edges within claim
    for token in claim_doc:
        if token.head != token:
            parent_idx = token_to_idx[("claim", token.head.i)]
            child_idx = token_to_idx[("claim", token.i)]
            graph.add_edge(parent_idx, child_idx, dep=token.dep_)
    
    # Add dependency edges within sentence
    for token in sentence_doc:
        if token.head != token:
            parent_idx = token_to_idx[("sentence", token.head.i)]
            child_idx = token_to_idx[("sentence", token.i)]
            graph.add_edge(parent_idx, child_idx, dep=token.dep_)
    
    # Add edges between claim and sentence for shared tokens
    for claim_token in claim_doc:
        for sent_token in sentence_doc:
            if claim_token.text.lower() == sent_token.text.lower():
                claim_idx = token_to_idx[("claim", claim_token.i)]
                sent_idx = token_to_idx[("sentence", sent_token.i)]
                graph.add_edge(claim_idx, sent_idx, dep="shared")
    
    return graph, tokens


def graph_to_geometric_data(
    graph: nx.DiGraph,
    tokens: list[str],
    tokenizer: AutoTokenizer,
    model = None,
) -> Data:
    """
    Convert networkx graph to PyTorch Geometric Data object with node embeddings.
    
    Args:
        graph: NetworkX directed graph
        tokens: List of token strings
        tokenizer: BERT tokenizer for embeddings
        model: Optional BERT model for getting embeddings
        
    Returns:
        PyTorch Geometric Data object
    """
    # Convert graph to PyTorch Geometric format
    data = from_networkx(graph)
    
    # Initialize node feature embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tokens = len(tokens)
    
    if model is not None:
        # Get embeddings from BERT model
        from transformers import AutoModel
        x = torch.zeros(num_tokens, 768)
        
        with torch.no_grad():
            for i, token in enumerate(tokens):
                inputs = tokenizer(token, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                # Use [CLS] token or mean pooling
                x[i] = outputs.last_hidden_state[0, 0]  # Use [CLS] token embedding
    else:
        # Use random initialization if no model provided
        x = torch.randn(num_tokens, 768)
    
    data.x = x
    return data


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def prepare_model_records(rows: list[dict]) -> list[dict]:
    """Prepare training records with claim and single evidence sentence."""
    records = []

    for row in rows:
        label = row.get("label")
        if label not in LABEL_TO_ID:
            continue

        claim = str(row.get("claim", "")).strip()
        if not claim:
            continue

        # Extract first evidence sentence
        evidence = row.get("evidence", [])
        articles = row.get("articles", {})
        
        sentence_text = "NO_EVIDENCE"
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
                    break

        records.append(
            {
                "claim": claim,
                "sentence": sentence_text,
                "label": LABEL_TO_ID[label],
            }
        )

    return records


def train_gat_verifier() -> None:
    """Train the GAT-based fact verifier."""
    cfg = GATVerifierConfig()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. A ROCm-compatible GPU is required. "
            "Install a ROCm-enabled PyTorch build and verify with: "
            "python -c 'import torch; print(torch.cuda.is_available())'"
        )

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")

    train_path = Path(cfg.train_path)
    test_path = Path(cfg.test_path)

    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    print(f"Loaded train file: {train_path}")
    print(f"Loaded test file: {test_path}")
    print(f"Train records: {len(train_data)}")
    print(f"Test records: {len(test_data)}")

    train_examples = prepare_model_records(train_data)
    test_examples = prepare_model_records(test_data)

    print(f"Train examples prepared: {len(train_examples)}")
    print(f"Test examples prepared: {len(test_examples)}")

    # Load models
    nlp = load_nlp_model()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    from transformers import AutoModel
    bert_model = AutoModel.from_pretrained(cfg.model_name).to(device)

    # Initialize model
    model = GraphAttentionVerifier(
        input_dim=768,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for i, example in enumerate(train_examples[:100]):  # Limit for testing
            claim = example["claim"]
            sentence = example["sentence"]
            label = example["label"]

            try:
                # Build graph
                graph, tokens = build_dependency_graph(claim, sentence, nlp)
                if len(tokens) == 0:
                    continue

                # Convert to geometric data with BERT embeddings
                data = graph_to_geometric_data(graph, tokens, tokenizer, bert_model).to(device)

                # Forward pass
                logits = model(data.x, data.edge_index)
                loss = F.cross_entropy(logits, torch.tensor([label], device=device))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = logits.argmax(dim=1).item()
                if pred == label:
                    correct += 1
                total += 1

                if (i + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1} - Loss: {total_loss / total:.4f}, Accuracy: {accuracy:.4f}")

    # Save model
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / "gat_verifier.pt")
    print(f"Model saved to {output_path}")


def verify_claim_with_gat(
    model_path: str,
    claim: str,
    sentence: str,
) -> dict:
    """
    Verify a claim using GAT model with selected sentence.
    
    Args:
        model_path: Path to saved GAT model
        claim: The claim to verify
        sentence: The selected sentence
        
    Returns:
        Dict with prediction and confidence scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nlp = load_nlp_model()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    from transformers import AutoModel
    bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    
    # Load model
    model = GraphAttentionVerifier(
        input_dim=768,
        hidden_dim=768,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        # Build graph
        graph, tokens = build_dependency_graph(claim, sentence, nlp)
        if len(tokens) == 0:
            return {"prediction": "NOT ENOUGH INFO", "confidence": 0.0}
        
        # Convert to geometric data with BERT embeddings
        data = graph_to_geometric_data(graph, tokens, tokenizer, bert_model).to(device)
        
        # Forward pass
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        pred_id = logits.argmax(dim=1).item()
        confidence = probs[0, pred_id].item()
        
        return {
            "prediction": ID_TO_LABEL[pred_id],
            "confidence": confidence,
            "scores": {
                "SUPPORTS": probs[0, 0].item(),
                "REFUTES": probs[0, 1].item(),
                "NOT ENOUGH INFO": probs[0, 2].item(),
            }
        }


if __name__ == "__main__":
    train_gat_verifier()
