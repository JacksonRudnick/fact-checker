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
from tqdm import tqdm
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
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.3
    learning_rate: float = 2e-5
    epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    preprocess_spacy_batch_size: int = 128


class GraphAttentionVerifier(nn.Module):
    """Graph Attention Network for fact verification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim * num_heads
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


def build_dependency_graph_from_docs(claim_doc, sentence_doc) -> Tuple[nx.DiGraph, list[str]]:
    """
    Build a directed graph from claim and sentence using syntactic dependencies.
    
    Args:
        claim: Claim text
        sentence: Sentence text
        nlp: spaCy language model
        
    Returns:
        Tuple of (networkx graph, list of tokens)
    """
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
    claim_doc = nlp(claim)
    sentence_doc = nlp(sentence)
    return build_dependency_graph_from_docs(claim_doc, sentence_doc)


def graph_to_geometric_data(
    graph: nx.DiGraph,
    claim_text: str,
    sentence_text: str,
    tokenizer: AutoTokenizer,
    model = None,
) -> Data:
    """
    Convert networkx graph to PyTorch Geometric Data object with node embeddings.
    Encodes claim and sentence together (as BERT was fine-tuned) and maps embeddings to graph nodes.
    
    Args:
        graph: NetworkX directed graph with nodes labeled with token text
        claim_text: Claim text
        sentence_text: Evidence sentence text
        tokenizer: BERT tokenizer
        model: Optional BERT model for getting embeddings
        
    Returns:
        PyTorch Geometric Data object with BERT embeddings as node features
    """
    # Convert graph to PyTorch Geometric format
    data = from_networkx(graph)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = data.num_nodes
    
    if model is not None and num_nodes > 0:
        # Encode the full paired input (claim, sentence) as BERT was fine-tuned
        with torch.no_grad():
            inputs = tokenizer(
                claim_text,
                sentence_text,
                return_tensors="pt",
                truncation=True,
                max_length=384,
                return_token_type_ids=True,
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            # Use [CLS] token embedding as a shared representation for all nodes
            cls_embedding = outputs.last_hidden_state[0, 0]  # Shape: (768,)
        
        # Initialize all node features with the paired [CLS] embedding
        # This gives each node access to the full claim-sentence relationship
        x = cls_embedding.unsqueeze(0).expand(num_nodes, -1).cpu()
    else:
        # Use random initialization if no model provided
        x = torch.randn(num_nodes, 768, device=device).cpu()
    
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


def preprocess_and_cache_data(
    examples: list[dict],
    nlp,
    tokenizer: AutoTokenizer,
    bert_model,
    cache_path: str,
    device,
    spacy_batch_size: int = 128,
) -> list[Tuple[Data, int]]:
    """
    Preprocess examples: build graphs, get BERT embeddings, convert to geometric Data.
    Cache the results to disk for fast loading during training.
    
    Args:
        examples: List of claim/sentence/label dicts
        nlp: spaCy model
        tokenizer: BERT tokenizer
        bert_model: BERT model for embeddings
        cache_path: Path to save cached data
        device: torch device
        
    Returns:
        List of (Data, label) tuples
    """
    cached_path = Path(cache_path)
    if cached_path.exists():
        print(f"Loading cached data from {cache_path}...")
        try:
            cached_data = torch.load(cached_path, weights_only=False)
            return cached_data
        except TypeError:
            cached_data = torch.load(cached_path)
            return cached_data
        except Exception as e:
            print(f"Failed to load cache ({e}). Rebuilding cache: {cache_path}")
    
    print(f"Preprocessing {len(examples)} examples...")
    preprocessed = []
    error_count = 0

    claims = [example["claim"] for example in examples]
    sentences = [example["sentence"] for example in examples]
    labels = [example["label"] for example in examples]

    claim_docs = list(
        tqdm(
            nlp.pipe(claims, batch_size=spacy_batch_size),
            total=len(claims),
            desc=f"Parsing claims {Path(cache_path).stem}",
            unit="doc",
        )
    )
    sentence_docs = list(
        tqdm(
            nlp.pipe(sentences, batch_size=spacy_batch_size),
            total=len(sentences),
            desc=f"Parsing evidence {Path(cache_path).stem}",
            unit="doc",
        )
    )

    progress_bar = tqdm(range(len(examples)), desc=f"Preprocessing {Path(cache_path).stem}", unit="example")
    for i in progress_bar:
        claim = claims[i]
        sentence = sentences[i]
        label = labels[i]

        try:
            # Build graph from pre-parsed docs
            graph, tokens = build_dependency_graph_from_docs(claim_docs[i], sentence_docs[i])
            if len(tokens) == 0:
                continue

            # Convert to geometric data with paired BERT embeddings (matching fine-tune setup)
            data = graph_to_geometric_data(graph, claim, sentence, tokenizer, bert_model)
            preprocessed.append((data, label))
            progress_bar.set_postfix({"cached": len(preprocessed), "errors": error_count})

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"Error processing example {i}: {e}")
            progress_bar.set_postfix({"cached": len(preprocessed), "errors": error_count})
            continue
    
    # Save to cache
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(preprocessed, cached_path)
    print(f"Cached {len(preprocessed)} preprocessed examples to {cache_path}")
    
    return preprocessed


def evaluate_gat_model(
    model: GraphAttentionVerifier,
    examples: list[Tuple[Data, int]],
    device: torch.device,
    desc: str = "Eval",
) -> tuple[float, float, int]:
    """Evaluate model on cached graph examples and return (avg_loss, accuracy, total)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    error_count = 0

    bar = tqdm(examples, desc=desc, unit="example")
    with torch.no_grad():
        for i, (data, label) in enumerate(bar):
            try:
                data = data.to(device)
                logits = model(data.x, data.edge_index)
                loss = F.cross_entropy(logits, torch.tensor([label], device=device))

                total_loss += loss.item()
                pred = logits.argmax(dim=1).item()
                if pred == label:
                    correct += 1
                total += 1

                bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{(correct / total):.4f}" if total else "0.0000",
                        "errors": error_count,
                    }
                )
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"Eval error example {i}: {e}")
                bar.set_postfix(
                    {
                        "loss": "n/a",
                        "acc": f"{(correct / total):.4f}" if total else "0.0000",
                        "errors": error_count,
                    }
                )

    if total == 0:
        return 0.0, 0.0, 0
    return total_loss / total, correct / total, total



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
    
    # Load fine-tuned BERT from final training output
    bert_model = AutoModel.from_pretrained("outputs/bert-fact-verifier").to(device)
    print(f"Loaded BERT model from: outputs/bert-fact-verifier")

    # Preprocess and cache training data
    train_cache_path = "outputs/gat-fact-verifier/train_data_cache.pt"
    train_data_processed = preprocess_and_cache_data(
        train_examples,
        nlp,
        tokenizer,
        bert_model,
        train_cache_path,
        device,
        spacy_batch_size=cfg.preprocess_spacy_batch_size,
    )
    
    # Preprocess and cache test data
    test_cache_path = "outputs/gat-fact-verifier/test_data_cache.pt"
    test_data_processed = preprocess_and_cache_data(
        test_examples,
        nlp,
        tokenizer,
        bert_model,
        test_cache_path,
        device,
        spacy_batch_size=cfg.preprocess_spacy_batch_size,
    )

    # Initialize model
    model = GraphAttentionVerifier(
        input_dim=768,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    # Training loop on cached data
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        error_count = 0

        epoch_bar = tqdm(
            train_data_processed,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            unit="example",
        )
        for i, (data, label) in enumerate(epoch_bar):
            data = data.to(device)
            
            try:
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
                epoch_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{(correct / total):.4f}" if total else "0.0000",
                        "errors": error_count,
                    }
                )

            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"Error processing example {i}: {e}")
                epoch_bar.set_postfix(
                    {
                        "loss": "n/a",
                        "acc": f"{(correct / total):.4f}" if total else "0.0000",
                        "errors": error_count,
                    }
                )
                continue

        if total == 0:
            print(f"Epoch {epoch + 1} - No valid examples processed.")
        else:
            train_accuracy = correct / total
            train_avg_loss = total_loss / total
            print(f"Epoch {epoch + 1} Train - Loss: {train_avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        eval_loss, eval_accuracy, eval_total = evaluate_gat_model(
            model,
            test_data_processed,
            device,
            desc=f"Eval {epoch + 1}/{cfg.epochs}",
        )
        if eval_total == 0:
            print(f"Epoch {epoch + 1} Eval - No valid test examples processed.")
        else:
            print(f"Epoch {epoch + 1} Eval - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")

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
        model_path: Path to saved GAT model (gat_verifier.pt)
        claim: The claim to verify
        sentence: The selected sentence
        
    Returns:
        Dict with prediction and confidence scores
    """
    cfg = GATVerifierConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nlp = load_nlp_model()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    from transformers import AutoModel
    
    # Load fine-tuned BERT from final training output
    bert_model = AutoModel.from_pretrained("outputs/bert-fact-verifier").to(device)

    
    # Load GAT model
    model = GraphAttentionVerifier(
        input_dim=768,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        # Build graph
        graph, tokens = build_dependency_graph(claim, sentence, nlp)
        if len(tokens) == 0:
            return {"prediction": "NOT ENOUGH INFO", "confidence": 0.0}
        
        # Convert to geometric data with paired BERT embeddings
        data = graph_to_geometric_data(graph, claim, sentence, tokenizer, bert_model).to(device)
        
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
