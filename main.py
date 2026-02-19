"""
Part 2: Graph Attention Network Fact Verifier
Classifies whether a claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO
using a Graph Attention Network with syntactic dependencies as edges.
Designed to work with the selected sentence from Part 1: Sentence Ranker.
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

# Import GAT verifier
from gat_verifier import (
    GraphAttentionVerifier,
    build_dependency_graph,
    graph_to_geometric_data,
    load_nlp_model,
    train_gat_verifier,
    verify_claim_with_gat,
)


LABEL_TO_ID = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass
class VerifierConfig:
    model_name: str = "bert-base-uncased"
    output_dir: str = "outputs/bert-fact-verifier"
    train_path: str = "data/fever/train_formatted_cleaned.jsonl"
    test_path: str = "data/fever/test_formatted_cleaned.jsonl"
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


def build_evidence_text(row: dict) -> str:
    """Extract a single supporting sentence from evidence."""
    evidence = row.get("evidence", [])
    articles = row.get("articles", {})

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
                return sentence_text

    return "NO_EVIDENCE"


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

        records.append(
            {
                "text": claim,
                "text_pair": build_evidence_text(row),
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
    """Train the GAT-based fact verifier (Part 2)."""
    train_gat_verifier()


def verify_claim(
    sentence_ranker_path: str,
    gat_verifier_path: str,
    claim: str,
    sentences: list[str],
) -> dict:
    """
    Two-part pipeline: Select relevant sentence, then verify the claim with GAT.
    
    Args:
        sentence_ranker_path: Path to trained sentence ranker model
        gat_verifier_path: Path to trained GAT verifier model
        claim: The claim to verify
        sentences: List of sentences from article
        
    Returns:
        Dict with selected_sentence_idx, sentence_text, and verification result
    """
    from sentence_ranker import select_sentence
    
    # Part 1: Select most relevant sentence
    sentence_idx = select_sentence(sentence_ranker_path, claim, sentences)
    selected_sentence = sentences[sentence_idx]
    
    # Part 2: Verify the claim using selected sentence with GAT
    result = verify_claim_with_gat(gat_verifier_path, claim, selected_sentence)
    
    return {
        "selected_sentence_idx": sentence_idx,
        "sentence": selected_sentence,
        **result,  # Includes prediction, confidence, and scores
    }


if __name__ == "__main__":
    main()