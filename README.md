# Fact Checker: Two-Part Pipeline

A two-part neural pipeline for fact verification using BERT and Graph Attention Networks.

## Architecture

### Part 1: Sentence Ranker (`sentence_ranker.py`)

- **Task**: Select the most relevant sentence from an article for a given claim
- **Model**: BERT-based binary classifier (verifiable/not verifiable)
- **Input**: Claim + sentence pairs
- **Output**: Index of the most relevant sentence
- **Training Data**: Requires sentence-level labels indicating relevance to claims

### Part 2: Fact Verifier (`gat_verifier.py`)

- **Task**: Classify whether a claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO
- **Model**: Graph Attention Network with syntactic dependencies
- **Input**: Claim + selected sentence (from Part 1)
- **Output**: Prediction (SUPPORTS/REFUTES/NOT ENOUGH INFO) with confidence scores
- **Graph Structure**:
  - Nodes: Tokens from claim and sentence
  - Edges: Syntactic dependencies (from spaCy dependency parser)

## Pipeline Flow

```
Claim + Article Sentences
         ↓
    [Part 1: Sentence Ranker]
         ↓
    Sentence Index → Selected Sentence
         ↓
    [Part 2: GAT Fact Verifier]
         ↓
    SUPPORTS / REFUTES / NOT ENOUGH INFO (with confidence)
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Training Part 1: Sentence Ranker

```bash
python sentence_ranker.py
```

**Expected input**: `data/sentence_ranker/train.jsonl` and `test.jsonl`

**Data format**:

```json
{
  "claim": "String describing the claim",
  "sentence": "String of the sentence to evaluate",
  "label": 0 or 1  // 0=NOT_VERIFIABLE, 1=VERIFIABLE
}
```

### Training Part 2: Fact Verifier (GAT)

```bash
python main.py
```

**Expected input**: `data/fever/train_formatted_cleaned.jsonl` and `test_formatted_cleaned.jsonl`

**Data format**:

```json
{
  "id": "claim_id",
  "claim": "String describing the claim",
  "label": "SUPPORTS|REFUTES|NOT ENOUGH INFO",
  "evidence": [...],
  "articles": {...}
}
```

### Using the Complete Pipeline

```python
from main import verify_claim

result = verify_claim(
    sentence_ranker_path="outputs/bert-sentence-ranker",
    gat_verifier_path="outputs/gat-fact-verifier/gat_verifier.pt",
    claim="The Earth is round",
    sentences=["The Earth is approximately spherical.", "The Sun is hot."]
)

print(result)
# Output:
# {
#   "selected_sentence_idx": 0,
#   "sentence": "The Earth is approximately spherical.",
#   "prediction": "SUPPORTS",
#   "confidence": 0.95,
#   "scores": {
#     "SUPPORTS": 0.95,
#     "REFUTES": 0.03,
#     "NOT ENOUGH INFO": 0.02
#   }
# }
```

## Model Configuration

### Sentence Ranker

- Model: `bert-base-uncased`
- Max length: 384 tokens
- Batch size: 8 (train), 8 (eval)
- Learning rate: 2e-5
- Epochs: 1.0

### GAT Fact Verifier

- Model: BERT embeddings + GAT
- Hidden dimension: 768
- Attention heads: 8
- GAT layers: 2
- Dropout: 0.1
- Max length: 384 tokens
- Batch size: 8 (train), 8 (eval)
- Learning rate: 2e-5
- Epochs: 1

## Features

- **Syntactic Understanding**: Uses dependency parsing to capture semantic relationships
- **Explainability**: Graph attention weights can show which sentence parts matter
- **Two-stage Approach**: Reduces noise by first selecting relevant sentences
- **Confidence Scores**: Returns probability distributions over all classes

## Output Structure

```python
{
  "selected_sentence_idx": int,      # Index of selected sentence
  "sentence": str,                    # The actual selected sentence
  "prediction": str,                  # SUPPORTS|REFUTES|NOT ENOUGH INFO
  "confidence": float,                # Confidence score (0-1)
  "scores": {                         # Full probability distribution
    "SUPPORTS": float,
    "REFUTES": float,
    "NOT ENOUGH INFO": float
  }
}
```

## Dependencies

Key dependencies:

- **transformers**: For BERT models
- **torch**: Deep learning framework
- **torch-geometric**: Graph neural networks
- **spacy**: Natural language processing & dependency parsing
- **datasets**: Hugging Face datasets library
- **networkx**: Graph manipulation

See `requirements.txt` for complete list.

## GPU Requirements

Both models require a CUDA-capable GPU. The code will automatically detect and use available GPUs.

To verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## File Structure

```
fact-checker/
├── main.py                 # Part 2: GAT fact verifier (training entry point)
├── sentence_ranker.py      # Part 1: Sentence ranker model
├── gat_verifier.py         # Part 2: GAT model implementation
├── dataset.py              # Dataset utilities
├── cleanup_formatted.py    # Data preparation utilities
├── requirements.txt        # Python dependencies
├── data/
│   ├── fever/              # FEVER dataset
│   └── sentence_ranker/    # Sentence ranker training data
└── outputs/
    ├── bert-sentence-ranker/  # Trained sentence ranker
    └── gat-fact-verifier/     # Trained GAT verifier
```
