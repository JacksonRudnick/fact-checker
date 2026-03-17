import torch
from torch.distributions import Transform
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer
from config import RobertaConfig, LABEL_MAP


class FeverStage1BertDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: BertTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for i, row in enumerate(data):
            if i % 1000 == 0:
                print(f"Building dataset: {i}/{len(data)} claims processed", flush=True)

            claim = row["claim"]
            label = row["label"]

            gold = set()
            if label != "NOT ENOUGH INFO":
                for evidence in row["evidence"]:
                    gold.add((evidence["doc_id"], evidence["sentence_id"]))

            for doc_id, sentences in row["articles"].items():
                for sent_id, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                    self.samples.append({
                        "claim": claim,
                        "sentence": sentence,
                        "label": 1 if (doc_id, sent_id) in gold else 0,
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "claim_id": row["id"]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["claim"],
            sample["sentence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0), #type: ignore
            "attention_mask": encoding["attention_mask"].squeeze(0), #type: ignore
            "token_type_ids": encoding["token_type_ids"].squeeze(0), #type: ignore
            "label": torch.tensor(sample["label"], dtype=torch.float),
            "claim_id": sample["claim_id"],
            "doc_id": sample["doc_id"],
            "sent_id": sample["sent_id"]
        }


class FeverStage1RobertaDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: RobertaTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for i, row in enumerate(data):
            if i % 1000 == 0:
                print(f"Building dataset: {i}/{len(data)} claims processed", flush=True)

            claim = row["claim"]
            label = row["label"]

            gold = set()
            if label != "NOT ENOUGH INFO":
                for evidence in row["evidence"]:
                    gold.add((evidence["doc_id"], evidence["sentence_id"]))

            for doc_id, sentences in row["articles"].items():
                for sent_id, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                    self.samples.append({
                        "claim": claim,
                        "sentence": sentence,
                        "label": 1 if (doc_id, sent_id) in gold else 0,
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "claim_id": row["id"]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["claim"],
            sample["sentence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0), #type: ignore
            "attention_mask": encoding["attention_mask"].squeeze(0), #type: ignore
            "label": torch.tensor(sample["label"], dtype=torch.float),
            "claim_id": sample["claim_id"],
            "doc_id": sample["doc_id"],
            "sent_id": sample["sent_id"]
        }


def bert_collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "claim_id": [b["claim_id"] for b in batch],
        "doc_id": [b["doc_id"] for b in batch],
        "sent_id": [b["sent_id"] for b in batch],
    }


def roberta_collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "claim_id": [b["claim_id"] for b in batch],
        "doc_id": [b["doc_id"] for b in batch],
        "sent_id": [b["sent_id"] for b in batch],
    }


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: list[dict], config: RobertaConfig):
        self.samples = []
        max_nodes = config.top_k + 1  # claim + top_k sentences

        for result in embeddings:
            label = LABEL_MAP[result["label"]]
            claim_embedding = result["claim_embedding"]
            candidates = result["candidates"]

            if claim_embedding is None:
                continue

            if candidates:
                sorted_candidates = sorted(candidates, key=lambda x: x["prob"], reverse=True)
                top_k = sorted_candidates[:config.top_k]
                top_k_embeddings = torch.stack([c["embedding"] for c in top_k])
                x = torch.cat([claim_embedding, top_k_embeddings], dim=0)
            else:
                x = claim_embedding

            # pad to max_nodes
            if x.size(0) < max_nodes:
                padding = torch.zeros(max_nodes - x.size(0), x.size(1))
                x = torch.cat([x, padding], dim=0)

            self.samples.append({"x": x, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class TransformerStage2Dataset(Dataset):
    def __init__(self, embeddings: list[dict], config: RobertaConfig):
        self.samples = []
        for result in embeddings:
            claim_embedding = result["claim_embedding"]
            candidates = result["candidates"]
            label = LABEL_MAP[result["label"]]

            if claim_embedding is None:
                continue

            if candidates:
                sorted_candidates = sorted(candidates, key=lambda x: x["prob"], reverse=True)
                top_k = sorted_candidates[:config.top_k]
                evidence_embeddings = [c["embedding"] for c in top_k]
            else:
                evidence_embeddings = []

            self.samples.append({
                "claim_embedding": claim_embedding.squeeze(0),
                "evidence_embeddings": evidence_embeddings,
                "label": label,
                "num_evidence": len(evidence_embeddings)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def transformer_collate_fn(batch, top_k: int):
    max_seq = top_k + 1  # claim + top_k evidence
    d_model = 768

    xs = []
    masks = []
    labels = []

    for sample in batch:
        claim = sample["claim_embedding"]  # [768]
        evs = sample["evidence_embeddings"]  # list of [768] tensors
        
        seq = [claim] + evs
        seq_len = len(seq)
        
        # pad to max_seq
        while len(seq) < max_seq:
            seq.append(torch.zeros(d_model))
        
        x = torch.stack(seq[:max_seq])  # [max_seq, 768]
        
        # padding mask: True = ignore this position
        mask = torch.zeros(max_seq, dtype=torch.bool)
        mask[seq_len:] = True
        
        xs.append(x)
        masks.append(mask)
        labels.append(sample["label"])

    return {
        "x": torch.stack(xs),           # [batch, max_seq, 768]
        "mask": torch.stack(masks),      # [batch, max_seq]
        "label": torch.tensor(labels, dtype=torch.long)
    }