import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer


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