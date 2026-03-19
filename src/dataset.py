import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from config import RobertaConfig, LABEL_MAP

# Dataset for stage 1, relevance scoring
class FeverStage1RobertaDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: RobertaTokenizer, max_length: int):
        # store tokenizer and max_length
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for i, row in enumerate(data):
            if i % 5000 == 0:
                print(f"Building dataset: {i}/{len(data)} claims processed", flush=True)

            claim = row["claim"]
            label = row["label"]

            gold = set()
            # get list of gold sentences where they exist
            if label != "NOT ENOUGH INFO":
                for evidence in row["evidence"]:
                    gold.add((evidence["doc_id"], evidence["sentence_id"]))

            # add all sentences from articles as samples, labeling as 1 if in gold, else 0
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
        # run sample through tokenizer
        encoding = self.tokenizer(
            sample["claim"],
            sample["sentence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # return input_ids, attention_mask, label, claim_id, doc_id, sent_id
        return {
            "input_ids": encoding["input_ids"].squeeze(0), #type: ignore
            "attention_mask": encoding["attention_mask"].squeeze(0), #type: ignore
            "label": torch.tensor(sample["label"], dtype=torch.float),
            "claim_id": sample["claim_id"],
            "doc_id": sample["doc_id"],
            "sent_id": sample["sent_id"]
        }

# collate function to stack samples together for stage 1
def roberta_collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "claim_id": [b["claim_id"] for b in batch],
        "doc_id": [b["doc_id"] for b in batch],
        "sent_id": [b["sent_id"] for b in batch],
    }

class RobertaStage2Dataset(Dataset):
    def __init__(self, embeddings: list[dict], tokenizer, config: RobertaConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.samples = []

        for result in embeddings:
            claim = result["claim"]
            label = LABEL_MAP[result["label"]]
            candidates = result["candidates"]

            if candidates:
                # sort and take top-k candidates by probability
                sorted_candidates = sorted(candidates, key=lambda x: x["prob"], reverse=True)
                top_k = sorted_candidates[:config.top_k]

                # concatenate all top-k sentences
                #evidence = " ".join(c["sentence"] for c in top_k)
                
                # reverse order so most relevant is closest to claim
                top_k = sorted_candidates[:config.top_k][::-1]
                evidence = " ".join(c["sentence"] for c in top_k)
            else:
                # if no candidates, use NO_EVIDENCE as placeholder
                evidence = "NO_EVIDENCE"

            self.samples.append({
                "claim": claim,
                "evidence": evidence,
                "label": label
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # tokenize claim and evidence together
        encoding = self.tokenizer(
            s["claim"],
            s["evidence"],
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # return input_ids, attention_mask, label
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(s["label"], dtype=torch.long)
        }