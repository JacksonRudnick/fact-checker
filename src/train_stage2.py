import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import RobertaTokenizer

from config import RobertaConfig, GatConfig, LABEL_MAP, TransformerConfig
from gat_model import GATFactVerifier
from dataset import RobertaStage2Dataset, TransformerStage2Dataset
from roberta_verifier_model import RobertaVerifier
from transformer_model import TransformerFactVerifier



def result_to_graph(result: dict, config: RobertaConfig, gat_config: GatConfig) -> Data | None:
    label = LABEL_MAP[result["label"]]
    claim = result["claim"]
    claim_embedding = result["claim_embedding"]
    candidates = result["candidates"]

    if claim_embedding is None:
        return None

    if candidates:
        sorted_candidates = sorted(candidates, key=lambda x: x["prob"], reverse=True)
        top_k = sorted_candidates[:config.top_k]
        top_k_embeddings = torch.stack([c["embedding"] for c in top_k])
        top_k_sentences = [c["sentence"] for c in top_k]
        x = torch.cat([claim_embedding, top_k_embeddings], dim=0)
    else:
        x = claim_embedding
        top_k_sentences = []

    edge_src = []
    edge_dst = []

    if top_k_sentences:
        claim_doc = gat_config.nlp(claim)
        for sent_idx_a, sentence_a in enumerate(top_k_sentences):
            for sent_idx_b, sentence_b in enumerate(top_k_sentences):
                if sent_idx_a >= sent_idx_b:
                    continue
                doc_a = gat_config.nlp(sentence_a)
                doc_b = gat_config.nlp(sentence_b)
                node_a = sent_idx_a + 1
                node_b = sent_idx_b + 1
                for token_a in doc_a:
                    for token_b in doc_b:
                        if token_a.text.lower() == token_b.text.lower():
                            if not token_a.is_stop:
                                edge_src.append(node_a)
                                edge_dst.append(node_b)
                                edge_src.append(node_b)
                                edge_dst.append(node_a)
                                break

    total_nodes = x.size(0)
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        valid_mask = (edge_index[0] < total_nodes) & (edge_index[1] < total_nodes)
        edge_index = edge_index[:, valid_mask]
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long)
    )


def build_gat_dataset(embeddings: list[dict], config: RobertaConfig, gat_config: GatConfig, cache_path: str) -> list[Data]:
    cache_path = Path(cache_path) # type: ignore
    if cache_path.exists(): # type: ignore
        print(f"Loading cached graphs from {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)

    graphs = []
    for i, result in enumerate(embeddings):
        if i % 1000 == 0:
            print(f"Building graphs: {i}/{len(embeddings)}", flush=True)
        graph = result_to_graph(result, config, gat_config)
        if graph is not None:
            graphs.append(graph)

        if i % 10000 == 0 and i > 0:
            torch.save(graphs, cache_path)
            print(f"Checkpoint saved at {i}", flush=True)

    torch.save(graphs, cache_path)
    print(f"Saved {len(graphs)} graphs to {cache_path}", flush=True)
    return graphs


def train_gat(roberta_config: RobertaConfig, gat_config: GatConfig, device: torch.device, train_embeddings: list[dict], test_embeddings: list[dict]):
    train_graphs = build_gat_dataset(train_embeddings, roberta_config, gat_config, gat_config.train_cache_path)
    test_graphs = build_gat_dataset(test_embeddings, roberta_config, gat_config, gat_config.eval_cache_path)

    print(f"Train graphs: {len(train_graphs)}", flush=True)
    print(f"Test graphs: {len(test_graphs)}", flush=True)

    model = GATFactVerifier(gat_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=gat_config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=gat_config.epochs)

    train_loader = PyGDataLoader(train_graphs, batch_size=gat_config.train_batch_size, shuffle=True)
    test_loader = PyGDataLoader(test_graphs, batch_size=gat_config.eval_batch_size, shuffle=False)

    for epoch in range(gat_config.epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.squeeze(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 1000 == 0:
                print(f"  Batch {i}/{len(train_loader)} — Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        print(f"Epoch {epoch+1}/{gat_config.epochs}, Loss: {total_loss/len(train_loader):.4f}")
        evaluate_gat(model, test_loader, device)

    Path(gat_config.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(gat_config.output_dir) / "gat_verifier.pt"
    torch.save(model.state_dict(), output_path)
    print(f"Saved GAT verifier to: {output_path}")


def evaluate_gat(model: GATFactVerifier, test_loader: PyGDataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = batch.y.squeeze(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

def train_roberta_stage2(roberta_config: RobertaConfig, device: torch.device,  train_embeddings: list[dict], test_embeddings: list[dict]):
    tokenizer = RobertaTokenizer.from_pretrained(roberta_config.model_name)

    # train on retrieved sentences
    train_dataset = RobertaStage2Dataset(train_embeddings, tokenizer, roberta_config)

    # test on retrieved sentences
    test_dataset = RobertaStage2Dataset(test_embeddings, tokenizer, roberta_config)

    train_loader = DataLoader(train_dataset, batch_size=roberta_config.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=roberta_config.eval_batch_size, shuffle=False)

    model = RobertaVerifier(roberta_config).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.roberta.encoder.layer[-4:].parameters(), "lr": roberta_config.learning_rate},
        {"params": model.roberta.pooler.parameters(), "lr": roberta_config.learning_rate}, # type: ignore
        {"params": model.classifier.parameters(), "lr": roberta_config.learning_rate * 10},
    ])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=roberta_config.epochs)

    for epoch in range(roberta_config.epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 1000 == 0:
                print(f"  Batch {i}/{len(train_loader)} — Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        print(f"Epoch {epoch+1}/{roberta_config.epochs}, Loss: {total_loss/len(train_loader):.4f}")
        evaluate_stage2(model, test_loader, device)

    Path(roberta_config.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(roberta_config.output_dir) / "stage2.pt")
    print(f"Saved Stage 2 to {roberta_config.output_dir}/stage2.pt")


def evaluate_stage2(model, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")