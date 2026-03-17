import json
import pickle
import torch
from pathlib import Path

from config import MainConfig, BertConfig, RobertaConfig, GatConfig, TransformerConfig
from train_stage1 import train_bert, train_roberta
from inference import run_bert_inference, run_roberta_inference, load_bert_model, load_roberta_model
from train_stage2 import train_roberta_stage2


def load_cuda(main_config: MainConfig):
    if torch.cuda.is_available():
        main_config.device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(main_config.device)}")
    else:
        print("CUDA is not available.")
        exit(-1)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_embeddings(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    main_config = MainConfig()
    bert_config = BertConfig()
    roberta_config = RobertaConfig()
    gat_config = GatConfig()
    trans_config = TransformerConfig()
    roberta2_config = RobertaConfig()

    roberta2_config.output_dir = "outputs/roberta-fact-verifier-stage2"

    load_cuda(main_config)
    device = main_config.device

    #train_data = load_jsonl(Path(main_config.train_path))
    #test_data = load_jsonl(Path(main_config.test_path))

    #print(f"Train rows: {len(train_data)}", flush=True)
    #print(f"Test rows: {len(test_data)}", flush=True)

    train_embeddings_path = Path(roberta_config.output_dir) / "stage1_train_embeddings.pkl"
    test_embeddings_path = Path(roberta_config.output_dir) / "stage1_test_embeddings.pkl"

    # Roberta
    #model, tokenizer = train_roberta(main_config, roberta_config, device, train_data, test_data)

    # Roberta Inference
    #model, tokenizer = load_roberta_model(roberta_config, device)
    #run_roberta_inference(model, train_data, tokenizer, roberta_config, device, train_embeddings_path)
    #run_roberta_inference(model, test_data, tokenizer, roberta_config, device, test_embeddings_path)

    # Stage 2 — Transformer over retrieved embeddings
    train_embeddings = load_embeddings(train_embeddings_path)
    test_embeddings = load_embeddings(test_embeddings_path)
    train_roberta_stage2(roberta_config, device,  train_embeddings, test_embeddings)


if __name__ == "__main__":
    main()