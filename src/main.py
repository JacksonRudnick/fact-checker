from pathlib import Path

from config import MainConfig, RobertaConfig
from train_stage1 import train_roberta
from inference import run_roberta_inference, load_roberta_model
from train_stage2 import train_roberta_stage2
from util import load_jsonl, load_embeddings, load_cuda

def main():
    main_config = MainConfig()
    roberta_config = RobertaConfig()
    roberta2_config = RobertaConfig()

    roberta2_config.output_dir = "outputs/roberta-fact-verifier-stage2"

    load_cuda(main_config)
    device = main_config.device

    train_data = load_jsonl(Path(main_config.train_path))
    test_data = load_jsonl(Path(main_config.test_path))

    print(f"Train rows: {len(train_data)}", flush=True)
    print(f"Test rows: {len(test_data)}", flush=True)

    train_embeddings_path = Path(roberta_config.output_dir) / "stage1_train_embeddings.pkl"
    test_embeddings_path = Path(roberta_config.output_dir) / "stage1_test_embeddings.pkl"

    # Stage 1 - training Roberta
    model, tokenizer = train_roberta(roberta_config, device, train_data, test_data)

    # Stage 1 - Roberta Inference
    model, tokenizer = load_roberta_model(roberta_config, device)
    run_roberta_inference(model, train_data, tokenizer, roberta_config, device, train_embeddings_path)
    run_roberta_inference(model, test_data, tokenizer, roberta_config, device, test_embeddings_path)

    # Stage 2 - Transformer over retrieved embeddings
    train_embeddings = load_embeddings(train_embeddings_path)
    test_embeddings = load_embeddings(test_embeddings_path)
    train_roberta_stage2(roberta_config, device,  train_embeddings, test_embeddings)

if __name__ == "__main__":
    main()