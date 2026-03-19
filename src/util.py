import json
import pickle
import torch
from pathlib import Path

from config import MainConfig

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