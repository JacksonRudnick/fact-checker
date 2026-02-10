import torch
from torch.utils.data import Dataset
import os

from fetch import FetchDataset

class FeverDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        self.data = data.get_data()

        self.data_len = len(self.data)

    def __len__(self):
        return self.data

    def __getitem__(self, index):
        return self.data[index]

if __name__ == "__main__":
    test = FetchDataset('data/fever/paper_test.jsonl')

    test_dataset = FeverDataset(test)

    print(len(test_dataset))
    print(test_dataset[0])
