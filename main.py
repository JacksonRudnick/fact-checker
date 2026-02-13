import torch
from transformers import pipeline, BertLMHeadModel, BertTokenizer

from dataset import FeverDataset
from fetch import FetchDataset

if __name__ == "__main__":
    # Load model and tokenizer
    #model = BertLMHeadModel.from_pretrained('bert-base-uncased')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # load data
    #train_data = FetchDataset('data/fever/train.jsonl')
    test_data = FetchDataset('data/fever/paper_test.jsonl')

    #train_dataset = FeverDataset(train_data)
    test_dataset = FeverDataset(test_data)

    #x_train = [item['claim'] for item in train_dataset]
    #y_train = [item['evidence'][0][0] for item in train_dataset]

    claim_test = [item['claim'] for item in test_dataset]
    x_test = [item['articles'] for item in test_dataset]
    y_test = [item['evidence'][0][0][1] for item in test_dataset]

    print(claim_test[:5])
    print([words[:5] for words in x_test[:5][0] if x_test[0] is not None])
    print(y_test[:5])
