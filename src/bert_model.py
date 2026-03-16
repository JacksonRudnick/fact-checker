import torch.nn as nn
from transformers import BertModel
from config import BertConfig


class BertRelevanceScorer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(768, 1)

        for param in self.bert.parameters():
            param.requires_grad = False

        for param in self.bert.encoder.layer[-3:].parameters():
            param.requires_grad = True

        for param in self.bert.pooler.parameters():  # type: ignore
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)
        return self.classifier(cls_token)

    def get_embeddings(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.last_hidden_state[:, 0, :]