import torch.nn as nn
from transformers import RobertaModel
from config import RobertaConfig


class RobertaRelevanceScorer(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(768, 1)

        for param in self.roberta.parameters():
            param.requires_grad = False

        for param in self.roberta.encoder.layer[-3:].parameters():
            param.requires_grad = True

        for param in self.roberta.pooler.parameters():  # type: ignore
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)
        return self.classifier(cls_token)

    def get_embeddings(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]
    