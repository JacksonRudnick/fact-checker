import torch.nn as nn
from transformers import RobertaModel
from config import RobertaConfig

class RobertaVerifier(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(768, 3)

        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # unfreeze last x layers
        for param in self.roberta.encoder.layer[-config.stage2_unfreeze_layers:].parameters():
            param.requires_grad = True
        for param in self.roberta.pooler.parameters(): # type: ignore
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls)