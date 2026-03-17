import torch
import torch.nn as nn
from config import TransformerConfig

class TransformerFactVerifier(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes)
        )

    def forward(self, x, padding_mask=None):
        # x: [batch_size, seq_len, 768]
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = x[:, 0, :]  # take claim token (position 0)
        return self.classifier(x)