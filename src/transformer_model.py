import torch
import torch.nn as nn

from config import TransformerConfig

class TransformerFactVerifier(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.in_channels,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_channels * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Linear(config.in_channels, config.out_channels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, 768]
        # first token is claim, rest are evidence sentences
        x = self.transformer(x)
        x = x[:, 0, :]  # use claim token output for classification
        x = self.dropout(x)
        x = self.classifier(x)
        return x