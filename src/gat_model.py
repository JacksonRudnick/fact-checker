import torch.nn as nn
from torch_geometric.nn.models import GAT
from torch_geometric.nn import global_mean_pool
from config import GatConfig


class GATFactVerifier(nn.Module):
    def __init__(self, config: GatConfig):
        super().__init__()
        self.gat = GAT(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.hidden_channels,
            heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        self.classifier = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.gat(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)