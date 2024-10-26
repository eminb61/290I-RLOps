import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch.nn as nn

class GATv2Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads=6):
        super(GATv2Encoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=heads, edge_dim=edge_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, edge_dim=edge_dim))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if x.dim() == 3:
            x = x.view(-1, x.size(-1)) # Flatten to [num_nodes, num_node_features]
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index, edge_attr))
        x = self.convs[-1](x, edge_index, edge_attr)
        return x