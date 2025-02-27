import torch_geometric.nn as geom_nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, GRU

gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}


class DyFraudNet(nn.Module):
    def __init__(self, input_dim, memory_size=16, hidden_size=16, out_put_size=2, gnn_type="GCN", num_layers=2,
                 dropout=0.0):
        super().__init__()
        gnn_layer = gnn_layer_by_name[gnn_type]
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, hidden_size)
        self.conv1 = EvolveGNN_O(hidden_size, memory_size, hidden_size)
        self.conv2 = EvolveGNN_O(hidden_size, memory_size, hidden_size)
        self.postprocessing1 = geom_nn.Linear(hidden_size, 2)
        self.dropout = dropout
        # self.memory_weights = [torch.tensor([0.0 for _ in range(memory_size)]),
        #                        torch.tensor([0.0 for _ in range(memory_size)])]

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocessing1.reset_parameters()

    def forward(self, x, edge_index):
        # Preprocess step
        h = self.preprocess1(x)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.conv1(h, edge_index)

        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.postprocessing1(h)
        h = torch.sum(h, dim=-1)
        return h


class EvolveGNN_O(nn.Module):
    def __init__(self, in_channels, memory_size, out_channels, gnn_type="GCN"):
        super().__init__()
        gnn_layer = gnn_layer_by_name[gnn_type]
        self.gnn = gnn_layer(in_channels, out_channels)
        self.gru = nn.GRU(memory_size, memory_size)
        self.weight_transform = nn.Linear(memory_size, in_channels * out_channels)
        self.register_buffer("memory_weights", torch.zeros(memory_size))

    def forward(self, x, edge_index):
        memory = self.memory_weights.to(x.device).unsqueeze(0)
        update_memory, _ = self.gru(memory)
        update_memory = update_memory.squeeze(0)
        new_weights = self.weight_transform(update_memory)
        new_weights = new_weights.view(self.gnn.lin.weight.shape)

        with torch.no_grad():
            self.gnn.lin.weight.copy_(new_weights)

        self.memory_weights.copy_(update_memory)
        out = self.gnn(x, edge_index)
        return out
