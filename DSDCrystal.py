

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention, SAGEConv
from torch_geometric.data import Data

class SEDenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(SEDenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.se = SEBlock(growth_rate)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.se(out)
        out = torch.cat([x, out], 1)
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ProteinClassifier(nn.Module):
    def __init__(self, num_features=21, num_classes=1, growth_rate=32, num_blocks=4, attention_heads=4, graph_layers=2):
        super(ProteinClassifier, self).__init__()
        self.num_features = num_features    
        # Initial conv layer
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Dense blocks with Squeeze-and-Excitation
        num_channels = 32
        self.dense_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.dense_blocks.add_module("dense_block_{}".format(i), SEDenseLayer(num_channels, growth_rate))
            num_channels += growth_rate
        
        # Graph processing layers using Graph Attention Networks (or other Graph Neural Network)
        self.graph_layers = graph_layers
        self.gat_layers = nn.ModuleList()
        self.gat_in_channels = num_features  # Adjust if you want to use different input features for graph
        for _ in range(graph_layers):
            self.gat_layers.append(GATConv(self.gat_in_channels, 64, heads=attention_heads))  # Use more heads for GAT
            self.gat_in_channels = 64 * attention_heads
        
        # Global attention pool
        self.attention_pool = GlobalAttention(gate_nn=nn.Linear(self.gat_in_channels, 1))
        
        # Structural feature processing
        self.linear_structural = nn.Linear(62, 32)
        
        # Final classifier layers
        self.fc1 = nn.Linear(num_channels + self.gat_in_channels + 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, sequence, graph_data, structural_features, batch_index):
        x_seq = sequence.permute(0, 2, 1)  # Change (batch_size, seq_len, channels) to (batch_size, channels, seq_len)
        out = F.relu(self.bn1(self.conv1(x_seq)))
        out = self.dense_blocks(out)
        out_pooled = F.adaptive_avg_pool1d(out, 1).squeeze(2)  # Reducing the dimension from [64, 160, 800] to [64, 160]
   
        # Process graph
        x_graph = graph_data.x 
        edge_index = graph_data.edge_index
        for gat_layer in self.gat_layers:
            x_graph = F.relu(gat_layer(x_graph, edge_index))
        x_graph_pooled = self.attention_pool(x_graph, batch_index)
        
        # Process structural features
        x_structural = F.relu(self.linear_structural(structural_features))

        # Concatenate all features
        x_concat = torch.cat((out_pooled,x_graph_pooled, x_structural), dim=1)
        
        # Final classification layers
        x = F.relu(self.fc1(x_concat))
        x = torch.sigmoid(self.fc2(x))
        
        return x

# Example usage
model = ProteinClassifier()
print(model)
