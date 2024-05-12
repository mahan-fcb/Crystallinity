from torch_geometric.nn import GATConv, GlobalAttention

class ProteinClassifier(nn.Module):
    def __init__(self, num_features=21, num_classes=1, growth_rate=32, num_blocks=4, return_attn_weights=False):
        super(ProteinClassifier, self).__init__()
        self.num_features = num_features
        self.return_attn_weights = return_attn_weights
        
        # Initialize layers as before
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        num_channels = 32
        self.dense_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.dense_blocks.add_module("dense_block_{}".format(i), SEDenseLayer(num_channels, growth_rate))
            num_channels += growth_rate
        
        self.gat1 = GATConv(in_channels=21, out_channels=64, heads=4)
        self.gat2 = GATConv(in_channels=64*4, out_channels=128, heads=1, return_attention_weights=return_attn_weights)
        
        self.attention_pool = GlobalAttention(gate_nn=nn.Linear(128, 1))
        self.linear_structural = nn.Linear(62, 32)
        self.fc1 = nn.Linear(num_channels + 128 + 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, sequence, graph_data, structural_features, batch_index):
        x_seq = sequence.permute(0, 2, 1)
        out = F.relu(self.bn1(self.conv1(x_seq)))
        out = self.dense_blocks(out)
        out_pooled = F.adaptive_avg_pool1d(out, 1).squeeze(2)
        
        x_graph, edge_index = graph_data.x, graph_data.edge_index
        x_graph = F.relu(self.gat1(x_graph, edge_index))
        
        if self.return_attn_weights:
            x_graph, attn_weights = self.gat2(x_graph, edge_index)
        else:
            x_graph = F.relu(self.gat2(x_graph, edge_index))[0]
        
        x_graph_pooled = self.attention_pool(x_graph, batch_index)
        x_structural = F.relu(self.linear_structural(structural_features))
        x_concat = torch.cat((out_pooled, x_graph_pooled, x_structural), dim=1)
        x = F.relu(self.fc1(x_concat))
        x = torch.sigmoid(self.fc2(x))
        
        if self.return_attn_weights:
            return x, attn_weights
        return x
