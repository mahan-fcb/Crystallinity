import torch, numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GCNConv,
    DiffGroupNorm
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
import torch
from torch import nn
import numpy as np
import pandas as pd

import torch
from torch import nn


# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

class ResidualBlock(nn.Module):
    def __init__(self, filters, d_rate):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.d_rate = d_rate
        self.shortcut = None

        self.bn1 = nn.BatchNorm1d(filters)
        self.conv1 = nn.Conv1d(filters, filters, kernel_size=1, dilation=d_rate, padding=d_rate)
        self.bn2 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=3, padding=1)
        self.se_block = SEBlock(filters)

    def forward(self, x):
        shortcut = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.se_block(out)
        out += shortcut
        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        out = self.pool(x)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.unsqueeze(-1)
        return out * x
class MLP(nn.Module):
    """Multi-layer perceptron with custom activation functions.

    Args:
        hs (list of int): Input, hidden, and output dimensions.
        act (torch activation function, or None): Activation function that
            applies to all but the output layer. For example, `torch.nn.ReLU()`.
            If None, no activation function is applied.
    """
    def __init__(self, hs, act=None):
        super().__init__()
        self.hs = hs
        self.act = act
        
        num_layers = len(hs)

        layers = []
        for i in range(num_layers-1):
            layers += [nn.Linear(hs[i], hs[i+1])]
            if (act is not None) and (i < num_layers-2):
                layers += [act]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(hs={self.hs}, act={self.act})'

class GATGNN_GIM1_globalATTENTION(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(GATGNN_GIM1_globalATTENTION, self).__init__()

        self.act       = act
        self.fc_layers = fc_layers
        if batch_track_stats      == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 

        self.batch_norm   = batch_norm
        self.dropout_rate = dropout_rate

        self.global_mlp   = torch.nn.ModuleList()
        self.bn_list      = torch.nn.ModuleList()

        assert fc_layers > 1, "Need at least 2 fc layer"   

        for i in range(self.fc_layers + 1):
            if i == 0:
                lin    = torch.nn.Linear(dim+108, dim)
                self.global_mlp.append(lin)       
            else: 
                if i != self.fc_layers :
                    lin = torch.nn.Linear(dim, dim)
                else:
                    lin = torch.nn.Linear(dim, 1)
                self.global_mlp.append(lin)   

            if self.batch_norm == "True":
                #bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)     

    def forward(self, x, batch, glbl_x ):
        out   = torch.cat([x,glbl_x],dim=-1)
        for i in range(0, len(self.global_mlp)):
            if i   != len(self.global_mlp) -1:
                out = self.global_mlp[i](out)
                out = getattr(F, self.act)(out)    
            else:
                out = self.global_mlp[i](out)   
                out = tg_softmax(out,batch)                
        return out

        x           = getattr(F, self.act)(self.node_layer1(chunk))
        x           = self.atten_layer(x)
        out         = tg_softmax(x,batch)
        return out
    


class GATGNN_AGAT_LAYER(MessagePassing):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2, **kwargs):
        super(GATGNN_AGAT_LAYER, self).__init__(aggr='add', flow='target_to_source', **kwargs)

        self.act = act
        self.fc_layers = fc_layers
        self.batch_track_stats = batch_track_stats.lower() == "true"
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.heads = 4
        self.add_bias = True
        self.neg_slope = 0.2

        self.bn1 = nn.BatchNorm1d(self.heads)
        self.W = Parameter(torch.Tensor(dim * 2, self.heads * dim))
        self.att = Parameter(torch.Tensor(1, self.heads, 2 * dim))
        self.dim = dim

        if self.add_bias:
            self.bias = Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        out_i = torch.cat([x_i, edge_attr], dim=-1)
        out_j = torch.cat([x_j, edge_attr], dim=-1)

        out_i = getattr(F, self.act)(torch.matmul(out_i, self.W))
        out_j = getattr(F, self.act)(torch.matmul(out_j, self.W))
        out_i = out_i.view(-1, self.heads, self.dim)
        out_j = out_j.view(-1, self.heads, self.dim)

        alpha = getattr(F, self.act)((torch.cat([out_i, out_j], dim=-1) * self.att).sum(dim=-1))
        alpha = getattr(F, self.act)(self.bn1(alpha))
        alpha = tg_softmax(alpha, edge_index_i)

        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        out_j = (out_j * alpha.view(-1, self.heads, 1)).transpose(0, 1)
        return out_j

    def update(self, aggr_out):
        out = aggr_out.mean(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out



class DSDCrystal(torch.nn.Module):
    def __init__(self, data,
        out_dims=64,
        d_model=512,
        N=3,
        heads=4,
        compute_device=None,
        residual_nn='roost',
        dim1=64,
        dim2=150,
        pre_fc_count=1,
        gc_count=5,
        gc_count2=5,
        post_fc_count=1,
        pool="global_add_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="softplus",
        dropout_rate=0.0,
        **kwargs
    ):
        super(DSDCrystal, self).__init__()
        
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm   = batch_norm
        self.struc_embedding = nn.Linear(54, 64)
        self.struc_bn = nn.BatchNorm1d(64)
        self.pool         = pool
        self.act          = act
        self.pool_order   = pool_order
        self.dropout_rate = dropout_rate
        self.avg = False
        self.out_dims = out_dims
        self.compute_device = compute_device
        self.global_att_LAYER = GATGNN_GIM1_globalATTENTION(dim1, act, batch_norm, batch_track_stats, dropout_rate)
        self.conv = nn.Conv1d(21, 20, kernel_size=1, padding='same')
        self.res1 = ResidualBlock(20, 2)
        self.res2 = ResidualBlock(20, 3)
        self.pool1 = nn.MaxPool1d(3)
        self.bn1 = nn.BatchNorm1d(20)
        self.dropout1 = nn.Dropout(0.5)

        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 gat layer"        
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1
        ##Determine output dimension length
        if data[0].y.ndim == 0:
            output_dim = 5
        else:
            output_dim = len(data[0].y)

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)

        self.pre_lin_list_E = torch.nn.ModuleList()  
        self.pre_lin_list_N = torch.nn.ModuleList()  

        data.num_edge_features
        for i in range(pre_fc_count):


            embed_atm = nn.Sequential(MLP([data.num_features,dim1, dim1],  act=nn.SiLU()), nn.LayerNorm(32))
            self.pre_lin_list_N.append(embed_atm)
            embed_bnd = nn.Sequential(MLP([data.num_edge_features,dim1,dim1], act=nn.SiLU()), nn.LayerNorm(32))
            self.pre_lin_list_E.append(embed_bnd)    



       
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = GATGNN_AGAT_LAYER(dim1, act, batch_norm, batch_track_stats, dropout_rate)
            self.conv_list.append(conv)
            if self.batch_norm == "True":
                bn = DiffGroupNorm(gc_dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim*2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)   

        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)
    def forward(self, data):

        for i in range(0, len(self.pre_lin_list_N)):
 
             out_x = self.pre_lin_list_N[i](data.x)

             out_x = F.softplus(out_x)
             out_e = self.pre_lin_list_E[i](data.edge_attr)
             #out_e = getattr(F, 'prelu')(out_e,out_e)
             out_e = F.softplus(out_e)
      
        prev_out_x = out_x    
        

        for i in range(0, len(self.conv_list)):

            out_x = self.conv_list[i](out_x, data.clone().edge_index, out_e)
            out_x = self.bn_list[i](out_x)

           
        out_x = torch.add(out_x, prev_out_x)
        out_x = F.dropout(out_x, p=self.dropout_rate, training=self.training)
        prev_out_x = out_x
    

        out_x = getattr(torch_geometric.nn, self.pool)(out_x, data.batch)
        for i in range(0, len(self.post_lin_list)):
            out_x = self.post_lin_list[i](out_x)
            out_x = getattr(F, self.act)(out_x)


        x = self.conv(data.x_seq)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        struc_fea = self.human_embedding(data.struc)
        struc_fea = self.human_bn(struc_fea)
        out_x = torch.cat((out_x,struc_fea,out_x),1)
        out       = self.lin_out(out_x)
       # print(out.shape)
        if out.shape[1] == 5:
            return out.view(-1)
        else:
            #print(out)
            return out
        

