import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules import LayerNorm

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    return
def build_linear(input_dim, output_dim, q_noise, qn_block_size):
    return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )
class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, quant_noise, qn_block_size, args):
        super().__init__()
        self.fc1 = build_linear(in_dim, hidden_dim, quant_noise, qn_block_size)
        self.fc2 = build_linear(hidden_dim, out_dim, quant_noise, qn_block_size)
        self.phrase_activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        ) #torch.sigmoid
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
    def forward(self, x):
        x = self.phrase_activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x
class EdgeConv(MessagePassing):
    def __init__(self, F_in, F_out, quant_noise, qn_block_size, args):
        super(EdgeConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.mlp = FeedForward(2*F_in, F_out, F_out, quant_noise, qn_block_size, args)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(edge_features)

class GraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, 
                quant_noise, qn_block_size, args,
                reducer='mean', normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        self.lin = build_linear(in_channels, out_channels, quant_noise, qn_block_size, args)
        self.agg_lin = build_linear(in_channels, out_channels, quant_noise, qn_block_size, args)
        init_weights(self.lin)
        init_weights(self.agg_lin)
        if normalize_embedding:
            self.normalize_emb = True
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        out = self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)
        out = self.agg_lin(out)
        out = self.lin(x) + out
        return out
    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j
    def update(self, aggr_out):
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, 2, dim=-1)
        return aggr_out

class GAT(MessagePassing):
    def __init__(self, in_channels, out_channels, 
                quant_noise, qn_block_size, args,
                num_heads=1, concat=True,
                bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat 
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.lin = build_linear(self.in_channels, self.heads * self.out_channels, quant_noise, qn_block_size, args)
        self.att = nn.Parameter(torch.Tensor(self.heads, 2 * self.out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
    def forward(self, x, edge_index, size=None):
        x = self.lin(x)
        return self.propagate(edge_index, size=size, x=x)
    def message(self, edge_index_i, x_i, x_j, size_i):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_cat = torch.cat([x_i, x_j], dim=-1)
        alpha = F.leaky_relu((x_cat * self.att).sum(-1), 0.2)
        alpha = pyg_utils.softmax(alpha, edge_index_i, num_nodes=size_i)

        alpha = self.dropout_module(alpha)

        return (x_j * alpha.unsqueeze(-1)).view(-1, self.heads * self.out_channels)
    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class UCCAEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, args):
        super(UCCAEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.num_layers = 3 # hard-code
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.graph_type = getattr(args, 'graph_type', None)
        if graph_type == "GAT":
            Model = GAT
            settings_first = (in_dim, hidden_dim, self.quant_noise, self.quant_noise_block_size, args, 8)
            settings_else = (hidden_dim, hidden_dim, self.quant_noise, self.quant_noise_block_size, args, 8)
        elif graph_type == "GraphSage":
            Model = GraphSage
            settings_first = (in_dim, hidden_dim, self.quant_noise, self.quant_noise_block_size, args)
            settings_else = (hidden_dim, hidden_dim, self.quant_noise, self.quant_noise_block_size, args)
        else:
            Model = EdgeConv
            settings_first = (in_dim, hidden_dim, self.quant_noise, self.quant_noise_block_size, args)
            settings_else = (hidden_dim, hidden_dim, self.quant_noise, self.quant_noise_block_size, args)

        self.convs_layer_norm = LayerNorm(self.in_dim)
        self.ffn_layer_norm = LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(Model(*settings_first))
        for i in range(self.num_layers-1):
            self.convs.append(Model(*settings_else))

        self.ffn = FeedForward(hidden_dim, hidden_dim, out_dim, self.quant_noise, self.quant_noise_block_size, args)

    def residual_connection(self, x, residual):
        return residual + x
    def forward(self, x, edge_index, selected_idx, edge_label):
        for convs in self.convs:
            residual = x
            x = self.convs_layer_norm(x)
            x = convs(x, edge_index)
            x = F.relu(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
        
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = self.residual_connection(x, residual)

        batch, dim = selected_idx.size(0), x.size(1) 
        x = x.reshape(batch, -1, dim)
        x = torch.gather(x, 1, selected_idx.unsqueeze(-1).repeat(1,1,dim))
        return x
