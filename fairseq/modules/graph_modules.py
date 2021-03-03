import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
        if m.bias != None:
          m.bias.data.fill_(0.01)
    return
def build_linear(input_dim, output_dim, q_noise, qn_block_size, bias=True):
    linear = quant_noise(
            nn.Linear(input_dim, output_dim, bias=bias), p=q_noise, block_size=qn_block_size
        )
    init_weights(linear)
    return linear
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
    def forward(self, x):
        x = self.phrase_activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x

class GatingResidual(nn.Module):
    def __init__(self, embed_dim, quant_noise, qn_block_size, args):
        super().__init__()
        self.fc1 = build_linear(embed_dim, 1, quant_noise, qn_block_size, False)
        self.fc_sublayer = build_linear(embed_dim, 1, quant_noise, qn_block_size, False)
    def forward(self, x, sublayer):
        alpha = torch.sigmoid(self.fc1(x) + self.fc_sublayer(sublayer))
        x = alpha * x + (1 - alpha) * sublayer
        return x

class SlotAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, quant_noise, qn_block_size, args, epsilon=1e-8):
        super().__init__()
        
        self.eps = epsilon
        self.num_iter = 3
        self.num_heads = num_heads
        self.dim_head = in_dim
        self.mlp_hidden_dim = 2048 // self.num_heads
        self.slots_mu = nn.Parameter(torch.Tensor(1, 1, self.dim_head))
        self.slots_log_sigma = nn.Parameter(torch.Tensor(1, 1, self.dim_head))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)
        self.atten = nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))
        nn.init.xavier_uniform_(self.atten)

        self.norm_slots = LayerNorm(self.dim_head)
        self.norm_mlp = LayerNorm(self.dim_head)
        
        
        self.project_q = build_linear(self.dim_head, self.dim_head, quant_noise, qn_block_size, False)
        self.project_k = build_linear(2*in_dim, self.dim_head, quant_noise, qn_block_size, False)
        self.gru = nn.GRUCell(in_dim*num_heads, in_dim*num_heads)
        
        self.mlp = FeedForward(self.dim_head, self.mlp_hidden_dim, 
                                     self.dim_head, quant_noise,
                                     qn_block_size, args)

    def forward(self, x, edge_index_i, size_i):
        norm_dist = Variable(torch.empty(x.size(0), self.num_heads, self.dim_head,dtype=x.dtype)\
                        .normal_(mean=0,std=1)).cuda()
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * norm_dist

        k = self.project_k(x) #Shape: [N, num_heads, embed_dim]
        k = k * self.dim_head ** -0.5

        # Shape: [N, num_labels, slot_dim]
        for _ in range(self.num_iter):
            slots_prev = slots.view(-1, self.dim_head * self.num_heads) 
            slots = self.norm_slots(slots)
            
            q = self.project_q(slots)
            q = q * self.dim_head ** -0.5
            alpha = F.leaky_relu((q * self.atten).sum(-1), 0.2)
            alpha = pyg_utils.softmax(alpha, edge_index_i, num_nodes=size_i)
            
            updates = k * alpha.unsqueeze(-1)
            updates = updates.view(-1, self.dim_head * self.num_heads)

            slots = self.gru(updates, slots_prev).view(-1, self.num_heads, self.dim_head)
            
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots.view(-1, self.num_heads * self.dim_head)


class EdgeConv(MessagePassing):
    def __init__(self, F_in, F_out, quant_noise, qn_block_size, args):
        super(EdgeConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.mlp = FeedForward(F_in, F_out, F_out, quant_noise, qn_block_size, args)
        self.label_linear = build_linear(2 * F_in, F_in, quant_noise, qn_block_size, bias=False)

    def forward(self, x, edge_index, x_label):
        self.x_label = x_label
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)
        edge_features = self.label_linear(edge_features) + self.x_label
        return self.mlp(edge_features)

class GraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, 
                quant_noise, qn_block_size, args,
                reducer='mean', normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        self.lin = build_linear(in_channels, out_channels, quant_noise, qn_block_size)
        self.agg_lin = build_linear(in_channels, out_channels, quant_noise, qn_block_size)
        if normalize_embedding:
            self.normalize_emb = True
    def forward(self, x, edge_index, x_label):
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
                bias=False, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_dim = out_channels // num_heads
        assert out_channels % num_heads == 0
        self.heads = num_heads
        self.concat = concat 
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.lin = build_linear(self.in_channels, self.out_channels, quant_noise, qn_block_size)
        self.att = nn.Parameter(torch.Tensor(self.heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.att)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        if bias:
            nn.init.zeros_(self.bias)
        self.label_linear = build_linear(self.out_channels, self.out_channels, quant_noise, qn_block_size, bias=False)
        self.slot_attn = SlotAttention(self.head_dim, self.in_channels, self.heads, quant_noise, qn_block_size, args)
        self.gating_label = GatingResidual(self.in_channels, quant_noise, qn_block_size, args)
        self.label_ffn_combine = FeedForward(2 * self.out_channels, 2048, self.out_channels, quant_noise, qn_block_size, args)

    def forward(self, x, edge_index, x_label, size=None):
        self.x_label = x_label
        x = self.lin(x)
        return self.propagate(edge_index, size=size, x=x)
    def message(self, edge_index_i, x_i, x_j, size_i):
        
        x_i = x_i.view(-1, self.heads, self.head_dim)
        x_j = x_j.view(-1, self.heads, self.head_dim)
        x_cat = torch.cat([x_i, x_j], dim=-1) # x_cat.shape = (N, heads, 2 * head_dim)

        label_attn = self.slot_attn(x_cat, edge_index_i, size_i)
        x_label = self.gating_label(label_attn, self.x_label)

        alpha = F.leaky_relu((x_cat * self.att).sum(-1), 0.2)
        alpha = pyg_utils.softmax(alpha, edge_index_i, num_nodes=size_i)

        alpha = self.dropout_module(alpha)
        edge_features = (x_j * alpha.unsqueeze(-1)).view(-1, self.out_channels)
        residual = edge_features
        edge_out = self.label_linear(edge_features) + x_label
        edge_out = self.label_ffn_combine(torch.cat([edge_features, edge_out], dim=-1))
        edge_out = self.dropout_module(edge_out)
        edge_features = residual + edge_out
        return edge_features
    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class UCCAEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, args):
        super(UCCAEncoder, self).__init__()
        self.label_embedding = nn.Embedding(13, in_dim)
        nn.init.normal_(self.label_embedding.weight, mean=0, std=in_dim ** -0.5)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.num_layers = 3 # hard-code
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        graph_type = getattr(args, 'graph_type', None)
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

        
        self.convs = nn.ModuleList()
        self.convs.append(Model(*settings_first))
        for i in range(self.num_layers-1):
            self.convs.append(Model(*settings_else))

        self.ffn = FeedForward(hidden_dim, 2048, out_dim, self.quant_noise, self.quant_noise_block_size, args)
        self.gru = nn.GRUCell(self.in_dim, self.in_dim)
        self.gru_ffn = FeedForward(self.in_dim, 2048, self.in_dim, self.quant_noise, self.quant_noise_block_size, args)
        self.gru_layer_norm = LayerNorm(self.in_dim)
        self.convs_layer_norm = LayerNorm(self.in_dim)
        self.ffn_layer_norm = LayerNorm(self.hidden_dim)

    def residual_connection(self, x, residual):
        return residual + x
    def forward(self, x, edge_index, selected_idx, edge_label):
        x_label = self.label_embedding(edge_label)
        for convs in self.convs:
            prev_x = x
            x = self.convs_layer_norm(x)
            x = convs(x, edge_index, x_label)
            x = F.relu(x)
            x = self.dropout_module(x)
            x = self.gru(x, prev_x)
            x = x + self.gru_ffn(self.gru_layer_norm(x))
        
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        batch, dim = selected_idx.size(0), x.size(1) 
        x = x.reshape(batch, -1, dim)
        x = torch.gather(x, 1, selected_idx.unsqueeze(-1).repeat(1,1,dim))
        return x
