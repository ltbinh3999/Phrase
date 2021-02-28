import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules import LayerNorm

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    return
class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, quant_noise, qn_block_size, args):
        super().__init__()
        self.fc1 = self.build_ffw(in_dim, hidden_dim, quant_noise, qn_block_size)
        self.fc2 = self.build_ffw(hidden_dim, out_dim, quant_noise, qn_block_size)
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
    def build_ffw(self, input_dim, output_dim, q_noise, qn_block_size):
      return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )
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

        self.convs_layer_norm = LayerNorm(self.in_dim)
        self.ffn_layer_norm = LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList()
        Model = EdgeConv
        self.convs.append(Model(in_dim, hidden_dim, self.quant_noise, self.quant_noise_block_size, args))
        for i in range(self.num_layers-1):
            self.convs.append(Model(hidden_dim, hidden_dim, self.quant_noise, self.quant_noise_block_size, args))

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
