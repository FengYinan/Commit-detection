import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
from torch.autograd import Variable
import math
from functools import partial

from utils import *

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##################################################################################
# Model
##################################################################################
class encoder(nn.Module):
    def __init__(self, in_channels, conv_ch, hbead=6, dropout=0.6, alpha=0.2, af=True):
        super(encoder, self).__init__()
        #self.conv1 = GATConv(in_channels, conv_ch, dropout=dropout, alpha=alpha)
        #self.norm2 = GroupNorm(2, conv_ch)
        #self.conv2 = GATConv(conv_ch, int(conv_ch/2), dropout=dropout, alpha=alpha)
        self.elinear1 = nn.Linear(in_channels,64)
        self.elinear2 = nn.Linear(64, 64)
        self.conv1 = GATConv(64, conv_ch, dropout=dropout, alpha=alpha, concat=False)
        self.attentions = [GATConv(conv_ch, int(conv_ch / 2), dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(hbead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.norm1 = GroupNorm(1, int(conv_ch / 2)*hbead)
        self.conv2 = GATConv(int(conv_ch / 2)*hbead, int(conv_ch / 2), dropout=dropout, alpha=alpha, concat=False)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, edge_index):
        #edge_index = edge_index.permute(1,0)
        x = self.elinear1(x)
        x = self.elinear2(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.relu(x)
        x = self.norm1(x)
        #x = self.conv1(x, edge_index)
        #x = self.norm2(x)
        z = self.conv2(x, edge_index)

        #z = F.leaky_relu(self.conv3(z, edge_index)) + z
        #z = F.relu(self.conv4(z, edge_index)) + z
        #z = F.relu(self.conv5(z, edge_index)) + z

        return z, edge_index

class decoder(nn.Module):
    def __init__(self, in_channels, conv_ch, dropout=0.6, alpha=0.2, af=True):
        super(decoder, self).__init__()
        self.norm3 = GroupNorm(1, int(conv_ch / 2))
        self.edge_decoder_conv = GATConv(int(conv_ch/2), conv_ch, dropout=dropout, alpha=alpha)
        #self.norm4 = GroupNorm(1, conv_ch)

        self.x_decoder_conv2 = GATConv(int(conv_ch/2), conv_ch, dropout=dropout, alpha=alpha)
        #self.norm5 = GroupNorm(1, conv_ch)
        self.x_decoder_conv3 = GATConv(conv_ch, 64, dropout=dropout, alpha=alpha, concat=False)

        self.dlinear1 = nn.Linear(64, 64)
        self.dlinear2 = nn.Linear(64, in_channels)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, z, edge_index):
        z = F.elu(z)
        z = self.norm3(z)
        recon_edge = self.edge_decoder_conv(z, edge_index)
        #recon_edge = torch.tanh(recon_edge)
        recon_edge = torch.sigmoid(torch.matmul(recon_edge, recon_edge.permute(1,0)))

        x = self.x_decoder_conv2(z, edge_index)
        x = torch.tanh(x)
        x = self.x_decoder_conv3(x, edge_index)
        x = self.dlinear1(x)
        x = self.dlinear2(x)

        return recon_edge, x

class autoencoder(nn.Module):
    def __init__(self, in_channels, conv_ch, dropout=0.6, alpha=0.2, af=True):
        super(autoencoder, self).__init__()

        self.encoder = encoder(in_channels, conv_ch, dropout=0.6, alpha=0.2, af=True)
        self.decoder = decoder(in_channels, conv_ch, dropout=0.6, alpha=0.2, af=True)

        self.reset_parameters()

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = edge_index.T
        z, edge_index = self.encoder(x, edge_index)
        recon_edge, x = self.decoder(z, edge_index)

        return recon_edge, x, z

class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]])
        new_X[idx] = X
        return A, new_X


class GraphPool(nn.Module):

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        # scores = torch.abs(scores)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores / 100)
        num_nodes = A.shape[0]
        if self.k > num_nodes:
            new_X = torch.zeros([self.k, X.shape[1]])
            new_A = torch.zeros([self.k, self.k])
            values, idx = torch.topk(scores, num_nodes)
            values = torch.unsqueeze(values, -1)
            new_X[:num_nodes] = torch.mul(X[idx, :], values)
            A = A[idx, :]
            A = A[:, idx]
            new_A[:num_nodes] = A
        else:
            values, idx = torch.topk(scores, int(self.k))
            new_X = X[idx, :]
            values = torch.unsqueeze(values, -1)
            new_X = torch.mul(new_X, values)
            A = A[idx, :]
            new_A = A[:, idx]
        return new_A, new_X, idx


class GATConv(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)).float())
        #nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)).float())
        #nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)
        self.W.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        h = torch.mm(inputs, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.in_channels = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C= x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.permute(1,0).reshape(G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.reshape(C, N).permute(1,0)
        return x * self.weight + self.bias

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
