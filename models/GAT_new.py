import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import numpy as np
from dgl.nn.pytorch import edge_softmax, GATConv
from utils.util import convert_to_gpu

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dist, mul_type, dropout=0.1):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2*out_dim, 1, bias=False)
        self.attn_l = nn.Linear(out_dim, 1, bias=False)
        self.attn_r = nn.Linear(out_dim, 1, bias=False)
        self.dist = dist
        self.mul_type = mul_type
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        #z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=-1)
        a = self.attn_l(edges.src['z']) + self.attn_r(edges.dst['z'])
        # print(z2.shape)
        # a = self.attn_fc(z2)
        # if self.mul_type:
        #     a = a*self.dist
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e':edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha*nodes.mailbox['z'], dim=1)
        return {'h':h}

    def forward(self, g, h):
        self.g = g.to(h.device)
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim,  nheads, dist, mul_type, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(nheads):
            self.heads.append(GATLayer(in_dim, out_dim, dist, mul_type))
        self.merge = merge

    def forward(self, g, h):
        head_out = [attn_head(g, h) for attn_head in self.heads]
        if self.merge=='cat':
            return torch.cat(head_out, dim=-1)
        else:
            return torch.mean(torch.stack(head_out))

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, nheads, dist, mul_type=False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, out_dim, nheads, dist, mul_type)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        return h

# g, features, labels, mask = load_cora_data()
# net = GAT(g, features.size()[1], hidden_dim=16, out_dim=7, nheads=2)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# for epoch in range(50):
#     logits = net(features)
#     logp = F.log_softmax(logits, 1)
#     loss = F.nll_loss(logp[mask], labels[mask])
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print('Epoch{:05d} | Loss{:.4f}'.format(epoch, loss.item()))
#
# embedding_weights = net(features).detach().numpy()
# print(embedding_weights[0])
#
#
# GATConv
# g = DGLGraph(([0,1,2,3,4,5,1], [2,4,1,5,0,3,2]))
# feat = torch.randn(6, 16, 20)
# model = GAT(g, 20, 16,7,2)
# y = model(feat)
# print(y)

GATConv