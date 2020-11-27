import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph
import dgl.function as fn
from dgl.ops import edge_softmax
from utils.util import get_Parameter

class GATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, attn_drop=0.1, negative_slope=0.2):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._out_feats = out_feats
        self.fc1 = nn.Linear(in_feats, out_feats*num_heads, bias=False)
        self.fc2 = nn.Linear(in_feats, out_feats*num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, feat):
        '''

        :param graph: DGLGraph
        :param feat: <N, b, F>
        :return:
        '''
        with graph.local_scope():
            N, b, _ = feat.size()
            graph = graph.local_var()
            graph = graph.to(feat.device)
            feat = torch.cat([self.fc1(feat[:get_Parameter('taxi_size')]), self.fc2(feat[get_Parameter('taxi_size'):])], dim=0)
            feat_src = feat_dst = feat.view(N, b, self._num_heads, self._out_feats)
            #feat_src = feat_dst = self.fc(feat).view(N, b, self._num_heads, self._out_feats)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})

            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            #graph.apply_edges(fn.u_mul_e('e', 'w', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            #print(graph.edata['a'].size())
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            rst = rst.reshape(N, -1, self._num_heads*self._out_feats)
            return rst, graph.edata['a']

# model = GATConv(in_feats=10, out_feats=20, num_heads=3)
# g = DGLGraph(([0,1,2,3,4,5,1], [2,4,1,5,0,3,2]))
# feat = torch.randn(6, 16, 10)
# # model = GAT(g, 20, 16,7,2)
# y = model(g, feat)
# print(y.shape)
# print(y)
