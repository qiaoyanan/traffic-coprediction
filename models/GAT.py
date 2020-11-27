import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import get_Parameter

class SimpleGAT(nn.Module):
    def __init__(self, in_features, hidden, out_dim, nheads, conv, alpha=0.2, dropout=0.2):
        super(SimpleGAT, self).__init__()
        self.out = nn.Linear(in_features, out_dim)

    def forward(self, input, mask=None, attn=None):
        return self.out(input), attn


class GAT(nn.Module):
    def __init__(self, in_features, hidden, out_dim, nheads, conv, alpha=0.2, dropout=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attns = nn.ModuleList([GraphAttentionLayer(in_features, hidden, dropout=dropout, alpha=alpha, conv=conv) for _ in range(nheads)])
        # for i, attention in enumerate(self.attns):
        #     self.add_module('attention_{}'.format(i), attention)
        self.out_attn = GraphAttentionLayer(nheads * hidden, out_dim, dropout, alpha)

    def forward(self, input, mask=None, attn=None):
        '''

        :param input: [batch, N, in_features]
        :return: [batch, N, out_dim]
        '''
        x = F.dropout(input, self.dropout, training=self.training)
        x = torch.cat([att(x, mask)[0] for att in self.attns], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x, attn = self.out_attn(x, mask)
        return x, attn


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, conv=False):
        super(GraphAttentionLayer, self).__init__()
        self.outfeatures = out_features
        self.dropout = dropout
        self.conv = conv
        if conv:
            self.W = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=3)
        else:
            self.W1 = nn.Linear(in_features, out_features, bias=False)
            self.W2 = nn.Linear(in_features, out_features, bias=False)
            # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
            # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, input, mask):
        '''

        :param input: [batch, N, F]
        :return:
        '''
        batch, N, input_dim = input.size()
        if self.conv:
            input = torch.cat((input, torch.zeros((batch, 2, input_dim), device=input.device)), dim=1)
            h = self.W(input.transpose(1, 2)).transpose(1, 2)
        else:
            # print(input.device, self.W.device)
            #print(input.shape, input.device, self.W.weight.device)
            h = torch.cat([self.W1(input[:, :get_Parameter('taxi_size')]), self.W2(input[:, get_Parameter('taxi_size'):])], dim=1)
            #h = self.W(input)
            # h = torch.matmul(input, self.W)
        # a_input = torch.cat([h.repeat(1, 1, N).view(batch, N * N, -1), h.repeat(1, N, 1)], dim=2).view(batch, N, -1,
        #                                                                                                2 * self.outfeatures)
        A1 = torch.matmul(h, self.a[:self.outfeatures])
        A2 = torch.matmul(h, self.a[self.outfeatures:])
        sim = torch.add(A1.expand(-1, -1, N), torch.transpose(A2.expand(-1, -1, N), 1, 2))
        del A1, A2
        # print(sim)
        e = self.leakyrelu(sim)
        if mask is not None:
            n = e.shape[0]
            #mask = (mask>0).unsqueeze(0).repeat(n, 1, 1)
            #print(mask)
            e = mask.unsqueeze(0).repeat(n, 1, 1)*e
            #e = e.masked_fill(mask, -np.inf)
        #attention = F.softmax(e, dim=2)

        # print(attention)

        # zero_vec = -9e15 * torch.ones_like(e)
        # print(attention)
        # print(torch.gt(attention, 0.1).sum(dim=2))
        # if mask is not None:
        #     mask = mask.repeat(156, 1, 1)
        #     e = torch.where(mask>0, e, zero_vec)

        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # print(attention)

        attention = F.softmax(e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h)

        return F.elu(h_prime, inplace=True), attention


class ScaledDotAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(ScaledDotAttention, self).__init__()
        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight)
        self.outfeatures = out_features
        self.dropout = dropout

    def forward(self, input):
        '''

        :param input: [batch, N, F]
        :return:
        '''
        x = self.W(input)
        attn = torch.bmm(x, x.transpose(1, 2)) / self.outfeatures
        attention = F.softmax(attn, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        out = torch.bmm(attention, x)
        return out, attention


class EmbeddingAttention(nn.Module):
    def __init__(self, input_dim, hidden, alpha=0.2, dropout=0.1):
        super(EmbeddingAttention, self).__init__()
        self.outfeatures = hidden
        self.W = nn.Parameter(torch.zeros(size=(input_dim, hidden)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha, inplace=True)
        self.dropout = dropout

    def forward(self, embedding, input):
        '''

        :param input: [batch, T, input_dim]
        :param embedding: [batch, T, input_dim]
        :return:
        '''
        batch, T, _ = embedding.size()
        h = torch.matmul(embedding, self.W)
        a_input = torch.cat([h.repeat(1, 1, T).view(batch, T * T, -1), h.repeat(1, T, 1)], dim=2).view(batch, T, -1, 2 * self.outfeatures)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        attn = F.softmax(e, dim=2)
        attn = attn.repeat(get_Parameter('input_size'), 1, 1)
        attention = F.dropout(attn, self.dropout, training=self.training)
        out = torch.bmm(attention, input)
        return out

class GroupedGAT(nn.Module):
    def __init__(self, in_features, hidden, out_dim, nheads, period_num, alpha=0.2, dropout=0.2):
        super(GroupedGAT, self).__init__()
        self.dropout = dropout
        self.attns = nn.ModuleList([GroupedGATLayer(in_features, hidden, period_num=period_num, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        # for i, attention in enumerate(self.attns):
        #     self.add_module('attention_{}'.format(i), attention)
        self.out_attn = GroupedGATLayer(nheads * hidden, out_dim, period_num, dropout, alpha)

    def forward(self, input, mask=None, attn=None):
        '''

        :param input: [batch, N, in_features]
        :return: [batch, N, out_dim]
        '''
        #x = F.dropout(input, self.dropout, training=self.training)
        x = torch.cat([att(input)[0] for att in self.attns], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x, attn = self.out_attn(x)
        return x, attn

class GroupedGATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, period_num, dropout, alpha):
        super(GroupedGATLayer, self).__init__()
        self.period = period_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.intraGAT = GraphAttentionLayer(input_dim, output_dim, dropout, alpha)
        self.intergroup = nn.Linear(self.period*output_dim, input_dim)
        self.interGAT = GraphAttentionLayer(input_dim, output_dim, dropout, alpha)

    def forward(self, input):
        '''

        :param input: shape[batch*N, T, input_dim]
        :return: [batch*N, T, output_dim]
        '''
        batch, T, input_dim = input.size()
        x_period = input.view(batch, -1, self.period, input_dim).view(-1, self.period, input_dim)
        x_period, attn = self.interGAT(x_period, mask=None)
        x_grouped = self.intergroup(x_period.view(-1, self.period*self.output_dim)).view(batch, -1, input_dim)
        x_grouped, grouped_attn = self.interGAT(x_grouped, mask=None)
        x = x_period.view(batch, -1, self.period, self.output_dim) + x_grouped.unsqueeze(2).repeat(1, 1, self.period, 1)
        x_out = x.view(batch, -1, self.output_dim)
        return x_out, (attn, grouped_attn)



