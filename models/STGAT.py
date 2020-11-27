import torch
import torch.nn as nn
import torch.nn.functional as fn
from utils.util import get_Parameter
from models.GAT_new import GAT
from models.GAT import GroupedGAT
from models.GATConv import GATConv
#from dgl.nn.pytorch import GATConv

class FeatureEmbedding(nn.Module):
    def __init__(self, embedding_size, hidden_dim):
        super(FeatureEmbedding, self).__init__()
        self.field_size = 7
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        # feature : traffic_mode, row, column, station, day, hour, work
        '''
        init fm part
        '''
        self.feature_sizes = [2, 1, 1, 200+get_Parameter('bike_size'), 7, 24, 2]
        self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, embedding_size) for feature_size in self.feature_sizes])

        '''
        init deep part
        '''
        all_dims = [self.field_size*embedding_size] + hidden_dim
        deep_layers = []
        for i in range(1, len(hidden_dim)+1):
            deep_layers.append(nn.Sequential(nn.Linear(all_dims[i-1], all_dims[i]), nn.BatchNorm1d(all_dims[i]), nn.ReLU()))
        self.deep_module = nn.ModuleList(deep_layers)
        self.concat = nn.Linear(self.field_size+self.embedding_size*2, embedding_size)

    def forward(self, covariate, type='all'):
        '''

        :param covariate: <batch, T, N1+N2, Features>
        :return: <batch, T, N, embedding_size>
        '''
        batch, T, N, F = covariate.size()
        features = covariate.reshape(-1, F).unsqueeze(2)
        features_index = features
        features_index[:, 1:3, :] = torch.zeros_like(features_index[:, 1:3, :])
        features_index = features_index.long()
        features_values = torch.ones_like(features)
        features_values[:, 1:3, :] = features[:, 1:3, :]
        '''
        fm part
        '''
        fm_first_order_emb_arr = [torch.sum(emb(features_index[:, i, :]), 1)*features_values[:, i] for i, emb in enumerate(self.fm_first_order_embeddings)]
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        fm_second_order_emb_arr = [(torch.sum(emb(features_index[:, i, :]), 1) * features_values[:, i]) for i, emb in
                                   enumerate(self.fm_second_order_embeddings)]
        if type=='spatial':
            spatial_embedding = sum(fm_second_order_emb_arr[:4])
            return spatial_embedding.reshape(batch, T, N, self.embedding_size)
        elif type=='temporal':
            temporal_embedding = sum(fm_second_order_emb_arr[4:])
            return temporal_embedding.reshape(batch, T, N, self.embedding_size)
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
                                         fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [
            item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(
            fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5
        '''
        deep part
        '''
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for deep_layer in self.deep_module:
            deep_out = deep_layer(deep_out)
        '''
        concat
        '''
        total_sum = self.concat(torch.cat([fm_first_order, fm_second_order, deep_out], axis=1))
        feature_embedding = total_sum.reshape(batch, T, N, self.embedding_size)
        return feature_embedding

class SingleEmbedding(nn.Module):
    def __init__(self, field_size, embedding_size):
        super(SingleEmbedding, self).__init__()
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.feature_sizes = [2, 1, 1, 200 + get_Parameter('bike_size'), 7, 24, 2]
        self.field_embeddings = nn.ModuleList([nn.Embedding(feature_size, embedding_size) for feature_size in self.feature_sizes])


class SpatialBlock(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(SpatialBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.homogeneousGAT = GAT(output_dim, hidden, output_dim, nheads=4, conv=False)
        # self.heterogeneousGAT = GAT(output_dim, hidden, output_dim, nheads=4, conv=False)
        # self.spatial_distance = GAT(in_dim=input_dim, out_dim=hidden, nheads=1, dist=None)
        # self.spatial_mobility = GAT(in_dim=input_dim, out_dim=hidden, nheads=1, dist=None)
        # self.spatial_similarity = GAT(in_dim=input_dim, out_dim=hidden, nheads=3, dist=None)
        self.spatial_distance = GATConv(in_feats=input_dim, out_feats=hidden, num_heads=3)
        self.spatial_mobility = GATConv(in_feats=input_dim, out_feats=hidden, num_heads=3)
        self.spatial_similarity = GATConv(in_feats=input_dim, out_feats=hidden, num_heads=3)
        self.weight = nn.Linear(3*output_dim, output_dim)
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, input, covariate, graph):
        '''

        :param input:<batch, T, N1+N2, F>
        :param covariate: <batch, T, N1+N2, F>
        :param graph:[]
        :return: <batch, T, N1+N2, F>
        '''
        batch, T, N, input_dim = input.size()
        if self.input_dim!=self.output_dim:
            residual = self.residual(input)
        else:
            residual = input
        distance_g, mobility_g, similarity_g = graph
        input = input + covariate
        x = input.transpose(0, 2).reshape(N, -1, input_dim)
        x_distance, attn_d = self.spatial_distance(distance_g, x)
        x_mobility, attn_m = self.spatial_mobility(mobility_g.to(x.device), x)
        x_similarity, attn_s = self.spatial_similarity(similarity_g, x)
        #x_attn = torch.mean(torch.stack([x_distance, x_mobility, x_similarity]), dim=0)
        #x_attn = self.weight(torch.cat([x_distance, x_mobility, x_similarity], dim=-1))
        x_attn = torch.cat([x_distance, x_mobility, x_similarity], dim=-1)
        x_attn = x_attn.transpose(0, 1).reshape(batch, T, N, -1)
        #out = residual + x_attn

        covariate = covariate[:, 0]#<batch, N, F>
        covariate = covariate.reshape(-1, N, input_dim)
        attn = torch.bmm(covariate, covariate.transpose(1, 2))
        attn = fn.softmax(attn, dim=2)
        attn = attn.repeat(T, 1, 1)
        embedding_x = torch.bmm(attn, input.reshape(-1, N, input_dim)).reshape(batch, T, N, -1)

        out = residual + x_attn*torch.sigmoid(embedding_x)
        out = residual + x_attn

        return out, attn

class TemporalBlock(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(TemporalBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.GAT = GroupedGAT(input_dim, hidden, output_dim, nheads=4, period_num=24)
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, input, covariate):
        '''

        :param input: <batch, T, N, input_dim>
        :param covariate: <batch, T, N, embedding_size>
        :return:<batch, T, N, output_dim>
        '''
        batch, T, N, input_dim = input.size()
        if self.input_dim!=self.output_dim:
            residual = self.residual(input)
        else:
            residual = input
        x = input.transpose(1, 2).contiguous().view(batch * N, T, -1)
        x_attn, g_attn = self.GAT(x)
        covariate = covariate[:, :, 0]
        #embedding_x = self.embedding_attn(ex_encoding_input, x)
        attn_trans = torch.bmm(covariate, covariate.transpose(1, 2))
        attn = fn.softmax(attn_trans, dim=1)
        attn = attn.repeat(N, 1, 1)
        embedding_x = torch.bmm(attn, x)
        x_attn = x_attn*torch.sigmoid(embedding_x)
        x_attn = x_attn.reshape(batch, N, T, -1).transpose(1, 2)
        out = residual + x_attn
        return out, g_attn


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(EncoderLayer, self).__init__()
        self.spatialblock = SpatialBlock(input_dim, hidden, output_dim)
        self.Temporalblock = TemporalBlock(input_dim, hidden, output_dim)
        self.W = nn.Linear(2 * output_dim, output_dim)

    def forward(self, input, spatial_embedding, temporal_embedding, graph):
        '''

        :param input: <batch, T, N1+N2, F>
        :param covariate: <batch, T, N1+N2, F>
        :param graph: 四个图 mobility, graph
        :return: <batch, T, N1+N2, F>
        '''
        spatial, attn_S = self.spatialblock(input, spatial_embedding, graph)
        temporal, attn_T = self.Temporalblock(input, temporal_embedding)
        z = torch.sigmoid(self.W(torch.cat([spatial, temporal], dim=3)))
        # out = temporal*torch.sigmoid(spatial)
        out = z * spatial + (1 - z) * temporal
        out = out + input
        return out, attn_S

class Decoder(nn.Module):
    def __init__(self, input_dim, output_size=2):
        super(Decoder, self).__init__()
        self.out1 = nn.Linear(input_dim, output_size)
        self.out2 = nn.Linear(input_dim, output_size)


    def forward(self, input, features_input, features_target):
        '''

        :param input: <batch, T, N, F>
        :param features_input: <batch, T, N, F>
        :param features_target: <batch, 1, N, F>
        :return: <batch, 1, N, 2>
        '''
        batch, T, N, input_dim = input.size()
        attn_trans = torch.matmul(features_input.transpose(1, 2), features_target.transpose(1, 2).transpose(2, 3))
        attn = torch.softmax(attn_trans, dim=2)
        out =torch.matmul(input.permute(0, 2, 3, 1), attn).transpose(2, 3)
        out = torch.cat([self.out1(out[:, :get_Parameter('taxi_size')]), self.out2(out[:, get_Parameter('taxi_size'):])], dim=1)
        return out.transpose(1, 2)


class SpatialTemporalGAT(nn.Module):
    def __init__(self, taxi_nodes, bike_nodes, num_layers, embedding_size, d_model, hidden, graph, attn=False):
        super(SpatialTemporalGAT, self).__init__()
        self.graph = graph

        self.taxi_nodes = taxi_nodes
        self.bike_nodes = bike_nodes
        self.feature_embedding = FeatureEmbedding(embedding_size=embedding_size, hidden_dim=[d_model])
        self.encoded = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=d_model, kernel_size=(3, 1), padding=(1, 0)),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(3, 1),
                                               padding=(1, 0)))
        encoder_list = []
        for _ in range(num_layers):
            encoder_list.append(EncoderLayer(d_model, hidden, d_model))
        self.encoder = nn.ModuleList(encoder_list)
        self.decoder = Decoder(d_model)
        self.attn = attn

    def forward(self, enc_input, covariate):
        '''

        :param enc_input: <batch, T, N1+N2, F(2)>
        :param covariate: <batch, T+1, N1+N2, Features>
        :return: <batch, N1+N2, 1>
        '''

        batch, T, N, F = enc_input.size()
        features = self.feature_embedding(covariate)
        spatial_emedding = self.feature_embedding(covariate, type='spatial')[:, :-1]
        temporal_embedding = self.feature_embedding(covariate, type='temporal')[:, :-1]
        source_covariate, target_covariate = features[:, :-1], features[:, [-1]]
        # taxi_input, bike_input = enc_input[:, :, :get_Parameter('taxi_size')], enc_input[:, :,
        #                                                                        get_Parameter('taxi_size'):]
        # taxi_mobility_graph, taxi_distance_graph, bike_mobility_graph, bike_distance_graph = self.graph
        x = self.encoded(enc_input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        for encoderlayer in self.encoder:
            #x, attn = encoderlayer(x, self.graph)
            x, attn = encoderlayer(x+source_covariate, spatial_emedding, temporal_embedding, [graph.local_var() for graph in self.graph])
        out = self.decoder(x, source_covariate, target_covariate)
        #out = self.decoder(x)
        if self.attn:
            return out, attn
        else:
            return out
