import torch
import torch.nn as nn
from utils.util import get_Parameter
import torch.functional as fn

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
        all_dims = [self.field_size*embedding_size] + hidden_dim*2
        deep_layers = []
        for i in range(1, len(hidden_dim)+1):
            deep_layers.append(nn.Sequential(nn.Linear(all_dims[i-1], all_dims[i]), nn.BatchNorm1d(all_dims[i]), nn.ReLU()))
        self.deep_module = nn.ModuleList(deep_layers)
        self.out = nn.Linear(self.field_size+self.embedding_size*2, 2)

    def forward(self, covariate):
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
        total_sum = self.out(torch.cat([fm_first_order, fm_second_order, deep_out], axis=1))
        feature_embedding = total_sum.reshape(batch, T, N, -1)
        return feature_embedding

class Embedding_Prediction(nn.Module):
    def __init__(self, embedding_size, hidden_dim):
        super(Embedding_Prediction, self).__init__()
        self.feature_embedding = FeatureEmbedding(embedding_size, [hidden_dim])

    def forward(self, enc_input, covariate):
        '''
        :param enc_input: <batch, T, N1+N2, F(2)>
        :param covariate: <batch, T+1, N1+N2, Features>
        :return: <batch, N1+N2, 1>
        '''
        target_covariate = covariate[:, [-1]]
        embedding = self.feature_embedding(target_covariate)
        return embedding