'''
主要是图处理
'''
import torch
from utils.load_config import get_Parameter
import os
import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import dgl

def convert_to_gpu(data):
    if torch.cuda.is_available():
        data = data.cuda(get_Parameter('cuda'))
    return data

def save_model(path, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)

def load_graph_data():
    taxi_mobility_graph = np.load('data/train-data/graph/taxi-graph-mobility.npy')
    taxi_distance_graph = np.load('data/train-data/graph/taxi-graph-distance.npy')
    bike_mobility_graph = np.load('data/train-data/graph/bike-graph-mobility.npy')
    bike_distance_graph = np.load('data/train-data/graph/bike-graph-distance.npy')
    return torch.FloatTensor(taxi_mobility_graph + np.identity(taxi_mobility_graph.shape[0])), \
           torch.FloatTensor(taxi_distance_graph + np.identity(taxi_mobility_graph.shape[0])), \
           torch.FloatTensor(bike_mobility_graph + np.identity(bike_mobility_graph.shape[0])), \
           torch.FloatTensor(bike_distance_graph + np.identity(bike_mobility_graph.shape[0]))

def create_graph():
    distance_graph = np.load('data/train-data/graph/distance-graph.npy')
    mobility_graph = np.load('data/train-data/graph/mobility-graph.npy')
    similarity_graph = np.load('data/train-data/graph/similarity-graph.npy')
    distance_g = add_graph_edges(distance_graph)
    mobility_g = add_graph_edges(mobility_graph)
    similarity_g = add_graph_edges(similarity_graph)

    return (distance_g, mobility_g, similarity_g)

def add_graph_edges(graph):
    adj_mx = sp.coo_matrix(graph)
    g_graph = dgl.from_scipy(adj_mx, eweight_name='w')
    return g_graph


