import torch
import torch.nn as nn
from torch import optim

from utils.load_config import get_Parameter
from utils.util import convert_to_gpu
from utils.calculate_metric import calculate_metrics, normalized_transform
from utils.data_loader import load_my_data
from train.train_my_model import train_my_model
# from models.transformer_aaa import Transformer
from models.STGAT import SpatialTemporalGAT
from models.Embedding import Embedding_Prediction
from models.EmbeddingSimilarity import EmbeddingGAT
from models.RawGAT import RawGAT
from utils.util import create_graph
import os
import numpy as np
from tqdm import tqdm
import shutil
# from utils.attn_plot import attn_plot


def create_model(model_name, params):
    if model_name.startswith('MyModel'):
        graph_file_path = get_Parameter('graph_path')
        #mobility_graph = os.path.join(graph_file_path, 'mobility_')
        graph = create_graph()
        #support = calculate_normalized_laplacian(adj)
        taxi_nodes = get_Parameter('taxi_size')
        bike_nodes = get_Parameter('bike_size')
        embedding_size = get_Parameter((model_name, 'd_model'))
        num_nodes = get_Parameter('input_size')
        covariate_size = get_Parameter('covariate_size')
        num_layers = get_Parameter((model_name, 'num_layers'))
        num_filters = get_Parameter((model_name, 'num_filters'))
        d_model = get_Parameter((model_name, 'd_model'))
        hidden_features = get_Parameter((model_name, 'hidden'))
        return SpatialTemporalGAT(taxi_nodes, bike_nodes, num_layers, embedding_size, d_model, hidden=hidden_features, graph=graph, attn=False)
    elif model_name.startswith('Embedding'):
        graph = create_graph()
        # support = calculate_normalized_laplacian(adj)
        taxi_nodes = get_Parameter('taxi_size')
        bike_nodes = get_Parameter('bike_size')
        embedding_size = get_Parameter((model_name, 'd_model'))
        num_nodes = get_Parameter('input_size')
        covariate_size = get_Parameter('covariate_size')
        num_layers = get_Parameter((model_name, 'num_layers'))
        num_filters = get_Parameter((model_name, 'num_filters'))
        d_model = get_Parameter((model_name, 'd_model'))
        hidden_features = get_Parameter((model_name, 'hidden'))
        return Embedding_Prediction(embedding_size, embedding_size)
        #return EmbeddingGAT(taxi_nodes, bike_nodes, num_layers, embedding_size, d_model, hidden=hidden_features, graph=graph, attn=False)
    elif model_name.startswith('RawGAT'):
        graph = create_graph()
        # support = calculate_normalized_laplacian(adj)
        taxi_nodes = get_Parameter('taxi_size')
        bike_nodes = get_Parameter('bike_size')
        embedding_size = get_Parameter((model_name, 'd_model'))
        num_nodes = get_Parameter('input_size')
        covariate_size = get_Parameter('covariate_size')
        num_layers = get_Parameter((model_name, 'num_layers'))
        num_filters = get_Parameter((model_name, 'num_filters'))
        d_model = get_Parameter((model_name, 'd_model'))
        hidden_features = get_Parameter((model_name, 'hidden'))
        return RawGAT(taxi_nodes, bike_nodes, num_layers, embedding_size, d_model, hidden=hidden_features, graph=graph, attn=False)

def test_model(model, data_loader, mode, teaching_force=False, **kwargs):
    predictions = list()
    targets = list()
    tqdm_loader = tqdm(enumerate(data_loader))
    model = convert_to_gpu(model)
    #model = nn.DataParallel(convert_to_gpu(model), [0, 1])
    if kwargs['return_attn']:
        attn_record = list()
    with torch.no_grad():
        model.eval()
        for step, (features, truth, covariate) in tqdm(tqdm_loader):
            features = convert_to_gpu(features)
            truth = convert_to_gpu(truth)
            covariate = convert_to_gpu(covariate)
            if kwargs['return_attn']:
                outputs, attn = model(features, covariate)
                #attn_record.append(attn_T.cpu().numpy())
            else:
                outputs = model(features, covariate)
            outputs, truth = normalized_transform(outputs, truth, **kwargs)
            targets.append(truth.cpu().numpy())
            predictions.append(outputs.cpu().detach().numpy())
    pre2 = np.concatenate(predictions)
    tar2 = np.concatenate(targets)
    print(pre2.shape)
    print(calculate_metrics(pre2, tar2, mode, **kwargs))
    if kwargs['return_attn']:
        #attn_plot(attn_record)
        # np.save('data/result/attn.npy', attn)
        return attn
    else:
        return pre2, tar2


def params_setting(scaler):
    params = dict()
    params['scaler'] = scaler
    # params['attn'] = False
    params['embedding'] = True
    params['require_embedding'] = False
    params['return_attn'] = False
    params['pre_train'] = False
    params['load_pretrain'] = False
    params['pca'] = False
    return params


def load_pre_train(model_name, model, pretrain_model_path):
    #load_path = get_Parameter('data_path').split('/')[1] + '-conv.pkl'
    load_path = 'best_model.pkl'
    pretrain_model = torch.load(os.path.join(pretrain_model_path, load_path), map_location={'cuda:0': 'cuda:3'})['model_state_dict']
    # print('pretrain model dict {}'.format(pretrain_model))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_model.items() if k in model_dict}
    print(pretrain_dict.keys())
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    return model


def train():
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3"
    batch_size = get_Parameter('batch_size')
    model_name = get_Parameter('model_name')
    data_path = get_Parameter('data_path')

    # Data = Data_utility(data_path=data_path, train=train_percent, valid=valid_percent, target=target, window=windows)
    data_loader, scaler = load_my_data(batch_size)
    params = params_setting(scaler)
    model = create_model(model_name, params)
    param_num = 0
    for name, param in model.named_parameters():
        print(name, ':', param.size())
        param_num = param_num + np.product(np.array(list(param.size())))
    print('param number:' + str(param_num))

    model_folder = f"save_models/{model_name}"
    tensorboard_folder = f'runs/{model_name}'

    # 训练
    if get_Parameter('mode') == 'train':
        num_epoches = get_Parameter('epochs')
        os.makedirs(model_folder, exist_ok=True)
        shutil.rmtree(tensorboard_folder, ignore_errors=True)
        os.makedirs(tensorboard_folder, exist_ok=True)
        loss = torch.nn.MSELoss()

        if get_Parameter('optim') == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=get_Parameter((model_name, 'lr')),
                                   weight_decay=get_Parameter('weight_decay'))
        elif get_Parameter('optim') == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=get_Parameter((model_name, 'lr')), momentum=0.9)
        elif get_Parameter('optim') == 'RMSProp':
            optimizer = optim.RMSprop(model.parameters(), lr=get_Parameter((model_name, 'lr')))
        else:
            raise NotImplementedError

        if params['load_pretrain']:
            model = load_pre_train(model_name, model, get_Parameter((model_name, 'pretrain_model')))
        model = train_my_model(model, data_loader=data_loader, loss_func=loss, optimizer=optimizer,
                                num_epochs=num_epoches, model_folder=model_folder,
                                tensorboard_folder=tensorboard_folder, **params)
    #model = nn.DataParallel(convert_to_gpu(model), [0, 1])
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location={'cuda:0': 'cuda:2'})['model_state_dict'])
    #model.load_state_dict(torch.load(os.path.join(model_folder, 'loss-normalized(3)-last.pkl'), map_location={'cuda:0': 'cuda:2'})['model_state_dict'])
    attn = test_model(model, data_loader['test'], mode='test', **params)
    return attn
