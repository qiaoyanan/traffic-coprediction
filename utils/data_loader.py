import numpy as np
import os

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from utils.load_config import get_Parameter
from utils.util import convert_to_gpu


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std, need_transform):
        if not need_transform:
            self.mean = mean
            self.std = std
        else:
            self.mean = np.expand_dims(mean, 1).repeat(need_transform, axis=1)
            self.std = np.expand_dims(std, 1).repeat(need_transform, axis=1)

    def transform(self, data):
        data_scale = (data - self.mean) / self.std
        #data = np.concatenate((data_scale, data[:, :, get_Parameter('input_size'):]), axis=2)
        return data_scale

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            mean_a = convert_to_gpu(torch.from_numpy(self.mean))
            std_a = convert_to_gpu(torch.from_numpy(self.std))
            data = convert_to_gpu(data)
        else:
            mean_a = self.mean
            std_a = self.std
        return (data * std_a) + mean_a


class MaxMinScaler:
    def __init__(self, Max, Min, need_transform):
        if not need_transform:
            self.Max = Max
            self.Min = Min
        else:
            self.Max = np.expand_dims(Max, 1).repeat(need_transform, axis=1)
            self.Min = np.expand_dims(Min, 1).repeat(need_transform, axis=1)
        print("max shape is:{}".format(self.Max.shape))

    def transform(self, data):
        data_scale = (data - self.Min) / (self.Max - self.Min)
        #data = np.concatenate((data_scale, data[:, :, get_Parameter('input_size'):]), axis=2)
        return data_scale

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            max_a = convert_to_gpu(torch.from_numpy(self.Max))
            min_a = convert_to_gpu(torch.from_numpy(self.Min))
            data = convert_to_gpu(data)
        else:
            max_a = self.Max
            min_a = self.Min
        # print(data.device)
        # print(max_a.device)
        return (data * (max_a - min_a)) + min_a

class FeatureDataSet(Dataset):
    def __init__(self, inputs, target, covariate_data):
        '''

        :param taxi_data: <n, T, input_size, 2>
        :param bike_data:
        '''
        self.inputs = inputs
        self.targets = target
        self.covariate = covariate_data

    def __getitem__(self, item):
        return self.inputs[item], self.targets[item], self.covariate[item]

    def __len__(self):
        return len(self.inputs)

def merge_data(data_dir, data, phases):
    data_all = None
    for phase in phases:
        cat_data = np.load(os.path.join(data_dir, phase + '.npy'))
        s = cat_data.shape[0]
        data_covariate = cat_data[:, :, :, 2:]
        data_train = cat_data[:, :, :, :2]
        x_data = data_train[:, :get_Parameter('window')]
        y_data = data_train[:, get_Parameter('window'):]

        key = 'x_' + phase
        if key not in data:
            data['x_' + phase] = x_data
            data['y_' + phase] = y_data
            data['covariate_' + phase] = data_covariate
        else:
            data['x_' + phase] = np.concatenate([data['x_' + phase], x_data], axis=2)
            data['y_' + phase] = np.concatenate([data['y_' + phase], y_data], axis=2)
            data['covariate_' + phase] = np.concatenate([data['covariate_' + phase], data_covariate], axis=2)
        if data_all is not None:
            data_all = np.vstack((data_all, data_train))
        else:
            data_all = data_train
    return data_all

def load_my_data(batch_size):
    loader = {}
    data = {}
    phases = ['train', 'val', 'test']
    data_all = None
    for dir in ['taxi', 'bike']:
        data_dir = 'data/train-data/' + dir
        data_merge = merge_data(data_dir, data, phases)
        if data_all is None:
            data_all = data_merge
        else:
            data_all = np.concatenate([data_all, data_merge], axis=2)
        #data_all <n, T, N1+N2, F>
        if get_Parameter('normalized') == 1:
            scaler = StandardScaler(mean=data_all.mean(), std=data_all.std())
        elif get_Parameter('normalized') == 2:
            scaler = MaxMinScaler(Max=data_all.max(), Min=data_all.min(), need_transform=False)
        elif get_Parameter('normalized') == 3:
            scaler = MaxMinScaler(Max=np.amax(data_all, (0, 1, 3)), Min=np.amin(data_all, (0, 1, 3)), need_transform=2)
        elif get_Parameter('normalized') == 4:
            scaler = StandardScaler(mean=np.mean(data_all, axis=(0, 1, 3)), std=np.std(data_all, axis=(0, 1, 3)), need_transform=2)
        else:
            pass
    loader['scaler'] = scaler
    print(data['covariate_test'].shape)
    for phase in phases:
        data['x_' + phase] = scaler.transform(data['x_' + phase])
        data['y_' + phase] = scaler.transform(data['y_' + phase])
        loader[phase] = DataLoader(FeatureDataSet(inputs=data['x_' + phase], target=data['y_' + phase], covariate_data=data['covariate_' + phase]), batch_size,
                                   shuffle=False, drop_last=False)
    return loader, scaler

