import torch
import numpy as np
import os

class Data_utility():
    def __init__(self, data_path, train, valid, window, target, horizon=0):
        self.raw_data = np.load(data_path, allow_pickle=True)
        # self.raw_data = np.loadtxt(data_path, delimiter=',')
        self.length, self.dimension, self.features = self.raw_data.shape
        self.window = window
        self.horizon = horizon
        self.target = target
        # normalized
        self.data = self.raw_data
        # self.data = self.normalized()
        # if self.horizon==0:
        #     self.seq_data = self.indice_from_data(window=window, target=target)
        # else:
        #     self.seq_data = self.indice_from_data(window=window, target=horizon)
        # print(self.seq_data)
        self._split(int(train * self.length), int((train + valid) * self.length), self.length)

    # def indice_from_data(self, window, target):
    #     seq_data = [(i, i + window + target) for i in range(self.length - window - target + 1)]
    #     np.random.shuffle(seq_data)
    #     return seq_data

    # def normalized(self):
    #     self.mean = np.mean(self.raw_data)
    #     self.std = np.std(self.raw_data)
    #     return (self.raw_data - self.mean)/self.std

    # 数据集的划分
    def _split(self, train, valid, all):
        train_set = range(self.window + self.target - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, all)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    # 数据集划分的迭代流程
    def _batchify(self, index_set):
        n = len(index_set)
        X = torch.zeros((n, self.window, self.dimension, self.features))
        Y = torch.zeros((n, self.target, self.dimension, self.features))

        for i in range(n):
            end_index = index_set[i] - self.target + 1
            start_index = end_index - self.window
            X[i, :, :, :] = torch.from_numpy(self.data[start_index:end_index, :, :])
            Y[i, :, :, :] = torch.from_numpy(self.data[end_index: index_set[i] + 1, :, :])

        return [X, Y]

data_path = '../data/feature-data/bike_data.npy'
to_path = '../data/train-data/bike'
d = Data_utility(data_path=data_path, train=0.6, valid=0.2, window=48, target=1)
train = np.concatenate(d.train, axis=1)
valid = np.concatenate(d.valid, axis=1)
test = np.concatenate(d.test, axis=1)
np.save(os.path.join(to_path, 'train.npy'), train)
np.save(os.path.join(to_path, 'val.npy'), valid)
np.save(os.path.join(to_path, 'test.npy'), test)