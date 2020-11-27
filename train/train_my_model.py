import itertools

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils.util import convert_to_gpu, save_model
from utils.calculate_metric import calculate_metrics, normalized_transform
from utils.load_config import get_Parameter
from tqdm import tqdm
import copy
import numpy as np
import os
from utils.util import create_graph


def train_my_model(model: nn.Module, data_loader, loss_func: callable, optimizer, num_epochs, model_folder,
                tensorboard_folder: str, **kwargs):
    phases = ['train', 'val', 'test']
    writer = SummaryWriter(tensorboard_folder)
    model = convert_to_gpu(model)
    #model = nn.DataParallel(convert_to_gpu(model), [0, 1])
    loss_func = convert_to_gpu(loss_func)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=8, threshold=1e-4, min_lr=1e-6)
    save_dict = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}
    loss_global = 100000
    for epoch in range(num_epochs):
        running_loss = {phase: 0.0 for phase in phases}
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            steps, predictions, targets = 0, list(), list()
            tqdm_loaders = tqdm(enumerate(data_loader[phase]))
            for step, (features, truth, covariate) in tqdm_loaders:
                features = convert_to_gpu(features)
                truth = convert_to_gpu(truth)
                covariate = convert_to_gpu(covariate)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(features, covariate)
                    if not get_Parameter('loss_normalized'):
                        outputs, truth = normalized_transform(outputs, truth, **kwargs)
                    taxi_pickup_loss = loss_func(truth[:, :, :get_Parameter('taxi_size'), 0], outputs[:, :, :get_Parameter('taxi_size'), 0])
                    taxi_dropoff_loss = loss_func(truth[:, :, :get_Parameter('taxi_size'), 1], outputs[:, :, :get_Parameter('taxi_size'), 1])
                    #taxi_loss = loss_func(truth[:, :, :get_Parameter('taxi_size')], outputs[:, :, :get_Parameter('taxi_size')])
                    taxi_loss = taxi_pickup_loss + taxi_dropoff_loss*1.5
                    bike_loss = loss_func(truth[:, :, get_Parameter('taxi_size'):], outputs[:, :, get_Parameter('taxi_size'):])
                    # if epoch<=100:
                    #     loss = (2*taxi_loss + bike_loss)*100
                    # else:
                    #     loss = taxi_loss
                    #loss = taxi_loss + 30*bike_loss
                    loss = (1.5*taxi_loss + bike_loss)*100
                    #loss = loss_func(truth, outputs)
                    #loss = bike_loss
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    if get_Parameter('loss_normalized'):
                        outputs, truth = normalized_transform(outputs, truth, **kwargs)
                targets.append(truth.cpu().numpy())
                with torch.no_grad():
                    predictions.append(outputs.cpu().numpy())
                running_loss[phase] += loss.item()
                steps += truth.size(0)

                tqdm_loaders.set_description(f'{phase} epoch:{epoch}, {phase} loss: {running_loss[phase]/steps}')

            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)

            scores = calculate_metrics(predictions.reshape(predictions.shape[0], -1),
                                       targets.reshape(targets.shape[0], -1), mode='train', **kwargs)
            print(scores)
            writer.add_scalars(f'score/{phase}', scores, global_step=epoch)
            if phase == 'val' and scores['RMSE'] < loss_global:
                loss_global = scores['RMSE']
                save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()), epoch=epoch,
                                 optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))

        scheduler.step(running_loss['train'])
        writer.add_scalars('Loss', {
            f'{phase} loss': running_loss[phase] for phase in phases
        }, global_step=epoch)

    save_model(f'{model_folder}/best_model.pkl', **save_dict)
    model.load_state_dict(save_dict['model_state_dict'])
    return model
