import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.load_config import get_Parameter

def seperate_for_mode(result):
    seperate_size = get_Parameter('taxi_size')
    return result[:, :, :seperate_size], result[:, :, seperate_size:]

def seperate_for_pickdrop(result):
    return result[:, :, :, 0], result[:, :, :, 1]

def normalized_transform(predict, target, **kwargs):
    scaler = kwargs['scaler']
    if get_Parameter('normalized'):
        predict, target = scaler.inverse_transform(predict), scaler.inverse_transform(target)
    return predict, target

def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
            #mask = labels>10
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        mse = np.nan_to_num(mse)
        return np.mean(mse)

def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            labels = labels.astype('int32')
            mask = np.not_equal(labels, null_val)
            #mask = labels>10
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels + 1e-5))
        mape = np.nan_to_num(mask * mape)
        mape = np.nan_to_num(mape)
        return np.mean(mape)

def get_metrics(predict, target):
    length_p, length_t = len(predict), len(target)
    if length_p != length_t:
        assert 'wrong ! cannot calculate metric'
    RMSEs, MAEs, MAPEs = list(), list(), list()
    for i in range(length_p):
        predict[i] = predict[i].reshape(predict[i].shape[0], -1)
        target[i] = target[i].reshape(target[i].shape[0], -1)
        mse = mean_squared_error(predict[i], target[i])
        mae = mean_absolute_error(predict[i], target[i])
        #pcc = pearsonr(predict[i].flatten(), target[i].flatten())
        mape = masked_mape_np(predict[i], target[i], null_val=0)
        RMSEs.append(np.sqrt(mse))
        MAEs.append(mae)
        MAPEs.append(mape)
    return RMSEs, MAEs, MAPEs


def calculate_metrics(predict, target, mode, **kwargs):
    # if get_Parameter('normalized') and get_Parameter('loss_normalized'):
    #     predict, target = normalized_transform(predict, target, **kwargs)
    if mode == 'train' or (not get_Parameter('seperate_mode') and not get_Parameter('seperate_pickdrop')):
        mse = mean_squared_error(predict, target)
        #mse = masked_mse_np(predict, target, null_val=0)
        mae = mean_absolute_error(predict, target)
        mape = masked_mape_np(predict, target, null_val=0)
        #pcc, _ = pearsonr(predict.flatten(), target.flatten())
        return {
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'MAPE': mape
        }
    if get_Parameter('seperate_mode'):
        if get_Parameter('taxi_size') + get_Parameter('bike_size') != get_Parameter('input_size'):
            assert 'wrong parameter setting!'
        taxi_predict, bike_predict = seperate_for_mode(predict)
        taxi_target, bike_target = seperate_for_mode(target)
        predict = [taxi_predict, bike_predict]
        target = [taxi_target, bike_target]
    if get_Parameter('seperate_pickdrop'):
        if isinstance(predict, list):
            combine_predict, combine_target = list(), list()
            for indice in predict:
                pickup, dropoff = seperate_for_pickdrop(indice)
                combine_predict.extend([pickup, dropoff])
            for indice in target:
                pickup, dropoff = seperate_for_pickdrop(indice)
                combine_target.extend([pickup, dropoff])
            predict, target = combine_predict, combine_target
        else:
            # pickup_predict, dropoff_predict = seperate_for_pickdrop(predict)
            # pickup_target, dropoff_target = seperate_for_pickdrop(target)
            # predict = [pickup_predict, dropoff_predict]
            # target = [pickup_target, dropoff_target]
            predict = predict.reshape(predict.shape[0], -1, 2)
            target = target.reshape(target.shape[0], -1, 2)
            predict = [predict[:,:,0], predict[:, :, 1]]
            target = [target[:,:,0], target[:, :, 1]]

    RMSEs, MAEs, MAPEs = get_metrics(predict, target)
    return {
        'RMSE': RMSEs,
        'MAE': MAEs,
        'MAPE': MAPEs
    }

