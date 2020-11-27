'''
特征提取：
① 交通模式：出租车0 自行车1
② 交通站点对应网格：行数 列数
③ 交通站点id: 出租车：0-184 自行车：200-421

① 星期：1-7
② 小时：0-23
③ 是否工作日：0，1
'''

import pandas as pd
import numpy as np

nyc_holidays = ['2016-01-01', '2016-01-18', '2016-02-15', '2016-05-30', '2016-07-04', '2016-09-05',
                '2016-10-10', '2016-10-31', '2016-11-11', '2016-11-24', '2016-12-25']

#bike station pos
def getBikePos(stations):
    origin = pd.read_csv('/mnt/windows-E/qyn/nycdata/201601-citibike-tripdata.csv', header=0)
    iterator = origin.groupby(by=['start station id']).first()
    station_dict = dict()
    for station in stations:
        lng = iterator.loc[station]['start station longitude']
        lat = iterator.loc[station]['start station latitude']
        station_dict[station] = [lng, lat]
    return station_dict

def getPosGrid(pos):
    import math
    lon, lat = pos
    max_lat = 40.917577
    min_lat = 40.477399
    max_lng = -73.700272
    min_lng = -74.259090
    row = 100
    col = 100
    unit_x = math.floor((lat - min_lat) / (max_lat - min_lat) * row)
    unit_y = math.floor((lon - min_lng) / (max_lng - min_lng) * col)
    return [unit_x, unit_y]

def get_origin_data(pickup_path, dropoff_path, type='bike'):
    taxi_pickup = pd.read_csv(pickup_path, header=0, index_col=0, parse_dates=True)
    taxi_dropoff = pd.read_csv(dropoff_path, header=0, index_col=0, parse_dates=True)
    T, N = taxi_pickup.shape
    extra_features = pd.DataFrame()
    extra_features['date'] = taxi_pickup.index
    extra_features['day'] = extra_features.apply(lambda x: x['date'].strftime('%w'), axis=1)
    extra_features['hour'] = extra_features.apply(lambda x: x['date'].strftime('%H'), axis=1)
    extra_features['work'] = extra_features.apply(lambda x: int(x['day']==6) or int(x['day']==7) or x['date'].strftime('%Y-%m-%d') in nyc_holidays, axis=1)
    extra_features.drop(columns=['date'], inplace=True)
    temporal_features = np.stack([extra_features.values for _ in range(N)], axis=1)
    extra_features = pd.DataFrame()
    stations = [int(s) for s in taxi_pickup.columns]
    if type=='bike':
        station_dict = getBikePos(stations)
        print(station_dict)
        extra_features['row'] = [getPosGrid(station_dict[s])[0] for s in stations]
        extra_features['columns'] = [getPosGrid(station_dict[s])[1] for s in stations]
        traffic_mode = np.ones((T, N, 1))
        station = np.arange(200, N+200).repeat(T).reshape(-1, T).transpose().reshape(T, N, 1)
    else:
        extra_features['row'] = [int(s)//10000 for s in taxi_pickup.columns]
        extra_features['columns'] = [int(s)%1000//10 for s in taxi_pickup.columns]
        traffic_mode = np.zeros((T, N, 1))
        station = np.arange(N).repeat(T).reshape(-1, T).transpose().reshape(T, N, 1)
    grid_features = np.stack([extra_features[['row', 'columns']].values for _ in range(T)], axis=0)
    features = np.concatenate([traffic_mode, grid_features, station, temporal_features], axis=2)
    result = np.concatenate([np.stack([taxi_pickup.values, taxi_dropoff.values], axis=2), features], axis=2)
    np.save('data/feature-data/'+type+'_data.npy', result)


pickup_path = '/mnt/windows-E/qyn/traffic-prediction/data/sequence-data/data2016/bike-pickup-filter.csv'

dropoff_path = '/mnt/windows-E/qyn/traffic-prediction/data/sequence-data/data2016/bike-dropoff-filter.csv'
get_origin_data(pickup_path, dropoff_path, 'bike')




