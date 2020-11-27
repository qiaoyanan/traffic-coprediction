import pandas as pd
import numpy as np

data_path = 'data/od_final_data/taxi-od-data.h5'
filter_data_path = 'data/final-data/taxi-pickup-half-filter.csv'
graph_path = 'data/all-graph-24.npy'


# use_period = ['']
def graph_construct(od_data, node_list, graph_path):
    grouped1 = od_data.groupby(['pickup_unit_no', 'dropoff_unit_no', (od_data['pickup_datetime'].dt.hour)]).size()
    grouped2 = od_data.groupby(['pickup_unit_no', 'dropoff_unit_no', (od_data['dropoff_datetime'].dt.hour)]).size()
    grouped3 = od_data[od_data['pickup_datetime'].dt.hour == od_data['dropoff_datetime'].dt.hour].groupby(
        ['pickup_unit_no', 'dropoff_unit_no', (od_data['dropoff_datetime'].dt.hour)]).size()
    graphs = list()
    for gidx in range(24):
        graph = np.ndarray(shape=(len(node_list), len(node_list)), dtype=np.float32)
        for pidx, pnode in enumerate(node_list):
            for didx, dnode in enumerate(node_list):
                p_count = grouped1.get((pnode, dnode, gidx), 0)
                d_count = grouped2.get((pnode, dnode, gidx), 0)
                union = grouped3.get((pnode, dnode, gidx), 0)
                graph[pidx, didx] = p_count + d_count - union
        graphs.append(graph)
    np.save(graph_path, np.array(graphs))
    return graphs


threshold = 0.005
data = pd.read_hdf(data_path, key='taxi')
filter_data = pd.read_csv(filter_data_path, header=0, index_col=0)
node_list = filter_data.columns.astype('int64').to_list()
data = data[data['pickup_datetime']<'2017-07-01']
#graph = graph_construct(data, node_list, graph_path)
groupedall = data.groupby(['pickup_cluster_no', 'dropoff_cluster_no']).size()
graph = np.ndarray(shape=(len(node_list), len(node_list)), dtype=np.float32)
for pidx, pnode in enumerate(node_list):
    for didx, dnode in enumerate(node_list):
        p_count = groupedall.get((pnode, dnode), 0)
        graph[pidx, didx] = p_count
graph_sum = graph/graph.sum(axis=1)
graph_filter = (graph_sum >= threshold).astype('float32') * graph
np.save('data/final-data/taxi-graph.npz', graph_filter)


#####同质图的构建
#mobility graph construct
import numpy as np
taxi_graph = np.load('data/graph-data/taxi-graph-raw.npy')
bike_graph = np.load('data/graph-data/bike-graph-raw.npy')
percentile = np.percentile(taxi_graph, 90, axis=1)
taxi_graph = taxi_graph*(taxi_graph.transpose() >= percentile).transpose()
graph = taxi_graph.transpose()/taxi_graph.sum(axis=1)
np.save('data/graph-data/taxi_graph_mobility', graph.transpose())


#distance graph construct
import pandas as pd
# taxi = pd.read_csv('data/sequence-data/data2016/taxi-pickup-filter.csv', header=0)
# stations = [int(s) for s in taxi.columns[1:]]
# stations = [(s//1000, s%1000) for s in stations]

def cal_distance(stations):
    import math
    n = len(stations)
    graph = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row1, col1 = stations[i]
            row2, col2 = stations[j]
            dis = math.sqrt((row1-row2)**2 + (col1-col2)**2)
            graph[i][j] = dis
    return graph
#
# graph = np.array(cal_distance(stations))
# dis_graph = 1/(graph+0.00001)
# percentile = np.percentile(dis_graph, 90, axis=1)
# filter_graph = dis_graph*(dis_graph.transpose() >= percentile).transpose()
# filter_graph = filter_graph-np.identity(len(filter_graph))*100000
# filter_graph = filter_graph.transpose()/filter_graph.sum(axis=1)
# np.save('data/graph-data/taxi-graph-distance', filter_graph.transpose())

def getDistance(pos1, pos2):
    import math
    R = 6378137
    long1 = pos1[0]
    lat1 = pos1[1]
    long2 = pos2[0]
    lat2 = pos2[1]
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0
    a = lat1 - lat2
    b = (long1 - long2) * math.pi / 180.0
    sa2 = math.sin(a / 2.0)
    sb2 = math.sin(b / 2.0)
    d = 2 * R * math.asin(math.sqrt(sa2 * sa2 + math.cos(lat1) * math.cos(lat2) * sb2 * sb2))
    return d


def GetBikePos(stations):
    origin = pd.read_csv('/mnt/windows-E/qyn/nycdata/201601-citibike-tripdata.csv', header=0)
    iterator = origin.groupby(by=['start station id']).first()
    station_dict = dict()
    for station in stations:
        lng = iterator.loc[station]['start station longitude']
        lat = iterator.loc[station]['start station latitude']
        station_dict[station] = [lng, lat]
    return station_dict

def cal_bike_distance(stations, station_dict):
    import math
    n = len(stations)
    graph = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            pos1 = station_dict[stations[i]]
            pos2 = station_dict[stations[j]]
            dis = getDistance(pos1, pos2)
            graph[i][j] = dis
    return graph

# bike = pd.read_csv('data/sequence-data/data2016/bike-pickup-filter.csv', header=0)
# stations = [int(s) for s in bike.columns[1:]]
# station_dict = GetBikePos(stations)
# graph = cal_bike_distance(stations, station_dict)
# dis_graph = 1/(graph+0.00001)
# percentile = np.percentile(dis_graph, 90, axis=1)
# filter_graph = dis_graph*(dis_graph.transpose() >= percentile).transpose()
# filter_graph = filter_graph-np.identity(len(filter_graph))*100000
# filter_graph = filter_graph.transpose()/filter_graph.sum(axis=1)
# np.save('data/graph-data/taxi-graph-distance', filter_graph.transpose())

def get_distance_graph():
    import math
    max_lat = 40.917577
    min_lat = 40.477399
    max_lng = -73.700272
    min_lng = -74.259090
    row = 1000
    col = 1000
    taxi_distance_graph = np.load('data/train-data/graph/taxi-graph-distance.npy')
    bike_distance_graph = np.load('data/train-data/graph/bike-graph-distance.npy')
    bike = pd.read_csv('/mnt/windows-E/qyn/traffic-prediction/data/sequence-data/data2016/bike-pickup-filter.csv', header=0)
    stations = [int(s) for s in bike.columns[1:]]
    station_dict = GetBikePos(stations)
    bike_station = []
    for station in stations:
        lon, lat = station_dict[station]
        unit_x = math.floor((lat - min_lat) / (max_lat - min_lat) * row)
        unit_y = math.floor((lon - min_lng) / (max_lng - min_lng) * col)
        bike_station.append((unit_x, unit_y))
    taxi = pd.read_csv('/mnt/windows-E/qyn/traffic-prediction/data/sequence-data/data2016/taxi-pickup-filter.csv', header=0)
    stations = [int(s) for s in taxi.columns[1:]]
    taxi_station = [(s // 1000, s % 1000) for s in stations]
    heterogeneous_graph = np.zeros(shape=(185, 221))
    for i in range(len(taxi_station)):
        for j in range(len(bike_station)):
            unit_x_i, unit_y_i = taxi_station[i]
            unit_x_j, unit_y_j = bike_station[j]
            if  math.sqrt((unit_x_i-unit_x_j)**2 + (unit_y_i-unit_y_j)**2)<=20:
                heterogeneous_graph[i][j] = 1
            else:
                pass
    a = np.concatenate([taxi_distance_graph, heterogeneous_graph], axis=1)
    b = np.concatenate([heterogeneous_graph.T, bike_distance_graph], axis=1)
    graph = np.concatenate([a, b])
    graph = graph + np.identity(len(taxi_station)+len(bike_station))

    return graph

def get_mobility_graph():
    taxi_size = 185
    bike_size = 221
    taxi_mobility_graph = np.load('data/train-data/graph/taxi-graph-mobility.npy')
    bike_mobility_graph = np.load('data/train-data/graph/bike-graph-mobility.npy')
    heterogeneous_graph = np.zeros(shape=(taxi_size, bike_size))
    a = np.concatenate([taxi_mobility_graph, heterogeneous_graph], axis=1)
    b = np.concatenate([heterogeneous_graph.T, bike_mobility_graph], axis=1)
    graph = np.concatenate([a, b])
    #graph = graph + np.identity(taxi_size+bike_size)
    np.save('data/train-data/graph/mobility-graph', graph)
    return graph

def get_similarity_graph(taxi_path, bike_path, threshold, length=168):
    import math
    taxi_data = pd.read_csv(taxi_path, header=0, index_col=0).values
    bike_data = pd.read_csv(bike_path, header=0, index_col=0).values
    demand_data = np.concatenate([taxi_data, bike_data], axis=1)
    demand_data = demand_data/demand_data.max(axis=0)
    node_num = demand_data.shape[1]
    adj_mx = np.zeros((node_num, node_num))
    demand_zero = np.zeros((length))
    for i in range(node_num):
        node_i = demand_data[-length:, i]
        adj_mx[i][i] = 1
        if np.array_equal(node_i, demand_zero):
            continue
        else:
            for j in range(i+1, node_num):
                node_j = demand_data[-length:, j]
                distance = math.exp(-(np.abs(node_j-node_i)).sum()/length)
                if distance>threshold:
                    adj_mx[i][j] = distance
                    adj_mx[j][i] = distance
    np.save('data/train-data/graph/similarity-graph', adj_mx)
    return adj_mx

taxi_path = '/mnt/windows-E/qyn/traffic-prediction/data/sequence-data/data2016/taxi-pickup-filter.csv'
bike_path = '/mnt/windows-E/qyn/traffic-prediction/data/sequence-data/data2016/bike-pickup-filter.csv'
graph = get_similarity_graph(taxi_path, bike_path, 0)

















