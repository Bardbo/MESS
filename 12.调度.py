# -*- coding: utf-8 -*-
# @Author: Bardbo
# @Date:   2020-12-06 19:08:52
# @Last Modified by:   Bardbo
# @Last Modified time: 2020-12-11 14:57:58
import osmnx as ox
import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings("ignore")

global i_d, G_ig, theta, pred_time, delay, cpus, params, data

# 空驶时间阈值 阈值越大连接数越多，车辆逐渐减少，空驶率逐渐增大
# 此处需考虑司机在原地等待的时间
theta = 15
# 预知时长 时长逐渐增长，连接数逐渐增多多，车辆数逐渐减少， 愈加难以等待或者预测
pred_time = 60
# 延误时间阈值
delay = 0

data = pd.read_csv('myCode/data/day_data/use_day_1.csv')
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
data['pred_time'] = data['pickup_datetime'] + pd.Timedelta(minutes=pred_time)

# 构建图网络
G = ox.load_graphml('myCode/data/Manhattan.graphml')
G_ig = ig.Graph(directed=True)
G_ig.add_vertices(list(G.nodes()))
G_ig.add_edges(list(G.edges()))
travel_time = pd.read_csv('myCode/multi/travel_time.csv')
G_ig.es['travel_time'] = travel_time['travel_time']

def get_timedelta(x):
    x = round(x)
    return pd.Timedelta(seconds=x)

def get_share(i):
    
    # print(i)
    # 获取可连接的需求 构建车辆共享网络
    # 预知时长约束，当前需求至pred_time 内的出行，
    pred_data = data[(data['pickup_datetime'][i] <= data['pickup_datetime']) & (data['pickup_datetime'] <= data['pred_time'][i])]
    # 当前出行的终点
    i_d = pred_data.iloc[0, :]['D']

    # 空驶行程时间计算函数
    def get_t_ij(j_o, i_d=i_d, G=G_ig):
        return G_ig.shortest_paths(i_d, j_o, weights='travel_time')[0][0]

    # 当前出行终点前往pred_time 内的出行起点的行程时间 5.56s
    t_ij = pred_data.iloc[1:, :]['D'].apply(get_t_ij).apply(get_timedelta)

    # 当前出行终点的预计到达时刻
    t_id = pred_data.iloc[0, :]['dropoff_datetime']

    # 实际的延误时间 
    real_delay = (t_id + t_ij - pred_data.iloc[0:, :]['pickup_datetime'])
    # 约束
    # 当前出行终点+空驶行程时间需 小于等于 下一个行程的起点时间+允许的延误
    condition1 = real_delay <= pd.Timedelta(minutes=delay)
    # 确定行程时间 小于空驶阈值， 且无论其晚到达或者早到达，当前出行终点时刻与下一个行程的起点时刻应该相差在空驶阈值内
    delta_two_trips = pred_data.iloc[1:, :]['pickup_datetime'] - t_id
    condition2 = (t_ij <= pd.Timedelta(minutes=theta)) & (delta_two_trips <= pd.Timedelta(minutes=theta))
    # 结合两个条件
    condition = condition1 & condition2
    # 当延误小于0时，没有乘客的时间等于delta_two_trips， 当延误大于等于0时，这个时间等于行驶时间
    condition3 = (real_delay < pd.Timedelta(minutes=0))
    pred_data['no_user_time'] = delta_two_trips[condition3].append(t_ij[~condition3]).sort_index()

    # 共享网络边保存
    share_trip = pred_data.iloc[1:, :][condition].index
    save_data = pd.DataFrame({'D': share_trip})
    save_data['O'] = i
    save_data['no_user_time'] = pred_data['no_user_time'][share_trip].tolist()
    # 实际可调度行程的起点时间差值
    save_data['real_pred_time'] = (pred_data['pickup_datetime'][share_trip] - data['pickup_datetime'][i]).tolist()
    # # 第一个需求的出发时间
    # save_data['O_time'] = data['pickup_datetime'][i]
    # # 调度的第二个需求的结束时间
    # save_data['D_time'] = pred_data['dropoff_datetime'][share_trip].tolist()
    save_data.to_csv(f'myCode/share_data/day_1_{theta}_{pred_time}_{delay}_306062', index=False, mode='a', header=None)


if __name__ == "__main__":
    try:
        data_len = len(data) -1
        cpus = mp.cpu_count() - 1
        params = [(i, ) for i in range(data_len)]
        # params = [(i, ) for i in range(247603, data_len)]
        # params = [(i, ) for i in range(299347, data_len)]
        # params = [(i, ) for i in range(306062, data_len)]
        pool = mp.Pool(cpus)
        sma = pool.starmap_async(get_share, params)

        start_time = time.time()
        sma.get()
        pool.close()
        pool.join()
        print(time.time() - start_time)
    except:
        print('error...')
        print(time.time() - start_time)
            

