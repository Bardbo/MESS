# -*- coding: utf-8 -*-
# @Author: Bardbo
# @Date:   2020-11-16 17:22:37
# @Last Modified by:   Bardbo
# @Last Modified time: 2020-11-16 17:22:47
import osmnx as ox
import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
import multiprocessing as mp
import time

# 0小时时段的OD对行程时间均值数据 0小时时段的OD对最短路距离计算结果
hour_0_OD_mean_duration = pd.read_csv('myCode/data/hour_mean_OD_duration_data/hour_0_mean_OD_duration.csv')
init_length = pd.read_csv('myCode/data/init_length/hour_0_init_length.csv')

len_1 = len(hour_0_OD_mean_duration)

# 数据去除不可达、去除速度过快和过慢 与 初始的平均速度计算
init_length.replace(np.inf, -1, inplace=True)
hour_0_OD_mean_duration['shortest_length'] = init_length['shortest_length']
hour_0_OD_mean_duration = hour_0_OD_mean_duration[~(
    hour_0_OD_mean_duration['shortest_length'] == -1)]
hour_0_OD_mean_duration['speed'] = hour_0_OD_mean_duration['shortest_length'] / \
              hour_0_OD_mean_duration['duration']
hour_0_OD_mean_duration = hour_0_OD_mean_duration[~((hour_0_OD_mean_duration['speed'] < 0.5) | \
    (hour_0_OD_mean_duration['speed'] > 30))].reset_index(drop=True)

len_2 = len(hour_0_OD_mean_duration)
print(f'原数据{len_1}条，现数据{len_2}条，去除数据{len_1-len_2}条')

hour_0_OD_mean_duration.to_csv('myCode/multi/hour0_trip_get_speed.csv', index=False)