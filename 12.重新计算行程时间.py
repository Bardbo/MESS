import osmnx as ox
import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
import multiprocessing as mp
import time

global G_ig, cpus, params, travel_time

G = ox.load_graphml('myCode/data/Manhattan.graphml')
G_ig = ig.Graph(directed=True)
G_ig.add_vertices(list(G.nodes()))
G_ig.add_edges(list(G.edges()))
travel_time = pd.read_csv(r'D:\大论文\myCode\multi\travel_time.csv')
G_ig.es['travel_time'] = travel_time['travel_time']

day_data = pd.read_csv('D:\大论文\myCode\data\day_data\day_1',
                       names=[
                           'pickup_datetime', 'dropoff_datetime',
                           'pickup_longitude', 'pickup_latitude',
                           'dropoff_longitude', 'dropoff_latitude', 'O', 'D',
                           'duration', 'weekday', 'day', 'hour'
                       ])
day_data.sort_values(['pickup_datetime', 'dropoff_datetime'], inplace=True)
day_data.reset_index(inplace=True, drop=True)

cpus = mp.cpu_count() - 1
params = ((G_ig, orig, dest) for orig, dest in zip(day_data.O, day_data.D))


def shortest(G_ig, orig, dest):
    return G_ig.shortest_paths(orig, dest, weights='travel_time')[0][0]


def get_travel(func_name, cpus=cpus, params=params):
    pool = mp.Pool(cpus)
    sma = pool.starmap_async(func_name, params)
    travel = sma.get()
    pool.close()
    pool.join()
    return travel


if __name__ == '__main__':
    start = time.time()
    new_duration = pd.DataFrame(get_travel(shortest), columns=['travel_time'])
    day_data['new_duration'] = new_duration
    day_data.to_csv('D:\大论文\myCode\data\day_data\day_1.csv', index=False)
    print(time.time() - start)