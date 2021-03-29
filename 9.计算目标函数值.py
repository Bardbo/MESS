# -*- coding: utf-8 -*-
# @Author: Bardbo
# @Date:   2020-11-16 17:37:20
# @Last Modified by:   Bardbo
# @Last Modified time: 2020-11-16 22:08:14
import osmnx as ox
import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse
# import time

global G_ig, cpus, params, travel_time

G = ox.load_graphml('myCode/data/Manhattan.graphml')
G_ig = ig.Graph(directed=True)
G_ig.add_vertices(list(G.nodes()))
G_ig.add_edges(list(G.edges()))
speed = pd.read_csv('D:\大论文\myCode\multi\speed.csv')
length = pd.read_csv('D:\大论文\myCode\multi\edges_data.csv')['length']
travel_time = (length / speed['highway']).tolist()
G_ig.es['travel_time'] = travel_time

cpus = mp.cpu_count() - 1
trip_data = pd.read_csv('D:\大论文\myCode\multi\hour0_trip_get_speed.csv')
params = ((G_ig, orig, dest) for orig, dest in zip(trip_data.O, trip_data.D))

def shortest(G_ig, orig, dest):
    return G_ig.shortest_paths(
        orig, dest, weights='travel_time')[0][0]


def get_travel(func_name, cpus=cpus, params=params):
    pool = mp.Pool(cpus)
    sma = pool.starmap_async(func_name, params)
    travel = sma.get()
    pool.close()
    pool.join()
    return travel


if __name__ == '__main__':
    # start = time.time()
    travel_data = pd.DataFrame(get_travel(shortest),
                               columns=['travel_time'])
    travel_data['bias'] = travel_data['travel_time'] - trip_data['duration']
    loss = travel_data['bias'].abs().mean()
    if loss == float('inf'):
        loss = 3600
    # print(loss)
    pd.DataFrame([loss]).to_csv('D:\大论文\myCode\multi\loss.csv', index=False)
    # print(time.time() - start)