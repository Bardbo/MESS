# -*- coding: utf-8 -*-
# @Author: Bardbo
# @Date:   2020-11-11 15:23:00
# @Last Modified by:   Bardbo
# @Last Modified time: 2020-11-12 20:12:04
import osmnx as ox
import networkx as nx
import igraph as ig
import pandas as pd
import multiprocessing as mp
import time

G = ox.load_graphml('myCode\data\Manhattan.graphml')
assert G.is_directed() is True, 'G is not directed'

# convert networkx graph to igraph
G_ig = ig.Graph(directed=True)
G_ig.add_vertices(list(G.nodes()))
G_ig.add_edges(list(G.edges()))
# G_ig.vs['osmid'] = list(nx.get_node_attributes(new_G, 'osmid').values())
G_ig.es['length'] = list(nx.get_edge_attributes(G, 'length').values())

hour_0_OD_mean_duration = pd.read_csv(
    'myCode\data\hour_mean_OD_duration_data\hour_0_mean_OD_duration.csv')

# def shortest_path(G, orig, dest):
#     try:
#         return nx.dijkstra_path_length(G, orig, dest, weight='length')
#     except:
#         return None


def shortest_path(G_ig, orig, dest):
    try:
        return G_ig.shortest_paths(orig, dest, weights='length')[0][0]
    except:
        return None


params = ((G_ig, orig, dest) for orig, dest in zip(hour_0_OD_mean_duration.O,
                                                   hour_0_OD_mean_duration.D))

if __name__ == "__main__":
    cpus = mp.cpu_count()
    pool = mp.Pool(cpus)
    sma = pool.starmap_async(shortest_path, params)

    start_time = time.time()
    routes = sma.get()
    pool.close()
    pool.join()
    print(len(hour_0_OD_mean_duration), time.time() - start_time)
    # how many total results did we get
    print(len(routes))

    # and how many were solvable paths
    routes_valid = [r for r in routes if r is not None]
    print(len(routes_valid))
    pd.DataFrame(routes, columns=['shortest_length']).to_csv(
        'myCode\data\init_length\hour_0_init_length.csv', index=False)
