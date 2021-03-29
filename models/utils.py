# -*- coding: utf-8 -*-
# @Author: Bardbo
# @Date:   2021-01-06 16:06:45
# @Last Modified by:   Bardbo
# @Last Modified time: 2021-01-10 14:12:09
import math
import numpy as np
import torch
def load_data():

#     A = np.load("./data/manhattan_adj_mat.npy", allow_pickle=True)
    X = np.load("./data/manhattan_node_values_1d.npy", allow_pickle=True)
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 2))
    X = (X - means.reshape(1, -1, 1)) / stds.reshape(1, -1, 1)

    return X, means, stds

def load_4d_data():

#     A = np.load("./data/manhattan_adj_mat.npy", allow_pickle=True)
    X = np.load("./data/manhattan_node_values_4d.npy", allow_pickle=True)
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 2))
    X = (X - means.reshape(1, -1, 1)) / stds.reshape(1, -1, 1)

    return X, means, stds

def load_A():
    A = np.load("./data/manhattan_adj_mat.npy", allow_pickle=True)
    return A

def load_random_A():
    A = np.load("./data/manhattan_random_adj.npy", allow_pickle=True)
    return A

def load_random2_A():
    A = np.load("./data/manhattan_random2_adj.npy", allow_pickle=True)
    return A


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
#     A = A + np.diag(np.ones(A.shape[0]))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_wave = A_wave.astype(np.float32)
    return A_wave

def load_dtw_A():
    A = np.load("./data/manhattan_dtw_ds.npy", allow_pickle=True)
    return A

def get_normalized_dtw_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))*A.max()
#     A = A + np.diag(np.ones(A.shape[0]))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_wave = A_wave.astype(np.float32)
    return A_wave
    

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features, dtype=np.float32)), \
           torch.from_numpy(np.array(target, dtype=np.float32))
