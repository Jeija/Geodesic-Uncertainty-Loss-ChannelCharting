#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sklearn.manifold
import tensorflow as tf
import numpy as np
import keras

def plot_colorized(positions, groundtruth_positions, title = None, show = True, alpha = 1.0):
    # Generate RGB colors for datapoints
    center_point = np.zeros(2, dtype = np.float32)
    center_point[0] = 0.5 * (np.min(groundtruth_positions[:, 0], axis = 0) + np.max(groundtruth_positions[:, 0], axis = 0))
    center_point[1] = 0.5 * (np.min(groundtruth_positions[:, 1], axis = 0) + np.max(groundtruth_positions[:, 1], axis = 0))
    NormalizeData = lambda in_data : (in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data))
    rgb_values = np.zeros((groundtruth_positions.shape[0], 3))
    rgb_values[:, 0] = 1 - 0.9 * NormalizeData(groundtruth_positions[:, 0])
    rgb_values[:, 1] = 0.8 * NormalizeData(np.square(np.linalg.norm(groundtruth_positions - center_point, axis=1)))
    rgb_values[:, 2] = 0.9 * NormalizeData(groundtruth_positions[:, 1])

    # Plot datapoints
    plt.figure(figsize=(6, 6))
    if title is not None:
        plt.title(title, fontsize=16)
    plt.scatter(positions[:, 0], positions[:, 1], c = rgb_values, alpha = alpha, s = 10, linewidths = 0)
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    if show:
        plt.show()

def affine_transform_channel_chart(groundtruth_pos, channel_chart_pos):
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    A, res, rank, s = np.linalg.lstsq(pad(channel_chart_pos), pad(groundtruth_pos), rcond = None)
    transform = lambda x: unpad(np.dot(pad(x), A))
    return transform(channel_chart_pos)

class PlotChartCallback(keras.callbacks.Callback):
    def __init__(self, groundtruth_positions, datapoint_count, batch_size, max_hops, update_period = 200):
        super().__init__()

        self.y_true = None
        self.y_pred = None

        self.groundtruth_positions = groundtruth_positions
        self.datapoint_count = datapoint_count
        self.batch_size = batch_size
        self.max_hops = max_hops

        self.update_period = update_period

    def set_model(self, model):
        self.training_model = model
        self.y_true = tf.Variable(np.zeros([self.batch_size, 1 + 2 + self.max_hops + 1]), dtype=tf.float32, shape=tf.TensorShape([self.batch_size, 1 + 2 + self.max_hops + 1]))
        self.y_pred = tf.Variable(np.zeros([self.datapoint_count, 2]), dtype=tf.float32, shape=tf.TensorShape([self.datapoint_count, 2]))

    def metric(self, y_true, y_pred):
        self.y_true.assign(y_true)
        self.y_pred.assign(y_pred)
        return 0

    def on_train_batch_end(self, batch, logs=None):
        print(self.training_model.optimizer.learning_rate)

        if batch % self.update_period == self.update_period - 1:
            pred_positions = self.y_pred.numpy()

            channel_chart_positions_transformed = affine_transform_channel_chart(self.groundtruth_positions, pred_positions)
            errorvectors = self.groundtruth_positions - channel_chart_positions_transformed
            errors = np.sqrt(errorvectors[:,0]**2 + errorvectors[:,1]**2)
            mae = np.mean(errors)
            cep = np.median(errors)
            
            plot_colorized(pred_positions, self.groundtruth_positions, title = f"Error Vectors, MAE = {mae:.4f}m, CEP = {cep:.4f}m", show = False)

            y_true_np = tf.cast(self.y_true, tf.int32).numpy()
            paths_indices = y_true_np[:,3:]

            for path_indices in paths_indices[:50]:
                path_positions = pred_positions[path_indices]
                plt.plot(path_positions[:,0], path_positions[:,1])
            
            plt.show()

    def on_train_end(self, logs=None):
        del self.y_pred


def continuity(*args, **kwargs):
    args = list(args)
    args[0], args[1] = args[1], args[0]
    return sklearn.manifold.trustworthiness(*args, **kwargs)

def kruskal_stress(X, X_embedded, *, metric="euclidean"):
    dist_X = sklearn.metrics.pairwise_distances(X, metric = metric)
    dist_X_embedded = sklearn.metrics.pairwise_distances(X_embedded, metric = metric)
    beta = np.divide(np.sum(dist_X * dist_X_embedded), np.sum(dist_X_embedded * dist_X_embedded))

    return np.sqrt(np.divide(np.sum(np.square((dist_X - beta * dist_X_embedded))), np.sum(dist_X * dist_X)))

def ct_tw_ks_on_subset(groundtruth_positions, channel_chart_positions, downsampling = 10):
    subset_indices = np.random.choice(range(len(groundtruth_positions)), len(groundtruth_positions) // downsampling)

    groundtruth_positions_subset = groundtruth_positions[subset_indices]
    channel_chart_positions_subset = channel_chart_positions[subset_indices]

    ct = continuity(groundtruth_positions_subset, channel_chart_positions_subset, n_neighbors = int(0.05 * len(groundtruth_positions_subset)))
    tw = sklearn.manifold.trustworthiness(groundtruth_positions_subset, channel_chart_positions_subset, n_neighbors = int(0.05 * len(groundtruth_positions_subset)))
    ks = kruskal_stress(groundtruth_positions_subset, channel_chart_positions_subset)

    return ct, tw, ks

def mean_absolute_error_transformed(groundtruth_positions, channel_chart_positions):
    def affine_transform_channel_chart(groundtruth_pos, channel_chart_pos):
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:,:-1]
        A, res, rank, s = np.linalg.lstsq(pad(channel_chart_pos), pad(groundtruth_pos), rcond = None)
        transform = lambda x: unpad(np.dot(pad(x), A))
        return transform(channel_chart_pos)

    channel_chart_positions_transformed = affine_transform_channel_chart(groundtruth_positions, channel_chart_positions)

    errorvectors = groundtruth_positions - channel_chart_positions_transformed
    errors = np.sqrt(errorvectors[:,0]**2 + errorvectors[:,1]**2)
    mae = np.mean(errors)
    cep = np.median(errors)

    return channel_chart_positions_transformed, errorvectors, errors, mae, cep

def plot_predecessors(groundtruth_positions, predecessors, subsampling = 100):
    plot_colorized(groundtruth_positions, groundtruth_positions, title="Predecessors", alpha = 0.5, show = False)

    paths = []
    current = np.arange(len(predecessors), dtype = np.int32)
    active = (current != -9999)
    while np.any(active):
        current[active] = predecessors[current[active]]
        active = (current != -9999)
        paths.append(groundtruth_positions[current])

    paths = np.asarray(paths)

    for start in range(len(predecessors))[::subsampling]:
        plt.plot(paths[:,start,0], paths[:,start,1], "r")

    plt.scatter(paths[-1,0,0], paths[-1,0,1], s = 100, zorder=3)
    plt.show()