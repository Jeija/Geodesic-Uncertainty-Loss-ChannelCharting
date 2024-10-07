import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.sparse.csgraph
from tqdm.auto import tqdm
import tensorflow as tf
import scipy.special
import numpy as np
import sklearn
import keras
import queue

def find_shortest_paths(pairwise_dissimilarity_matrix, n_neighbors = 20):
    nbrs_alg = sklearn.neighbors.NearestNeighbors(n_neighbors = n_neighbors, metric="precomputed", n_jobs = -1)
    nbrs = nbrs_alg.fit(pairwise_dissimilarity_matrix)
    nbg = sklearn.neighbors.kneighbors_graph(nbrs, n_neighbors, metric = "precomputed", mode="distance")

    geodesic_predecessor_matrix = np.zeros((nbg.shape[0], nbg.shape[1]), dtype = np.int32)
    longest_shortest_path = 0

    def shortest_path_worker(todo_queue, output_queue):
        while True:
            index = todo_queue.get()
    
            if index == -1:
                output_queue.put((-1, None, None))
                break
    
            d, predecessors = scipy.sparse.csgraph.dijkstra(nbg, directed=False, indices=index, return_predecessors=True)
            del d
    
            # Now also compute longest found path from predecessor vector
            # Unfortunately, scipy.sparse.csgraph.dijkstra does not return that information...
            current = np.arange(len(predecessors))
    
            pathhops = 0
            while np.any(current != -9999):
                current[current != -9999] = predecessors[current[current != -9999]]
                pathhops = pathhops + 1
            
            output_queue.put((index, predecessors, pathhops))
    
    with tqdm(total = nbg.shape[0]**2) as pbar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()
    
        for i in range(nbg.shape[0]):
            todo_queue.put(i)
        
        for i in range(mp.cpu_count()):
            todo_queue.put(-1)
            p = mp.Process(target = shortest_path_worker, args = (todo_queue, output_queue))
            p.start()
    
        finished_processes = 0
        while finished_processes != mp.cpu_count():
            i, p, l = output_queue.get()
    
            if i == -1:
                finished_processes = finished_processes + 1
            else:
                geodesic_predecessor_matrix[i,:] = p
                longest_shortest_path = max(longest_shortest_path, l)
                pbar.update(len(p))

    del nbg
    del nbrs
    del nbrs_alg
    
    return geodesic_predecessor_matrix, longest_shortest_path

class GaussianDissimilarityModel:
    def __init__(self):
        self.funcs_mean_variance_along_path = []
        self.funcs_dissimilarity_realization = []

    def add_metric(self, func_dissimilarity_realization, func_mean_variance_along_path):
        self.funcs_dissimilarity_realization.append(func_dissimilarity_realization)
        self.funcs_mean_variance_along_path.append(func_mean_variance_along_path)

    def generate_short_paths(self, realization_count = 5):
        self.predecessor_matrices = np.zeros((realization_count, datapoint_count, datapoint_count), dtype = np.int32)
        self.dissimilarity_matrix_choices = np.zeros((realization_count, datapoint_count, datapoint_count), dtype = np.int8)
        self.longest_shortest_path = 0

        # TODO: Do not need to run shortest path algorithm for every (realization_index, datapoint_index) pair,
        # good enough to run it for sufficiently many (randomly selected) realizations, datapoint indices.
        # Reducing the number of shortest path algorithm executions could significantly speed up the Channel Charting
        # process.
        
        # Does not return anything, but does the processing...
        # Rounds determines how many times realizations should be drawn randomly
        for realization_index in tqdm(range(realization_count)):
            print("Generating dissimilarity realizations...")
            dissimilarity_metrics_count = len(self.funcs_dissimilarity_realization)
            realizations = np.zeros((datapoint_count, datapoint_count, dissimilarity_metrics_count))
            for i, realization_func in enumerate(tqdm(self.funcs_dissimilarity_realization)):
                realizations[:,:,i] = realization_func()

            # For every datapoint pair, select smallest dissimilarity realization
            print("Choosing smallest dissimilarity realization pair-wise...")
            dissimilarity_matrix_choice = np.argmin(realizations, axis = -1, keepdims = True)
            pairwise_dissimilarity_matrix = np.take_along_axis(realizations, dissimilarity_matrix_choice, axis = -1)[:,:,0]
            self.dissimilarity_matrix_choices[realization_index] = dissimilarity_matrix_choice[:,:,0]

            # Run shortest path algorithm
            print("Running shortest path algorithm...")
            self.predecessor_matrices[realization_index], longest_shortest_path = find_shortest_paths(pairwise_dissimilarity_matrix)#
            self.longest_shortest_path = max(self.longest_shortest_path, longest_shortest_path)

            del pairwise_dissimilarity_matrix
            del dissimilarity_matrix_choice
            del realizations

    def get_longest_shortest_path(self):
        return self.longest_shortest_path
    
    def get_short_paths(self, starts, targets, subsampled_pathhops):
        # returns (paths, path_hops, path_means, path_variances)
        # where paths is of shape (path_count, maximum path length) and all others are of shape path_count
        realization_choice = np.random.randint(self.predecessor_matrices.shape[0], size = len(starts))

        current = np.copy(np.asarray(starts))
        paths = np.zeros((len(starts), self.longest_shortest_path), dtype = np.int32)
        path_hops = np.zeros(len(starts), dtype = np.int32)

        for i in range(self.longest_shortest_path):
            paths[:, i] = current
            previous = current
            active = (current != targets)

            current[active] = self.predecessor_matrices[realization_choice[active], targets[active], current[active]]
            path_hops[np.logical_and(active, current == targets)] = i + 1

        # Compute mean total dissimilarity as well as uncertainty about it (variance) along paths
        # Use provided models to compute means / variances for individual dissimilarity types
        # Assume that dissimilarity models are independent, i.e., variances are just added up
        paths_sections_a = paths[:,:-1]
        paths_sections_b = paths[:,1:]
    
        dissim_choice = self.dissimilarity_matrix_choices[realization_choice[:, np.newaxis], paths_sections_a, paths_sections_b]
        
        # Assume that p(d) from different models are entirely uncorrelated
        total_dissimilarity_means = np.zeros(len(paths))
        total_dissimilarity_variances = np.zeros(len(paths))

        for type in range(len(self.funcs_mean_variance_along_path)):
            means, variances = self.funcs_mean_variance_along_path[type](paths_sections_a, paths_sections_b, dissim_choice == type)
            total_dissimilarity_means += means
            total_dissimilarity_variances += variances
        
        # Subsample paths
        if subsampled_pathhops is not None:
            for i in range(len(paths)):
                l = min(subsampled_pathhops, path_hops[i])
                paths[i,:l+1] = paths[i, np.linspace(0, path_hops[i], l+1, dtype = np.int32)]
                paths[i,l+1:] = paths[i, -1]
                path_hops[i] = l
    
        return paths, path_hops, total_dissimilarity_means, total_dissimilarity_variances

GDM = GaussianDissimilarityModel()