import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.sparse.csgraph
from tqdm.auto import tqdm
import sklearn.neighbors
import tensorflow as tf
import scipy.special
import numpy as np
import sklearn
import keras
import queue

def find_shortest_paths(pairwise_dissimilarity_matrix, target_nodes = None, n_neighbors = 40):
    nbrs_alg = sklearn.neighbors.NearestNeighbors(n_neighbors = n_neighbors, metric="precomputed", n_jobs = -1)
    nbrs = nbrs_alg.fit(pairwise_dissimilarity_matrix)
    nbg = sklearn.neighbors.kneighbors_graph(nbrs, n_neighbors, metric = "precomputed", mode="distance")

    if target_nodes is None:
        target_nodes = np.arange(nbg.shape[0], dtype = np.int32)

    geodesic_predecessor_matrix = np.zeros((target_nodes.shape[0], nbg.shape[1]), dtype = np.int32)
    longest_shortest_path = 0

    def shortest_path_worker(todo_queue, output_queue):
        while True:
            index = todo_queue.get()
    
            if index == -1:
                output_queue.put((-1, None, None))
                break
    
            d, predecessors = scipy.sparse.csgraph.dijkstra(nbg, directed=False, indices=target_nodes[index], return_predecessors=True)
            del d
    
            # Now also compute longest found path from predecessor vector
            # Unfortunately, scipy.sparse.csgraph.dijkstra does not return that information...
            current = np.arange(len(predecessors))
    
            pathhops = 0
            while np.any(current != -9999):
                current[current != -9999] = predecessors[current[current != -9999]]
                pathhops = pathhops + 1
            
            output_queue.put((index, predecessors, pathhops))
    
    with tqdm(total = len(target_nodes) * nbg.shape[0]) as pbar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()

        for i in range(len(target_nodes)):
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
    def __init__(self, metrics):
        self.metrics = metrics

        self.datapoint_count = metrics[0].get_datapoint_count()
        for metric in metrics[1:]:
            assert(metric.get_datapoint_count() == self.datapoint_count)

    def generate_short_paths(self, total_path_count = 40000, realization_count = 8):
        assert(total_path_count % realization_count == 0)
        paths_per_realization = total_path_count // realization_count

        self.predecessor_matrix = np.zeros((total_path_count, self.datapoint_count), dtype = np.int32)
        self.target_nodes = np.random.randint(self.datapoint_count, size = total_path_count)
        self.dissimilarity_matrix_choices = np.zeros((total_path_count, self.datapoint_count), dtype = np.int8)
        self.longest_shortest_path = 0

        # Does not return anything, but does the processing...
        # Rounds determines how many times realizations should be drawn randomly
        for realization_index in tqdm(range(realization_count)):
            first_path_index = realization_index * paths_per_realization
            last_path_index = (realization_index + 1) * paths_per_realization
            
            print("Generating dissimilarity realizations...")
            dissimilarity_metrics_count = len(self.metrics)
            realizations = np.zeros((self.datapoint_count, self.datapoint_count, dissimilarity_metrics_count))
            for i, metric in enumerate(tqdm(self.metrics)):
                realizations[:,:,i] = metric.get_realization()

            # For every datapoint pair, select smallest dissimilarity realization
            print("Choosing smallest dissimilarity realization pair-wise...")
            dissimilarity_matrix_choice = np.argmin(realizations, axis = -1, keepdims = True)
            pairwise_dissimilarity_matrix = np.take_along_axis(realizations, dissimilarity_matrix_choice, axis = -1)[:,:,0]

            # Run shortest path algorithm
            # dissimilarity_matrix_choices stores which type of dissimilarity (velocity model, adp model, ...) was used to go from datapoint x along
            # the path towards the target datapoint to the next hop.
            # It has shape (total_path_count, self.datapoint_count), so the first axis determines the path we are on (and hence also the target datapoint) and the
            # second axis determines the datapoint (node) from which the current hop starts.
            print("Running shortest path algorithm...")
            current_target_nodes = self.target_nodes[first_path_index:last_path_index]
            pred_matrix_for_realization, longest_shortest_path = find_shortest_paths(pairwise_dissimilarity_matrix, current_target_nodes)
            self.predecessor_matrix[first_path_index:last_path_index] = pred_matrix_for_realization
            self.dissimilarity_matrix_choices[first_path_index:last_path_index] = dissimilarity_matrix_choice[np.arange(self.datapoint_count)[np.newaxis,:], pred_matrix_for_realization][...,0]
            self.longest_shortest_path = max(self.longest_shortest_path, longest_shortest_path)

            del pairwise_dissimilarity_matrix
            del dissimilarity_matrix_choice
            del realizations

    def get_longest_shortest_path(self):
        return self.longest_shortest_path
    
    def get_random_short_paths(self, path_count, subsampled_pathhops = None):
        # returns (path_targets, path_sources, paths, path_hops, path_means, path_variances)
        # where paths is of shape (path_count, maximum path length) and all others are of shape path_count

        # Target and source indices to cached predecessor matrix
        # Source indices are also datapoint indices, but target indices must be translated to datapoint indices
        # using self.target_nodes[path_target_indices]
        path_target_indices = np.random.randint(self.predecessor_matrix.shape[0], size = path_count)
        path_source_indices = np.random.randint(self.datapoint_count, size = path_count)

        # Prevent pairs where both indices refer to the same datapoint
        path_source_indices[path_source_indices == self.target_nodes[path_target_indices]] = (path_source_indices[path_source_indices == self.target_nodes[path_target_indices]] + 1) % self.datapoint_count
        
        current = np.copy(path_source_indices)
        paths = np.zeros((len(current), self.longest_shortest_path), dtype = np.int32)
        path_hops = np.zeros(len(current), dtype = np.int32)

        for i in range(self.longest_shortest_path):
            paths[:, i] = current
            previous = current
            active = (current != self.target_nodes[path_target_indices])

            current[active] = self.predecessor_matrix[path_target_indices[active], current[active]]
            path_hops[np.logical_and(active, current == self.target_nodes[path_target_indices])] = i + 1

        # Compute mean total dissimilarity as well as uncertainty about it (variance) along paths
        # Use provided models to compute means / variances for individual dissimilarity types
        # Assume that dissimilarity models are independent, i.e., variances are just added up
        dissim_choice = np.take_along_axis(self.dissimilarity_matrix_choices[path_target_indices], paths[:,:-1], 1)
        
        # Assume that p(d) from different models are entirely uncorrelated
        total_dissimilarity_means = np.zeros(len(paths))
        total_dissimilarity_variances = np.zeros(len(paths))

        for metric_type, metric in enumerate(self.metrics):
            means, variances = metric.mean_variance_along_path(paths, dissim_choice == metric_type)
            total_dissimilarity_means += means
            total_dissimilarity_variances += variances
        
        # Subsample paths such that there are no more than subsampled_pathhops hops
        if subsampled_pathhops is not None:
            for i in range(len(paths)):
                l = min(subsampled_pathhops, path_hops[i])
                paths[i,:l+1] = paths[i, np.linspace(0, path_hops[i], l+1, dtype = np.int32)]
                paths[i,l+1:] = paths[i, -1]
                path_hops[i] = l
    
        return paths, path_hops, total_dissimilarity_means, total_dissimilarity_variances

class CSIProviderLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_csi(self, csi):
        self.csi = tf.constant(csi)

    def call(self, index):
        csi_cplx = tf.gather(self.csi, index)
        return tf.stack([tf.math.real(csi_cplx), tf.math.imag(csi_cplx)], axis = -1)

    def get_config(self):
        return super().get_config()

class FeatureEngineeringLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, csi):
        # Compute sample correlations for any combination of two antennas in the whole system
        # for the same datapoint and time tap.
        csi = tf.complex(csi[...,0], csi[...,1])
        sample_autocorrelations = tf.einsum("dbmt,dbnt->dtbmn", csi, tf.math.conj(csi))
        return tf.stack([tf.math.real(sample_autocorrelations), tf.math.imag(sample_autocorrelations)], axis = -1)

    def get_config(self):
        return super().get_config()

class ChannelChartingLoss(keras.losses.Loss):
    def __init__(self, timestamps, name="CCLoss"):
        super().__init__(name=name)
        self.timestamps = tf.constant(timestamps)

    def acceleration(self, pred_positions):
        pred_velocities = tf.experimental.numpy.diff(pred_positions, axis = 0) / tf.experimental.numpy.diff(self.timestamps)[:,tf.newaxis]
        pred_accelerations = tf.experimental.numpy.diff(pred_velocities, axis = 0) / tf.experimental.numpy.diff(self.timestamps)[:-1,tf.newaxis]
        pred_accelerations_abs = tf.math.sqrt(tf.math.reduce_sum(pred_accelerations**2, axis = -1) + 1e-6)

        return tf.math.reduce_mean(tf.square(pred_accelerations_abs))
    
    def call(self, y_true, y_pred):
        # This is an ugly workaround, the loss function always gets y_pred as float, convert back to integer for index
        # This works as long as CSI tensor is not absolutely huge (16M+ entries), which can be assumed.
        path_hops = tf.cast(y_true[:,0], tf.int32)
        path_means = y_true[:,1]
        path_variances = y_true[:,2]
        paths = tf.cast(y_true[:,3:], tf.int32)
        path_end_indices = tf.transpose([tf.range(tf.shape(y_true)[0]), path_hops])

        index_A = tf.cast(y_true[:,3], tf.int32)
        index_B = tf.cast(tf.gather_nd(paths, path_end_indices), tf.int32)

        pos_A = tf.gather(y_pred, index_A)
        pos_B = tf.gather(y_pred, index_B)

        # Acceleration loss
        acceleration_loss = self.acceleration(y_pred)

        # Geodesic loss
        # paths has shape (BATCHSIZE, longest_shorest_path)
        # path_positions has shape (BATCHSIZE, longest_shorest_path, 2), where last dimension is x/y position
        # path_positions_delta has shape (BATCHSIZE, longest_shorest_path - 1, 2), where last dimension is x/y delta
        # path_distances has shape (BATCHSIZE, longest_shorest_path - 1)
        # endpoint_distances has shape (BATCHSIZE)
        path_positions = tf.gather(y_pred, paths)
        path_positions_delta = path_positions[:,1:,:] - path_positions[:,:-1,:]
        path_distances = tf.math.sqrt(tf.math.reduce_sum(path_positions_delta**2, axis = -1) + 1e-6)
        geodesic_distance = tf.math.reduce_sum(path_distances, axis = 1)
        geodesic_loss = tf.reduce_mean(tf.square(geodesic_distance - path_means) / (path_variances + 1e-6))

        # Make sure all path distances are smaller the endpoint distances
        # Otherwise, shortest path would just go from endpoint to endpoint
        endpoint_distances = tf.math.sqrt(tf.math.reduce_sum(tf.square(pos_A - pos_B), axis = 1))
        geodesic_loss = geodesic_loss + 0.01 * tf.math.reduce_sum(tf.math.maximum(path_distances - endpoint_distances[:,tf.newaxis], 0))

        # Combination
        return geodesic_loss + 1e-4 * acceleration_loss

class ChannelChart:
    def __init__(self, GDM, csi_time_domain, timestamps, batch_size = 3000, learning_rate_initial = 1e-2, learning_rate_final = 1e-4, min_pathhops = 1, max_pathhops = 30, training_batches = 2000, plot_callback = None):
        # Build forward charting function
        fcf_input = keras.Input(shape=csi_time_domain.shape[1:] + (2,), name="input", dtype = tf.float32)
        fcf_output = FeatureEngineeringLayer()(fcf_input)
        fcf_output = keras.layers.Flatten()(fcf_output)
        fcf_output = keras.layers.Dense(1024, activation = "relu")(fcf_output)
        fcf_output = keras.layers.BatchNormalization()(fcf_output)
        fcf_output = keras.layers.Dense(512, activation = "relu")(fcf_output)
        fcf_output = keras.layers.BatchNormalization()(fcf_output)
        fcf_output = keras.layers.Dense(256, activation = "relu")(fcf_output)
        fcf_output = keras.layers.BatchNormalization()(fcf_output)
        fcf_output = keras.layers.Dense(128, activation = "relu")(fcf_output)
        fcf_output = keras.layers.BatchNormalization()(fcf_output)
        fcf_output = keras.layers.Dense(64, activation = "relu")(fcf_output)
        fcf_output = keras.layers.BatchNormalization()(fcf_output)
        fcf_output = keras.layers.Dense(2, activation = "linear")(fcf_output)
        self.fcf = keras.Model(inputs=fcf_input, outputs=fcf_output, name = "ForwardChartingFunction")

        # Prepend CSI provider layer during training
        training_input = keras.layers.Input(shape = (), dtype = tf.int64)
        csiprov = CSIProviderLayer(dtype = tf.int64)
        csiprov.set_csi(csi_time_domain)
        csi_layer = csiprov(training_input)
        output = self.fcf(csi_layer)
        training_model = tf.keras.models.Model(training_input, output, name = "TrainingModel")

        # Random path generator
        def random_pair_batch_generator():
            batch_count = 0
            while True:
                batch_count = batch_count + 1
                all_datapoints = np.arange(csi_time_domain.shape[0])

                # Determine number of hops for current subsampling ratio
                pathhops = min(max(min_pathhops, int(batch_count / training_batches * max_pathhops)), max_pathhops)

                # Generate random short paths and assemble y_true, consisting of batch_size paths, each made up of
                # * number of path hops
                # * mean value of dissimilarity random variable
                # * variance of  dissimilarity random variable
                # * datapoint indices along path; ends with repeating last index if too few hops
                paths, path_hops, total_dissimilarity_means, total_dissimilarity_variances = GDM.get_random_short_paths(batch_size, pathhops)
                paths = paths[:,:max_pathhops + 1]
                y_true = np.hstack([path_hops[:,np.newaxis], total_dissimilarity_means[:,np.newaxis], total_dissimilarity_variances[:,np.newaxis], paths])

                yield all_datapoints, tf.cast(y_true, tf.float32)

        random_path_dataset = tf.data.Dataset.from_generator(random_pair_batch_generator,
            output_signature=(tf.TensorSpec(shape=(csi_time_domain.shape[0]), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, 1 + 2 + max_pathhops + 1), dtype=tf.float32)))

        # Train Forward Charting Function
        training_loss = ChannelChartingLoss(timestamps)

        learning_rate_decay_factor = (learning_rate_final / learning_rate_initial)**(100000/6000000)
        #learning_rate_decay_factor = learning_rate_final / learning_rate_initial

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=learning_rate_initial,
                        decay_steps=100000,
                        decay_rate=learning_rate_decay_factor,
                        staircase=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

        # Metrics and callbacks
        train_callbacks = [keras.callbacks.TerminateOnNaN()]
        train_metrics = []
        if plot_callback is not None:
            train_callbacks.append(plot_callback)
            train_metrics.append(plot_callback.metric)

        # Compile and fit
        training_model.compile(loss = training_loss, optimizer = optimizer, metrics = train_metrics)
        training_model.fit(random_path_dataset, steps_per_epoch = training_batches, callbacks = train_callbacks)

    def predict(self, csi_time_domain):
        csi_time_domain_tensor = tf.constant(csi_time_domain)
        csi_time_domain_tensor_re_im = tf.stack([tf.math.real(csi_time_domain_tensor), tf.math.imag(csi_time_domain_tensor)], axis = -1)
        return self.fcf.predict(csi_time_domain_tensor_re_im)
