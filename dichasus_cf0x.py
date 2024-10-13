import tensorflow as tf
import numpy as np
import json

inputpaths = [
    {
        "tfrecords" : "dichasus-cf0x-rev2/dichasus-cf02.tfrecords",
        "offsets" : "dichasus-cf0x-rev2/reftx-offsets-dichasus-cf02.json"
    },
    {
        "tfrecords" : "dichasus-cf0x-rev2/dichasus-cf03.tfrecords",
        "offsets" : "dichasus-cf0x-rev2/reftx-offsets-dichasus-cf03.json"
    },
    {
        "tfrecords" : "dichasus-cf0x-rev2/dichasus-cf04.tfrecords",
        "offsets" : "dichasus-cf0x-rev2/reftx-offsets-dichasus-cf04.json"
    }
]

spec = None

antenna_assignments = []
antenna_count = 0

with open("dichasus-cf0x-rev2/spec.json") as specfile:
    spec = json.load(specfile)
    for antenna in spec["antennas"]:
        antenna_count = antenna_count + sum([len(row) for row in antenna["assignments"]])
        antenna_assignments.append(antenna["assignments"])

def load_calibrate_timedomain(path, offset_path, array_to_cut = None):
    offsets = None
    with open(offset_path, "r") as offsetfile:
        offsets = json.load(offsetfile)
    
    def record_parse_function(proto):
        record = tf.io.parse_single_example(
            proto,
            {
                "csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
                "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=""),
                "time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
            },
        )

        csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (antenna_count, 1024, 2))
        csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
        csi = tf.signal.fftshift(csi, axes=1)

        # Convert from rev2 back to rev1 format (normalize STO)
        incr = tf.cast(tf.math.angle(tf.math.reduce_sum(csi[:,1:] * tf.math.conj(csi[:,:-1]))), tf.complex64)
        csi = csi * tf.exp(-1.0j * incr * tf.cast(tf.range(csi.shape[-1]), tf.complex64))[tf.newaxis,:]

        position = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3))
        time = tf.ensure_shape(record["time"], ())

        return csi, position[:2], time

    def order_by_antenna_assignments(csi, pos, time):
        csi = tf.stack([[tf.gather(csi, antenna_indices) for antenna_indices in array] for array in antenna_assignments])
        return csi, pos, time

    def cut_array(to_cut):
        def cut_func(csi, pos, time):
            return tf.gather(csi, to_cut)[tf.newaxis], pos, time

        return cut_func

    def apply_calibration_cut(to_cut):
        if to_cut is None:
            to_cut = np.arange(len(spec["antennas"]))
        else:
            to_cut = np.asarray([to_cut])

        def apply_calibration(csi, pos, time):
            sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi)[-1], dtype = tf.float32) / tf.cast(tf.shape(csi)[-1], tf.float32), axes = 0)
            cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[-1], dtype = tf.float32), axes = 0)
    
            compensation = tf.exp(tf.complex(0.0, sto_offset + cpo_offset))
            compensation_by_antenna = tf.stack([[tf.gather(compensation, antenna_indices) for antenna_indices in array] for array in antenna_assignments])
            compensation_by_antenna = tf.gather(compensation_by_antenna, to_cut)
            csi = csi * compensation_by_antenna
    
            return csi, pos, time

        return apply_calibration

    def csi_time_domain(csi, pos, time):
        csi = tf.signal.fftshift(tf.signal.ifft(tf.signal.fftshift(csi, axes=-1)),axes=-1)

        return csi, pos, time

    def cut_out_taps(tap_start, tap_stop):
        def cut_out_taps_func(csi, pos, time):
            return csi[...,tap_start:tap_stop], pos, time

        return cut_out_taps_func

    dataset = tf.data.TFRecordDataset(path)
    
    dataset = dataset.map(record_parse_function, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.map(order_by_antenna_assignments, num_parallel_calls = tf.data.AUTOTUNE)

    if array_to_cut is not None:
        dataset = dataset.map(cut_array(array_to_cut))
    
    dataset = dataset.map(apply_calibration_cut(array_to_cut), num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.map(csi_time_domain, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.map(cut_out_taps(507, 520), num_parallel_calls = tf.data.AUTOTUNE)

    return dataset

def load_inputpaths(array_to_cut):
    dichasus_cf0x = load_calibrate_timedomain(inputpaths[0]["tfrecords"], inputpaths[0]["offsets"], array_to_cut)
    
    for path in inputpaths[1:]:
        dichasus_cf0x = dichasus_cf0x.concatenate(load_calibrate_timedomain(path["tfrecords"], path["offsets"], array_to_cut))
    
    testset = dichasus_cf0x.shard(4, 2)
    trainingset = dichasus_cf0x.shard(4, 0)

    return testset, trainingset

testset, trainingset = load_inputpaths(None)

singlearray_testsets = []
singlearray_trainingsets = []

for array in range(4):
    sa_testset, sa_trainingset = load_inputpaths(array)
    singlearray_testsets.append(sa_testset)
    singlearray_trainingsets.append(sa_trainingset)
