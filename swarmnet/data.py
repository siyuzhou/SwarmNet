import os
import glob
import numpy as np
from . import utils


def stack_time_series(time_series, seg_len, axis=2):
    # time_series shape [num_sims, time_steps, num_agents, ndims]
    time_steps = time_series.shape[1]
    return np.stack([time_series[:, i:time_steps+1-seg_len+i, :, :] for i in range(seg_len)],
                    axis=axis)


def load_data(data_path, transpose=None, load_time=False, prefix='train', size=None, padding=None):
    if not os.path.exists(data_path):
        raise ValueError(f"path '{data_path}' does not exist")

    timeseries_files = sorted(glob.glob(os.path.join(data_path, f'{prefix}_timeseries*.npy')))

    if not timeseries_files:
        raise FileNotFoundError(f"no files matching pattern {prefix}_timeseries*.npy found")

    all_data = []
    for timeseries_f in timeseries_files:
        timeseries = np.load(timeseries_f)

        if transpose:
            timeseries = np.transpose(timeseries, transpose)

        data = timeseries.astype(np.float32)

        num_nodes = data.shape[2]
        # Add padding to num_nodes dim if `padding` is not None.
        if padding is not None and padding > num_nodes:
            pad_len = padding - num_nodes
            data = np.pad(data, [(0, 0), (0, 0), (0, pad_len), (0, 0)],
                          mode='constant', constant_values=0)

        all_data.append(data)

    all_data = np.concatenate(all_data, axis=0)

    # Load edge data.
    edge_files = sorted(glob.glob(os.path.join(data_path, f'{prefix}_edge*.npy')))

    if not edge_files:
        raise FileNotFoundError(f"no files matching pattern {prefix}_edges*.npy found")

    all_edges = []
    for edge_f in edge_files:
        # Edge data.
        edge_data = np.load(edge_f).astype(np.int)

        # Padding
        num_nodes = edge_data.shape[1]
        if padding is not None and padding > num_nodes:
            pad_len = padding - num_nodes
            edge_data = np.pad(edge_data, [(0, 0), (0, pad_len), (0, pad_len)],
                               mode='constant', constant_values=0)

        all_edges.append(edge_data)

    all_edges = np.concatenate(all_edges, axis=0)

    # Truncate data samples if `size` is given.
    if size:
        all_data = all_data[:size]
        all_edges = all_edges[:size]

    # Load time labels only when required.
    if load_time:
        time_files = sorted(glob.glob(os.path.join(data_path, f'{prefix}_time*.npy')))

        if not time_files:
            raise FileNotFoundError(f"no files matching pattern {prefix}_time*.npy found")

        all_times = []
        for time_f in time_files:
            time_data = np.load(time_f).astype(np.float32)
            all_times.append(time_data)

        all_times = np.concatenate(all_times, axis=0)

        if size:
            all_times = all_times[:size]

        return all_data, all_edges, all_times

    return all_data, all_edges


def preprocess_data(data, seg_len=1, pred_steps=1, edge_type=1, ground_truth=True):
    time_series, edges = data[:2]
    time_steps, num_nodes, ndims = time_series.shape[1:]

    if (seg_len + pred_steps > time_steps):
        if ground_truth:
            raise ValueError('time_steps in data not long enough for seg_len and pred_steps')
        else:
            stop = 1
    else:
        stop = -pred_steps

    edge_label = edge_type + 1  # Accounting for "no connection"

    # time_series shape [num_sims, time_steps, num_nodes, ndims]
    # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, seg_len, num_nodes, ndims]
    time_segs_stack = stack_time_series(time_series[:, :stop, :, :],
                                        seg_len)
    time_segs = time_segs_stack.reshape([-1, seg_len, num_nodes, ndims])
    if ground_truth:
        # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, num_nodes, ndims]
        expected_time_segs_stack = stack_time_series(time_series[:, seg_len:, :, :],
                                                     pred_steps)
        assert (time_segs_stack.shape[1] == expected_time_segs_stack.shape[1]
                == time_steps - seg_len - pred_steps + 1)
        expected_time_segs = expected_time_segs_stack.reshape([-1, pred_steps, num_nodes, ndims])
    else:
        expected_time_segs = None

    edges_one_hot = utils.one_hot(edges, edge_label, np.float32)
    edges_one_hot = np.repeat(edges_one_hot, time_segs_stack.shape[1], axis=0)

    if len(data) > 2:
        time_stamps = data[2]

        time_stamps_stack = stack_time_series(time_stamps[:, :stop], seg_len)
        time_stamps_segs = time_stamps_stack.reshape([-1, seg_len])

        if ground_truth:
            expected_time_stamps_stack = stack_time_series(
                time_stamps[:, seg_len:], pred_steps)
            expected_time_stamps_segs = expected_time_stamps_stack.reshape([-1, pred_steps])
        else:
            expected_time_stamps_segs = None

        return [time_segs, edges_one_hot], expected_time_segs, [time_stamps_segs, expected_time_stamps_segs]

    return [time_segs, edges_one_hot], expected_time_segs
