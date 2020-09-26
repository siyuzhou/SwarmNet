import os
import glob
import numpy as np
from . import utils


def stack_time_series(time_series, seg_len, axis=2):
    # time_series shape [num_sims, time_steps, num_agents, ndims]
    time_steps = time_series.shape[1]
    return np.stack([time_series[:, i:time_steps+1-seg_len+i, :, :] for i in range(seg_len)],
                    axis=axis)


def _load_files(file_pattern, dtype, padding=None, pad_dims=None):
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"no files matching pattern {file_pattern} found")

    all_data = []
    for f in files:
        data = np.load(f).astype(dtype)

        if padding is not None and pad_dims is not None:
            pad_shape = [(0, padding - s if i in pad_dims else 0) for i, s in enumerate(data.shape)]
            data = np.pad(data, pad_shape, mode='constant', constant_values=0)

        all_data.append(data)

    return np.concatenate(all_data, axis=0)


def load_data(data_path, prefix='train', size=None, padding=None, load_time=False):
    if not os.path.exists(data_path):
        raise ValueError(f"path '{data_path}' does not exist")

    # Load timeseries data.
    timeseries_file_pattern = os.path.join(data_path, f'{prefix}_timeseries*.npy')
    all_data = _load_files(timeseries_file_pattern, np.float32, padding=padding, pad_dims=(2,))

    # Load edge data.
    edge_file_pattern = os.path.join(data_path, f'{prefix}_edge*.npy')
    all_edges = _load_files(edge_file_pattern, np.int, padding, pad_dims=(1, 2))

    shuffled_idx = np.random.permutation(len(all_data))
    # Truncate data samples if `size` is given.
    if size:
        samples = shuffled_idx[:size]

        all_data = all_data[samples]
        all_edges = all_edges[samples]

    # Load time labels only when required.
    if load_time:
        time_file_pattern = os.path.join(data_path, f'{prefix}_time*.npy')
        all_times = _load_files(time_file_pattern, np.float32)

        if size:
            samples = shuffled_idx[:size]

            all_times = all_times[samples]

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
