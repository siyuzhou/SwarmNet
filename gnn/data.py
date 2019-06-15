import os
import numpy as np
from . import utils


def load_data(data_path, transpose=None, edge=True, prefix='train'):
    loc = np.load(os.path.join(data_path, '{}_position.npy'.format(prefix)))
    vel = np.load(os.path.join(data_path, '{}_velocity.npy'.format(prefix)))

    if transpose:
        loc = np.transpose(loc, transpose)
        vel = np.transpose(vel, transpose)

    data = np.concatenate([loc, vel], axis=-1).astype(np.float32)

    if edge:
        # Edge data.
        edge_data = np.load(os.path.join(data_path, '{}_edge.npy'.format(prefix))).astype(np.int)

        instances, n_agents, _ = edge_data.shape
        edge_data = np.stack([edge_data[i][np.where(utils.fc_matrix(n_agents))]
                              for i in range(instances)], 0)
        # Shape [instances, n_edges]
        return data, edge_data

    return data


def preprocess_data(data, seg_len, pred_steps, edge_type=None):
    if edge_type > 1:
        time_series, edge_types = data
    else:
        time_series = data

    time_steps, nagents, ndims = time_series.shape[1:]
    # time_series shape [num_sims, time_steps, nagents, ndims]
    # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, seg_len, nagents, ndims]
    time_segs_stack = utils.stack_time_series(time_series[:, :-pred_steps, :, :],
                                              seg_len)
    # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, nagents, ndims]
    expected_time_segs_stack = utils.stack_time_series(time_series[:, seg_len:, :, :],
                                                       pred_steps)
    assert (time_segs_stack.shape[1] == expected_time_segs_stack.shape[1]
            == time_steps - seg_len - pred_steps + 1)

    time_segs = time_segs_stack.reshape([-1, seg_len, nagents, ndims])
    expected_time_segs = expected_time_segs_stack.reshape([-1, pred_steps, nagents, ndims])

    if edge_type > 1:
        edge_types = utils.one_hot(edge_types, edge_type, np.float32)
        # Shape [instances, n_edges, edge_type]
        n_edges = edge_types.shape[1]
        edge_types = np.stack([edge_types for _ in range(time_segs_stack.shape[1])], axis=1)
        edge_types = np.reshape(edge_types, [-1, n_edges, edge_type])

        return [time_segs, edge_types], expected_time_segs

    else:
        return [time_segs], expected_time_segs
