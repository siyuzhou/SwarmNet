import os

import numpy as np

from gnn import utils


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

        return data, edge_data

    return data
