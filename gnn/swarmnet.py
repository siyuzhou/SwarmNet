import numpy as np
import tensorflow as tf
from tensorflow import keras

from .modules import *
from .utils import fc_matrix


class SwarmNet(keras.Model):
    def __init__(self, params):
        super().__init__(name='SwarmNet')

        # NOTE: For the moment assume Conv1D is always applied
        self.pred_steps = params['pred_steps']
        self.time_seg_len = 2 * len(params['cnn']['filters']) + 1

        self.conv1d = Conv1D(params['cnn']['filters'], name='Conv1D')

        self.edge_encoder = MLP(params['mlp']['hidden_units'], name='edge_encoder')
        self.node_encoder = MLP(params['mlp']['hidden_units'], name='node_encoder')
        self.node_decoder = MLP(params['mlp']['hidden_units'], name='node_decoder')

        self.dense = keras.layers.Dense(params['ndims'], name='out_layer')

        edges = fc_matrix(params['nagents'])

        self.node_aggr = NodeAggregator(edges)
        self.edge_aggr = EdgeAggregator(edges)

    def _pred_next(self, time_segs, edge_type=None):
        # NOTE: For the moment, ignore edge_type.
        condensed_state = self.conv1d(time_segs)
        # condensed_state shape [batch, num_agents, 1, filters]

        # Form edges. Shape [batch, num_edges, 1, filters]
        edge_msg = self.node_aggr(condensed_state)
        edge_msg = self.edge_encoder(edge_msg)

        # Edge aggregation. Shape [batch, num_nodes, 1, filters]
        node_msg = self.edge_aggr(edge_msg)

        # The last state in each timeseries of the stack.
        prev_state = time_segs[:, :, -1:, :]
        # Skip connection
        node_state = tf.concat([prev_state, node_msg], axis=-1)
        node_state = self.node_decoder(node_state)

        # Predicted difference added to the prev state.
        next_state = self.dense(node_state) + prev_state
        return next_state

    def call(self, time_segs, edge_type=None):
        # NOTE: For the moment, ignore edge_type
        # time_segs shape [batch, time_seg_len, num_agents, ndims]
        # Transpose to [batch, num_agents, time_seg_len,ndims]
        print(time_segs.shape.as_list())
        extended_time_segs = tf.transpose(time_segs, [0, 2, 1, 3])

        for i in range(self.pred_steps):
            next_state = self._pred_next(extended_time_segs[:, :, i:, :], edge_type)
            extended_time_segs = tf.concat([extended_time_segs, next_state], axis=2)

        # Transpose back to [batch, time_seg_len+pred_steps, num_agetns, ndims]
        extended_time_segs = tf.transpose(extended_time_segs, [0, 2, 1, 3])

        # Return only the predicted part of extended_time_segs
        return extended_time_segs[:, self.time_seg_len:, :, :]
