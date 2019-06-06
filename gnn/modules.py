import tensorflow as tf
from tensorflow import keras
import numpy as np


class MLP(keras.layers.Layer):
    def __init__(self, units):
        if not units:
            raise ValueError("'units' must not be empty")

        super().__init__()
        self.hidden_layers = []

        for unit in units[:-1]:
            self.hidden_layers.append(keras.layers.Dense(unit, activation='relu'))
            # NOTE: Support for dropout to be added.
            # NOTE: Does one need BatchNorm?

        self.hidden_layers.append(keras.layers.Dense(units[-1], activation='relu'))

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x


class NodeAggregator(keras.layers.Layer):
    """
    Pass message from both nodes to the edge in between.
    """

    def __init__(self, edges):
        super().__init__()
        # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
        edge_sources, edge_targets = np.where(edges)
        self.edge_sources = tf.constant(edge_sources)
        self.edge_targets = tf.constant(edge_targets)

    def call(self, node_msg):
        # node_msg shape [batch, num_agents, 1, out_units].
        msg_from_source = tf.transpose(tf.tensordot(node_msg, self.edge_sources, axes=[[1], [1]]),
                                       perm=[0, 3, 1, 2])
        msg_from_target = tf.transpose(tf.tensordot(node_msg, self.edge_targets, axes=[[1], [1]]),
                                       perm=[0, 3, 1, 2])
        # msg_from_source and msg_from_target in shape [batch, num_edges, 1, out_units]
        edge_msg = tf.concat([msg_from_source, msg_from_target], axis=-1)
        return edge_msg


class EdgeAggregator(keras.layers.Layer):
    """
    Pass message from incoming edges to the node.
    """

    def __init__(self, edges):
        super().__init__()

        # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
        edge_targets = np.where(edges)[1]
        self.edge_targets = tf.constant(edge_targets)

    def call(self, edge_msg):
        # edge_msg shape [batch, num_edges, 1, out_units]
        node_msg = tf.transpose(tf.tensordot(edge_msg, self.edge_targets, axes=[[1], [0]]),
                                perm=[0, 3, 1, 2])  # Shape [batch, num_agents, 1, out_units].
        return node_msg


class Conv1D(keras.layers.Layer):
    """
    Condense and abstract the time segments.
    """

    def __init__(self, filters):
        super().__init__()
        # time segment length before being reduced to 1 by Conv1D
        self.seg_len = 2 * len(filters) + 1

        self.conv1d_layers = []
        for channels in filters:
            self.conv1d_layers.append(keras.layers.Conv1D())

    def call(self, time_segs):
        # Reshape to [batch*num_agents, time_seg_len, ndims], since conv1d only accept
        # tensor with 3 dimensions.
        _, num_agents, time_seg_len, ndims = time_segs.shape.as_list()
        state = tf.reshape(time_segs, shape=[-1, time_seg_len, ndims])

        # Node state encoder with 1D convolution along timesteps and across ndims as channels.
        encoded_state = state
        for conv1d in self.conv1d_layers:
            encoded_state = conv1d(encoded_state)

        encoded_state = tf.reshape(encoded_state, shape=[-1, num_agents, 1, filters])

        return encoded_state
