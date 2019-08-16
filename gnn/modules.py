import tensorflow as tf
from tensorflow import keras
import numpy as np


class MLP(keras.layers.Layer):
    def __init__(self, units, dropout=0., batch_norm=False, kernel_l2=0., name=None):
        if not units:
            raise ValueError("'units' must not be empty")

        super().__init__(name=name)
        self.hidden_layers = []
        self.dropout_layers = []

        for i, unit in enumerate(units[:-1]):
            name = f'hidden{i}'
            layer = keras.layers.Dense(unit, activation='relu',
                                       kernel_regularizer=keras.regularizers.l2(kernel_l2),
                                       name=name)
            self.hidden_layers.append(name)
            setattr(self, name, layer)

            dropout_name = f'dropout{i}'
            dropout_layer = keras.layers.Dropout(dropout)
            self.dropout_layers.append(dropout_name)
            setattr(self, dropout_name, dropout_layer)

        self.out_layer = keras.layers.Dense(units[-1], activation='relu', name='out_layer')

        if batch_norm:
            self.batch_norm = keras.layers.BatchNormalization()
        else:
            self.batch_norm = None

    def call(self, x, training=False):
        for name, dropout_name in zip(self.hidden_layers, self.dropout_layers):
            layer = getattr(self, name)
            dropout_layer = getattr(self, dropout_name)

            x = layer(x)
            x = dropout_layer(x, training=training)

        x = self.out_layer(x)
        if self.batch_norm:
            return self.batch_norm(x, training=training)
        return x


class NodeAggregator(keras.layers.Layer):
    """
    Pass message from both nodes to the edge in between.
    """

    def __init__(self, edges):
        super().__init__()
        # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
        edge_sources, edge_targets = np.where(edges)
        self.edge_sources = tf.one_hot(edge_sources, len(edges))
        self.edge_targets = tf.one_hot(edge_targets, len(edges))

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
        self.edge_targets = tf.one_hot(edge_targets, len(edges))

    def call(self, edge_msg):
        # edge_msg shape [batch, num_edges, 1, out_units]
        node_msg = tf.transpose(tf.tensordot(edge_msg, self.edge_targets, axes=[[1], [0]]),
                                perm=[0, 3, 1, 2])  # Shape [batch, num_agents, 1, out_units].
        return node_msg


class Conv1D(keras.layers.Layer):
    """
    Condense and abstract the time segments.
    """

    def __init__(self, filters, name=None):
        if not filters:
            raise ValueError("'filters' must not be empty")

        super().__init__(name=name)
        # time segment length before being reduced to 1 by Conv1D
        self.seg_len = 2 * len(filters) + 1

        self.conv1d_layers = []
        for i, channels in enumerate(filters):
            name = f'conv{i}'
            layer = keras.layers.TimeDistributed(
                keras.layers.Conv1D(channels, 3, activation='relu', name=name))
            self.conv1d_layers.append(name)
            setattr(self, name, layer)

        self.channels = channels

    def call(self, time_segs):
        # Reshape to [batch*num_agents, time_seg_len, ndims], since conv1d only accept
        # tensor with 3 dimensions.

        # Node state encoder with 1D convolution along timesteps and across ndims as channels.
        encoded_state = time_segs
        for name in self.conv1d_layers:
            conv = getattr(self, name)
            encoded_state = conv(encoded_state)

        return encoded_state
