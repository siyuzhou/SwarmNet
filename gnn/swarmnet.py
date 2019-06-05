import numpy as np
import tensorflow as tf
from tensorflow import keras

from .modules import *
from .utils import fc_matrix


class SwarmNet(keras.Model):
    def __init__(self, params):
        super().__init__()

        # NOTE: For the moment assume Conv1D is always applied
        self.conv1d = Conv1D(params['conv1d']['filters'])
        
    def call(self, x):
        pass


def cnn_dynamical(time_segs, edge_type, params, training=False):
    """Next step prediction using CNN and GNN."""
    # Tensor `time_series` has shape [batch, num_agents, time_seg_len, ndims].
    batch, num_agents, time_seg_len, ndims = time_segs.shape.as_list()
    time_seg_len = 2 * len(params['cnn']['filters']) + 1

    if params['cnn']['filters']:
        # Input Layer
        # Reshape to [batch*num_agents, time_seg_len, ndims], since conv1d only accept
        # tensor with 3 dimensions.
        state = tf.reshape(time_segs, shape=[-1, time_seg_len, ndims])

        # Node state encoder with 1D convolution along timesteps and across ndims as channels.
        encoded_state = state
        for filters in params['cnn']['filters']:
            encoded_state = tf.layers.conv1d(encoded_state, filters, 3, activation=tf.nn.relu)
            # No pooling layer

        # encoded_state shape [batch, num_agents, 1, filters]
        encoded_state = tf.reshape(encoded_state,
                                   shape=[-1, num_agents, 1, filters])
    else:
        encoded_state = time_segs

    # Send encoded state to edges.
    # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
    edge_sources, edge_targets = np.where(fc_matrix(num_agents))
    # One-hot representation of indices of edge sources and targets.
    with tf.name_scope("one_hot"):
        edge_sources = tf.one_hot(edge_sources, num_agents)
        edge_targets = tf.one_hot(edge_targets, num_agents)

    # Form edges. Shape [batch, num_edges, 1, filters]
    edge_msg = node_to_edge(encoded_state, edge_sources, edge_targets)

    # Encode edge messages with MLP. Shape [batch, num_edges, 1, hidden_units]
    if edge_type is not None:
        start = 1 if params.get('skip_zero', False) else 0
        encoded_msg_by_type = []

        for i in range(start, params['edge_types']):
            # mlp_encoder for one edge type.
            encoded_msg = mlp_layers(edge_msg,
                                     params['mlp']['hidden_units'],
                                     params['mlp']['dropout'],
                                     params['mlp']['batch_norm'],
                                     training=training,
                                     name='edge_MLP_encoder_{}'.format(i))

            encoded_msg_by_type.append(encoded_msg)

        encoded_msg_by_type = tf.concat(encoded_msg_by_type, axis=2)
        # Shape [batch, num_edges, edge_types, hidden_units]
        with tf.name_scope('edge_encoding_avg'):
            # Sum of the edge encoding from all possible types.
            edge_msg = tf.reduce_sum(tf.multiply(encoded_msg_by_type,
                                                 edge_type[:, :, start:, :]),
                                     axis=2,
                                     keepdims=True)
            # Shape [batch, num_edges, 1, hidden_units]
    else:
        edge_msg = mlp_layers(edge_msg,
                              params['mlp']['hidden_units'],
                              params['mlp']['dropout'],
                              params['mlp']['batch_norm'],
                              training=training,
                              name='edge_encoding_MLP_1')

    # Compute edge influence to node. Shape [batch, num_agents, 1, hidden_units]
    edge_msg_aggr = edge_to_node(edge_msg, edge_targets)

    # Encode node messages with MLP
    node_msg = mlp_layers(edge_msg_aggr,
                          params['mlp']['hidden_units'],
                          params['mlp']['dropout'],
                          params['mlp']['batch_norm'],
                          training=training,
                          name='node_encoding_MLP_1')

    # The last state in each timeseries of the stack.
    prev_state = time_segs[:, :, -1:, :]

    node_state = tf.concat([prev_state, node_msg], axis=-1)

    # Decode next step. Shape [batch, num_agents, 1, hidden_units]
    node_state = mlp_layers(node_state,
                            params['mlp']['hidden_units'],
                            params['mlp']['dropout'],
                            params['mlp']['batch_norm'],
                            training=training,
                            name='node_decoding_MLP')

    next_state = tf.layers.dense(node_state, ndims, name='linear')

    return next_state


def dynamical_multisteps(features, params, pred_steps, training=False):
    # features shape [batch, time_seg_len, num_agents, ndims]
    time_segs = features['time_series']
    batch, time_seg_len, num_agents, ndims = time_segs.shape.as_list()

    # Transpose to [batch, num_agents, time_seg_len, ndims]
    time_segs = tf.transpose(time_segs, [0, 2, 1, 3])

    with tf.variable_scope('prediction_one_step') as scope:
        pass

    if params.get('edge_types', 0) > 1:
        with tf.name_scope('edge_type'):
            edge_type = features['edge_type']
            # Shape [batch, num_edges, num_edge_types]
            # Expand edge_type so that it has same number of dimensions as time_series.
            edge_type = tf.expand_dims(edge_type, 3)
            # edge_type shape [batch, num_edges, num_edge_types, 1]
    else:
        edge_type = None

    def one_step(i, time_segs):
        with tf.name_scope(scope.original_name_scope):
            prev_step = time_segs[:, :, -1:, :]
            next_state = prev_step + cnn_dynamical(
                time_segs[:, :, i:, :], edge_type, params, training=training)

            return i+1, tf.concat([time_segs, next_state], axis=2)

    i = 0
    _, time_series_stack = tf.while_loop(
        lambda i, _: i < pred_steps,
        one_step,
        [i, time_segs],
        shape_invariants=[tf.TensorShape(None),
                          tf.TensorShape([batch, num_agents, None, ndims])]
    )

    # Transpose to [batch, time_seg_len+pred_steps, num_agents, ndims]
    time_series_stack = tf.transpose(time_series_stack, [0, 2, 1, 3])

    return time_series_stack[:, time_seg_len:, :, :]
