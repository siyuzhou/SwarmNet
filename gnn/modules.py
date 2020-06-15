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
        # Node state encoder with 1D convolution along timesteps and across ndims as channels.
        encoded_state = time_segs
        for name in self.conv1d_layers:
            conv = getattr(self, name)
            encoded_state = conv(encoded_state)

        return encoded_state


class NodePropagator(keras.layers.Layer):
    """
    Pass message between every pair of nodes.
    """

    # def __init__(self):
    #     super().__init__()

    # Construct full connection matrix, mark source node and target node for each connection.
    # `self._edge_sources` and `self._edge_targets` with size [num_edges, num_nodes]
    # edge_sources, edge_targets = np.where(np.ones((graph_size, graph_size)))
    # self._edge_sources = tf.one_hot(edge_sources, len(edges))
    # self._edge_targets = tf.one_hot(edge_targets, len(edges))

    def call(self, node_states):
        # node_states shape [batch, num_nodes, out_units].
        num_nodes = node_states.shape[1]

        msg_from_source = tf.repeat(tf.expand_dims(node_states, 2), num_nodes, axis=2)
        msg_from_target = tf.repeat(tf.expand_dims(node_states, 1), num_nodes, axis=1)
        # msg_from_source = tf.transpose(tf.tensordot(node_states, self._edge_sources, axes=[[1], [1]]),
        #                                perm=[0, 2, 1])
        # msg_from_target = tf.transpose(tf.tensordot(node_states, self._edge_targets, axes=[[1], [1]]),
        #                                perm=[0, 2, 1])
        # msg_from_source and msg_from_target in shape [batch, num_nodes, num_nodes, out_units]
        node_msgs = tf.concat([msg_from_source, msg_from_target], axis=-1)

        return node_msgs


class EdgeSumAggregator(keras.layers.Layer):
    """
    Sum up messages from incoming edges to the node.
    """

    # def __init__(self):
    #     super().__init__()

    # `edge_sources` and `edge_targets` in shape [num_edges, num_nodes].
    # edge_targets = np.where(np.ones((graph_size, graph_size)))[1]
    # self._edge_targets = tf.one_hot(edge_targets, len(edges))

    def call(self, edge_msgs, node_states, edges):
        # edge_msg shape [batch, num_nodes, num_nodes, edge_type, out_units]

        # Add messsages of all edge types. Shape becomes [batch, num_nodes, out_units]
        edge_msg_sum = tf.reduce_sum(edge_msgs, axis=[1, 3])

        # Sum edge msgs in each neighborhood.
        # edge_msg_sum = tf.transpose(tf.tensordot(edge_msg_sum, self._edge_targets, axes=[[1], [0]]),
        #                             perm=[0, 2, 1])  # Shape [batch, num_nodes, out_units].

        return edge_msg_sum


class EdgeEncoder(keras.layers.Layer):
    """
    Propagate messages to edge from the two nodes connected via edge encoders.
    """

    def __init__(self, edge_type, encoder_params):
        super().__init__()

        self.edge_type = edge_type

        self.edge_encoders = [MLP(encoder_params['hidden_units'],
                                  encoder_params['dropout'],
                                  encoder_params['batch_norm'],
                                  encoder_params['kernel_l2'],
                                  name=f'edge_encoder_{i}')
                              for i in range(1, self.edge_type+1)]

    def call(self, node_msgs, edges, training=False):
        # `node_msgs` shape [batch, num_nodes*num_nodes, units]
        # `edges` shape [batch, num_nodes, num_nodes, num_edge_label]
        edge_types = tf.expand_dims(edges, axis=-1)
        # Shape [batch, num_nodes, num_nodes, num_edge_label, 1]
        # edge_types = tf.reshape(edges, [-1, num_nodes*num_nodes, num_edge_label, 1])

        encoded_msgs_by_type = []
        for i in range(self.edge_type):
            # mlp_encoder for each edge type.
            encoded_msgs = self.edge_encoders[i](node_msgs, training=training)

            encoded_msgs_by_type.append(encoded_msgs)

        encoded_msgs_by_type = tf.stack(encoded_msgs_by_type, axis=3)
        # Shape [batch, num_nodes, num_nodes, edge_types, units]

        # Only encoded message of the type same as the edge type gets retaind.
        # Force skip 0 type, 0 means no connection, no message.
        edge_msgs = tf.multiply(encoded_msgs_by_type, edge_types[:, :, :, 1:, :])

        return edge_msgs


class GraphConv(keras.layers.Layer):
    def __init__(self, graph_size, edge_type, params, name=None):
        super().__init__(name=name)

        self.node_prop = NodePropagator()

        self.edge_aggr = EdgeSumAggregator()

        self.edge_encoder = EdgeEncoder(edge_type, params['edge_encoder'])

        self.node_decoder = MLP(params['node_decoder']['hidden_units'],
                                params['node_decoder']['dropout'],
                                params['node_decoder']['batch_norm'],
                                params['node_decoder']['kernel_l2'],
                                name='node_decoder')

    def call(self, node_states, edges, training=False):
        # Propagate node states.
        node_msgs = self.node_prop(node_states)

        # Form edges. Shape [batch, num_edges, edge_type, units]
        edge_msgs = self.edge_encoder(node_msgs, edges, training)

        # Edge aggregation. Shape [batch, num_nodes, units]
        edge_msgs_aggr = self.edge_aggr(edge_msgs, node_states, edges)

        # Update node_states
        node_states = self.node_decoder(
            tf.concat([node_states, edge_msgs_aggr], axis=-1), training=training)

        return node_states
