import tensorflow as tf
import numpy as np


def mlp_layers(inputs, layers, dropout=0., batch_norm=False, training=False, name=None):
    with tf.variable_scope(name, default_name="MLP", reuse=tf.AUTO_REUSE):
        h = inputs
        for units in layers[:-1]:
            h = tf.layers.dense(h, units, activation=tf.nn.relu)
            h = tf.layers.dropout(h, dropout, training=training)

        output = tf.layers.dense(h, layers[-1], activation=tf.nn.relu)
        # shape [num_sims, time_steps, num_edges, out_units]

        if batch_norm:
            output = tf.layers.batch_normalization(output, training=training)

    return output
