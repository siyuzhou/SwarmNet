import tensorflow as tf
import numpy as np


def fc_matrix(n):
    return np.ones((n, n)) - np.eye(n)


def one_hot(labels, num_classes, dtype=np.int):
    identity = np.eye(num_classes, dtype=dtype)
    one_hots = identity[labels.reshape(-1)]
    return one_hots.reshape(labels.shape + (num_classes,))


def sample_gumbel(shape, eps=1e-20):
    """
    Borrowed from
    https://github.com/vithursant/VAE-Gumbel-Softmax/blob/master/vae_gumbel_softmax.py
    """
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    """
    Borrowed from
    https://github.com/vithursant/VAE-Gumbel-Softmax/blob/master/vae_gumbel_softmax.py
    """
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


def stack_time_series(time_series, seg_len, axis=2):
    # time_series shape [num_sims, time_steps, num_agents, ndims]
    time_steps = time_series.shape[1]
    return np.stack([time_series[:, i:time_steps+1-seg_len+i, :, :] for i in range(seg_len)],
                    axis=axis)
