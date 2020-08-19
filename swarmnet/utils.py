import os
import json
import tensorflow as tf
import numpy as np


def off_diag_matrix(n):
    return np.ones((n, n)) - np.eye(n)


def one_hot(labels, num_classes, dtype=np.int):
    identity = np.eye(num_classes, dtype=dtype)
    one_hots = identity[labels.reshape(-1)]
    return one_hots.reshape(labels.shape + (num_classes,))


def stack_time_series(time_series, seg_len, axis=2):
    # time_series shape [num_sims, time_steps, num_agents, ndims]
    time_steps = time_series.shape[1]
    return np.stack([time_series[:, i:time_steps+1-seg_len+i, :, :] for i in range(seg_len)],
                    axis=axis)


def load_model(model, log_dir):
    checkpoint = os.path.join(log_dir, 'weights.h5')
    if os.path.exists(checkpoint):
        model.load_weights(checkpoint)


def save_model(model, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    checkpoint = os.path.join(log_dir, 'weights.h5')

    model.save_weights(checkpoint)

    return tf.keras.callbacks.ModelCheckpoint(checkpoint, save_weights_only=True)


def load_model_params(config):
    with open(config) as f:
        model_params = json.load(f)

    seg_len = 2 * len(model_params['cnn']['filters']) + 1
    model_params['time_seg_len'] = seg_len
    model_params['edge_type'] = model_params.get('edge_type', 1)
    return model_params
