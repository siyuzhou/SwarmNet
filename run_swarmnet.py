import os
import argparse
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

import gnn
from gnn.data import load_data
from gnn.utils import gumbel_softmax


def data_preprocess(time_series, seg_len, pred_steps, edge_type=None, mode='train'):
    time_steps, nagents, ndims = time_series.shape[1:]
    # time_series shape [num_sims, time_steps, nagents, ndims]
    # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, seg_len, nagents, ndims]
    time_segs_stack = gnn.utils.stack_time_series(time_series[:, :-pred_steps, :, :],
                                                  seg_len)
    # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, nagents, ndims]
    expected_time_segs_stack = gnn.utils.stack_time_series(time_series[:, seg_len:, :, :],
                                                           pred_steps)
    assert (time_segs_stack.shape[1] == expected_time_segs_stack.shape[1]
            == time_steps - seg_len - pred_steps + 1)

    time_segs = time_segs_stack.reshape([-1, seg_len, nagents, ndims])
    expected_time_segs = expected_time_segs_stack.reshape([-1, pred_steps, nagents, ndims])

    return time_segs, expected_time_segs


def build_model(params):
    model = gnn.SwarmNet(params)

    optimizer = keras.optimizers.Adam(lr=params['learning_rate'])
    loss = keras.losses.MeanSquaredError()

    model.compile(optimizer, loss=loss)

    return model


def load_model(model, log_dir):
    checkpoint = os.path.join(log_dir, 'weights.h5')
    if os.path.exists(checkpoint):
        model.load_weights(checkpoint)


def save_model(model, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    checkpoint = os.path.join(log_dir, 'weights.h5')

    model.save_weights(checkpoint)


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    # model_params['pred_steps'] = ARGS.pred_steps
    seg_len = 2 * len(model_params['cnn']['filters']) + 1

    if ARGS.train:
        train_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                               prefix='train')
        nagents, ndims = train_data.shape[-2:]

        time_segs, expected_time_segs = data_preprocess(train_data, seg_len, ARGS.pred_steps)

        model_params.update({'nagents': nagents, 'ndims': ndims,
                             'pred_steps': ARGS.pred_steps, 'seg_len': seg_len})
        model = build_model(model_params)

        model.fit(time_segs[:1], expected_time_segs[:1], epochs=0, batch_size=ARGS.batch_size)
        # print(model.summary())

        load_model(model, ARGS.log_dir)

        model.fit(time_segs, expected_time_segs, epochs=ARGS.epochs, batch_size=ARGS.batch_size)
        save_model(model, ARGS.log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--data-transpose', type=int, nargs=4, default=None,
                        help='axes for data transposition')
    parser.add_argument('--data-size', type=int, default=None,
                        help='optional data size cap to use for training')
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training steps')
    parser.add_argument('--pred-steps', type=int, default=1,
                        help='number of steps the estimator predicts for time series')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='turn on logging info')
    parser.add_argument('--train', action='store_true', default=False,
                        help='turn on training')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='turn on evaluation')
    parser.add_argument('--test', action='store_true', default=False,
                        help='turn on test')
    ARGS = parser.parse_args()

    ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    if ARGS.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    main()
