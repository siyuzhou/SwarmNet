import os
import argparse
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

import gnn
from gnn.data import load_data
from gnn.utils import one_hot


def data_preprocess(data, seg_len, pred_steps, edge_type=None, mode='train'):
    if edge_type > 1:
        time_series, edge_types = data
    else:
        time_series = data

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

    if edge_type > 1:
        edge_types = one_hot(edge_types, edge_type, np.float32)
        # Shape [instances, n_edges, edge_type]
        n_edges = edge_types.shape[1]
        edge_types = np.stack([edge_types for _ in range(time_segs_stack.shape[1])], axis=1)
        edge_types = np.reshape(edge_types, [-1, n_edges, edge_type])

        return [time_segs, edge_types], expected_time_segs

    else:
        return time_segs, expected_time_segs


def build_model(params):
    model = gnn.SwarmNet(params)

    optimizer = keras.optimizers.Adam(lr=params['learning_rate'])
    loss = keras.losses.MeanSquaredError()

    model.compile(optimizer, loss=loss)

    if params['edge_type'] > 1:
        input_shape = [(None, params['time_seg_len'], params['nagents'], params['ndims']),
                       (None, params['nagents']*(params['nagents']-1), params['edge_type'])]
    else:
        input_shape = (None, params['time_seg_len'], params['nagents'], params['ndims'])

    model.build(input_shape)

    return model


def load_model(model, log_dir):
    checkpoint = os.path.join(log_dir, 'weights.h5')
    if os.path.exists(checkpoint):
        print('Model loaded.')
        model.load_weights(checkpoint)


def save_model(model, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    checkpoint = os.path.join(log_dir, 'weights.h5')

    model.save_weights(checkpoint)
    print('Model saved.')


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    # model_params['pred_steps'] = ARGS.pred_steps
    seg_len = 2 * len(model_params['cnn']['filters']) + 1

    if ARGS.train:
        prefix = 'train'
    elif ARGS.eval:
        prefix = 'valid'
    elif ARGS.test:
        prefix = 'test'

    model_params['edge_type'] = model_params.get('edge_type', 1)
    # data contains edge_types if `edge=True`.
    data = load_data(ARGS.data_dir, ARGS.data_transpose,
                     edge=model_params['edge_type'] > 1, prefix=prefix)

    # input_data: a list which is [time_segs, edge_types] if `edge_type` > 1, else [time_segs]
    input_data, expected_time_segs = data_preprocess(
        data, seg_len, ARGS.pred_steps, edge_type=model_params['edge_type'])

    nagents, ndims = expected_time_segs.shape[-2:]

    model_params.update({'nagents': nagents, 'ndims': ndims,
                         'pred_steps': ARGS.pred_steps, 'time_seg_len': seg_len})
    model = build_model(model_params)

    load_model(model, ARGS.log_dir)

    if ARGS.train:
        history = model.fit(input_data, expected_time_segs,
                            epochs=ARGS.epochs, batch_size=ARGS.batch_size)
        save_model(model, ARGS.log_dir)
        print(history.history)

    elif ARGS.eval:
        result = model.evaluate(input_data, expected_time_segs, batch_size=ARGS.batch_size)
        print(result)

    elif ARGS.test:
        prediction = model.predict(input_data)
        np.save(os.path.join(ARGS.log_dir, f'prediction_{ARGS.pred_steps}.npy'), prediction)


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
