import os
import argparse
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

import gnn
from gnn.data import load_data, preprocess_data
from gnn.utils import one_hot


def eval_base_line(eval_data):
    time_segs = eval_data[0]
    return np.mean(np.square(time_segs[:, :-1, :, :] -
                             time_segs[:, 1:, :, :]))


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
    print(f"\nData from {ARGS.data_dir} loaded.")
    # input_data: a list which is [time_segs, edge_types] if `edge_type` > 1, else [time_segs]
    input_data, expected_time_segs = preprocess_data(
        data, seg_len, ARGS.pred_steps, edge_type=model_params['edge_type'])
    print(f"\nData processed.\n")

    nagents, ndims = expected_time_segs.shape[-2:]

    model_params.update({'nagents': nagents, 'ndims': ndims,
                         'pred_steps': ARGS.pred_steps, 'time_seg_len': seg_len})
    model = gnn.build_model(model_params)
    model.summary()

    gnn.load_model(model, ARGS.log_dir)

    if ARGS.train:
        checkpoint = gnn.save_model(model, ARGS.log_dir)
        history = model.fit(input_data, expected_time_segs,
                            epochs=ARGS.epochs, batch_size=ARGS.batch_size,
                            callbacks=[checkpoint])
        # print(history.history)

    elif ARGS.eval:
        result = model.evaluate(input_data, expected_time_segs, batch_size=ARGS.batch_size)
        # print(result)
        print('Baseline: ', eval_base_line(input_data))

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

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    main()
