import os
import argparse
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

import gnn
from gnn.data import load_data, preprocess_data
from gnn.utils import one_hot


def define_model():
    pass


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    # model_params['pred_steps'] = ARGS.pred_steps
    seg_len = 2 * len(model_params['cnn']['filters']) + 1

    prefix = 'test'

    model_params['edge_type'] = model_params.get('edge_type', 1)
    # data contains edge_types if `edge=True`.
    data = load_data(ARGS.data_dir, ARGS.data_transpose,
                     edge=model_params['edge_type'] > 1, prefix=prefix)

    # input_data: a list which is [time_segs, edge_types] if `edge_type` > 1, else [time_segs]
    input_data, expected_time_segs = preprocess_data(
        data, seg_len, ARGS.pred_steps, edge_type=model_params['edge_type'])

    nagents, ndims = expected_time_segs.shape[-2:]

    model_params.update({'nagents': nagents, 'ndims': ndims,
                         'pred_steps': ARGS.pred_steps, 'time_seg_len': seg_len})
    model = gnn.build_model(model_params)
    print(model.summary())
    gnn.load_model(model, ARGS.log_dir)

    # Create Debug model
    edge_encoder_name = 'edge_encoder_1'
    print(model.get_layer(edge_encoder_name).input)

    edge_encoder_model = keras.Model(
        inputs=model.inputs, output=model.get_layer(edge_encoder_name).output)
    edge_encoder_output = edge_encoder_model.predict(input_data)
    np.save(os.path.join(ARGS.log_dir, f'edge_encoder_output'), edge_encoder_output)


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
    parser.add_argument('--pred-steps', type=int, default=1,
                        help='number of steps the estimator predicts for time series')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    ARGS = parser.parse_args()

    ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    main()
