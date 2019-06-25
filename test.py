import os
import argparse
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

import gnn
from gnn.data import load_data, preprocess_data
from gnn.utils import one_hot


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
    print(f"\nData from {ARGS.data_dir} loaded.")

    # input_data: a list which is [time_segs, edge_types] if `edge_type` > 1, else [time_segs]
    input_data, expected_time_segs = preprocess_data(
        data, seg_len, ARGS.pred_steps, edge_type=model_params['edge_type'])
    print(f"Data processed.\n")
    nagents, ndims = expected_time_segs.shape[-2:]

    model_params.update({'nagents': nagents, 'ndims': ndims,
                         'pred_steps': ARGS.pred_steps, 'time_seg_len': seg_len})
    model, inputs = gnn.build_model(model_params, return_inputs=True)

    print("Original model summary:")
    model.summary()
    print('\n')

    gnn.load_model(model, ARGS.log_dir)

    # Create Debug model
    outlayer_name = ARGS.layer_name

    outlayer_model = keras.Model(
        inputs=inputs, outputs=model.get_layer(outlayer_name).output)
    print(f"\nOutput up to {outlayer_name}\n")
    outlayer_model.summary()
    layer_output = outlayer_model.predict(input_data)
    np.save(os.path.join(ARGS.log_dir, f'{outlayer_name}_output'), layer_output)
    print(f"Layer {outlayer_name} output saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--data-transpose', type=int, nargs=4, default=None,
                        help='axes for data transposition')
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--pred-steps', type=int, default=1,
                        help='number of steps the estimator predicts for time series')
    parser.add_argument('--layer-name', type=str,
                        help='name of layer whose output is saved.')
    ARGS = parser.parse_args()

    ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    main()
