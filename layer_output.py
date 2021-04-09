import os
import argparse
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

import swarmnet
from swarmnet.data import load_data, preprocess_data


def main():
    prefix = 'test'

    model_params = swarmnet.utils.load_model_params(ARGS.config)

    # data contains edge_types if `edge=True`.
    data = load_data(ARGS.data_dir, prefix=prefix)
    print(f"\nData from {ARGS.data_dir} loaded.")

    # input_data: a list which is [time_segs, edge_types] if `edge_type` > 1, else [time_segs]
    input_data, expected_time_segs = preprocess_data(
        data, model_params['time_seg_len'], ARGS.pred_steps, edge_type=model_params['edge_type'], ground_truth=False)
    print(f"Data processed.\n")
    nagents, ndims = data[0].shape[-2:]

    model, inputs = swarmnet.SwarmNet.build_model(
        nagents, ndims, model_params, pred_steps=ARGS.pred_steps, return_inputs=True)

    print("Original model summary:")
    model.summary()
    print('\n')

    swarmnet.utils.load_model(model, ARGS.log_dir)

    # Create Debug model
    outlayer_name = ARGS.layer_name
    layers = {'edge_encoder': model.graph_conv.edge_encoder,
              'edge_aggr': model.graph_conv.edge_aggr,
              'node_decoder': model.graph_conv.node_decoder}

    outlayer_model = keras.Model(
        inputs=inputs, outputs=layers[outlayer_name].output)

    print(f"\nOutput up to {outlayer_name}\n")
    outlayer_model.summary()

    layer_output = outlayer_model.predict(input_data)
    np.save(os.path.join(ARGS.log_dir,
            f'{outlayer_name}_output'), layer_output)
    print(f"Layer {outlayer_name} output saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
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
