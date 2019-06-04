import os
import argparse
import json

import tensorflow as tf
import numpy as np

import gnn
from gnn.data import load_data
from gnn.utils import gumbel_softmax


def model_fn(features, labels, mode, params):
    pred_stack = gnn.dynamical.dynamical_multisteps(features,
                                                    params,
                                                    params['pred_steps'],
                                                    training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {'next_steps': pred_stack}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels, pred_stack)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.train.get_global_step(),
            decay_steps=1000,
            decay_rate=0.99,
            staircase=True,
            name='learning_rate'
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Use the loss between adjacent steps in original time_series as baseline
    time_series_loss_baseline = tf.metrics.mean_squared_error(features['time_series'][:, 1:, :, :],
                                                              features['time_series'][:, :-1, :, :])

    eval_metric_ops = {'time_series_loss_baseline': time_series_loss_baseline}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(features, seg_len, pred_steps, batch_size, mode='train'):
    time_series = features['time_series']
    num_sims, time_steps, num_agents, ndims = time_series.shape
    # Shape [num_sims, time_steps, num_agents, ndims]
    time_series_stack = gnn.utils.stack_time_series(time_series[:, :-pred_steps, :, :],
                                                    seg_len)
    # Shape [num_sims, time_steps-seg_len-pred_steps+1, seg_len, num_agents, ndims]
    expected_time_series_stack = gnn.utils.stack_time_series(time_series[:, seg_len:, :, :],
                                                             pred_steps)
    # Shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, num_agents, ndims]
    assert time_series_stack.shape[:2] == expected_time_series_stack.shape[:2]

    time_segs = time_series_stack.reshape([-1, seg_len, num_agents, ndims])
    expected_time_segs = expected_time_series_stack.reshape([-1, pred_steps, num_agents, ndims])

    processed_features = {'time_series': time_segs}
    if 'edge_type' in features:
        edge_type = features['edge_type']
        nedges, ntypes = edge_type.shape[1:]
        edge_type = np.stack([edge_type for _ in range(time_series_stack.shape[1])], axis=1)
        edge_type = np.reshape(edge_type, [-1, nedges, ntypes])
        processed_features['edge_type'] = edge_type
    labels = expected_time_segs

    if mode == 'train':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            y=labels,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True
        )
    elif mode == 'eval':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            y=labels,
            batch_size=batch_size,
            shuffle=False
        )
    elif mode == 'test':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            batch_size=batch_size,
            shuffle=False
        )


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    model_params['pred_steps'] = ARGS.pred_steps
    seg_len = 2 * len(model_params['cnn']['filters']) + 1

    cnn_multistep_regressor = tf.estimator.Estimator(
        model_fn=model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    if ARGS.train:
        if model_params.get('edge_types', 0) > 1:
            train_data, train_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                               prefix='train')
            if ARGS.data_size:
                train_data, train_edge = train_data[:ARGS.data_size], train_edge[:ARGS.data_size]

            train_edge = gnn.utils.one_hot(train_edge, model_params['edge_types'], np.float32)

            features = {'time_series': train_data, 'edge_type': train_edge}
        else:
            train_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                                   prefix='train')
            if ARGS.data_size:
                train_data = train_data[:ARGS.data_size]

            features = {'time_series': train_data}

        train_input_fn = input_fn(features, seg_len, ARGS.pred_steps, ARGS.batch_size, 'train')

        cnn_multistep_regressor.train(input_fn=train_input_fn,
                                      steps=ARGS.train_steps)

    # Evaluation
    if ARGS.eval:
        if model_params.get('edge_types', 0) > 1:
            valid_data, valid_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                               prefix='valid')
            valid_edge = gnn.utils.one_hot(valid_edge, model_params['edge_types'], np.float32)

            features = {'time_series': valid_data, 'edge_type': valid_edge}
        else:
            valid_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                                   prefix='valid')
            features = {'time_series': valid_data}

        eval_input_fn = input_fn(features, seg_len, ARGS.pred_steps, ARGS.batch_size, 'eval')

        eval_results = cnn_multistep_regressor.evaluate(input_fn=eval_input_fn)

        if not ARGS.verbose:
            print('Evaluation results: {}'.format(eval_results))

    # Prediction
    if ARGS.test:
        if model_params.get('edge_types', 0) > 1:
            test_data, test_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                             prefix='test')
            test_edge = gnn.utils.one_hot(test_edge, model_params['edge_types'], np.float32)

            features = {'time_series': test_data, 'edge_type': test_edge}
        else:
            test_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                                  prefix='test')
            features = {'time_series': test_data}

        predict_input_fn = input_fn(features, seg_len, ARGS.pred_steps, ARGS.batch_size, 'test')

        prediction = cnn_multistep_regressor.predict(input_fn=predict_input_fn)
        prediction = np.array([pred['next_steps'] for pred in prediction])
        np.save(os.path.join(ARGS.log_dir, 'prediction_{}.npy'.format(
            ARGS.pred_steps)), prediction)


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
    parser.add_argument('--train-steps', type=int, default=1000,
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

    if ARGS.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    main()
