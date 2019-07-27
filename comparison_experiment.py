#! /usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sum_of_gaussians as sog
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import sys
import os.path
import functools as ft


BASE_FILENAME = '.'
NUM_EPOCHS = 1000

MIN_DATA_VALUE = -8
MAX_DATA_VALUE = 8

NUM_DIMENSIONS = 2

NUM_TRAIN_EXAMPLES = 10000
NUM_TEST_EXAMPLES = 256

NUM_DATASET_GENERATIONS_PER_TRUE_NUM_GAUSSIANS = 8
NUM_RANDOM_INITIALIZATIONS = 4
NUM_LEARNED_GAUSSIANS = [2, 4, 8, 12, 16, 24, 32]


def generate_random_parameters(min_data_value, max_data_value, num_gaussians, num_dimensions):
    mus = np.random.uniform(
        low=min_data_value, high=max_data_value,
        size=(num_gaussians, num_dimensions)).astype(np.float32)
    _sigmas_tmp = np.triu(np.random.normal(
        loc=2.0, scale=1.0,
        size=(num_gaussians, num_dimensions, num_dimensions)).astype(np.float32))
    sigmas = _sigmas_tmp @ np.transpose(_sigmas_tmp, (0, 2, 1))
    amps = np.random.uniform(
        low=0., high=5.,
        size=(num_gaussians,)).astype(np.float32)
    return mus, sigmas, amps


def evaluate_by_params(x, mus, sigmas, amps):
    return sum(sog.calculate_multivariate_gaussian(x, amp, mu, sigma)
               for mu, sigma, amp in zip(mus, sigmas, amps))


def build_dataset(X_train, y_train, eval_mode=False):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    if not eval_mode:
        dataset = dataset.shuffle(NUM_TRAIN_EXAMPLES)  # shuffle all in train.

    dataset = dataset.batch(2048)

    return dataset


def get_loss_distribution_for_multiple_runs(num_iterations, num_epochs, train_dataset, test_dataset,
                                            **build_model_kwargs):
    all_results = []
    for _ in range(num_iterations):
        try:
            model = build_model(**build_model_kwargs)
            model.fit(train_dataset, validation_data=test_dataset, epochs=num_epochs, verbose=0)
        except tf.errors.InvalidArgumentError as e:
            # log and try once again
            print(f'!!! Failed on {e}')
            model = build_model(**build_model_kwargs)
            model.fit(train_dataset, validation_data=test_dataset, epochs=num_epochs, verbose=0)
        result = (model.history.history['mse'], model.history.history['val_mse'])
        all_results.append(result)

    return pd.Series({
        'mean': np.mean([test_loss[-1] for _, test_loss in all_results]),
        'std': np.std([test_loss[-1] for _, test_loss in all_results]),
        'min': np.min([test_loss[-1] for _, test_loss in all_results])})


class Rename(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        self.layer = tf.keras.layers.Lambda(lambda x: x)
        super(Rename, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        return self.layer(inputs)


def build_model(num_learned_gaussians, amps_l1_reg=1e-3, use_means_spread_regularizer=True):
    inp = tf.keras.Input(shape=(NUM_DIMENSIONS,), dtype=tf.float32)

    sog_layer = sog.SumOfGaussians(
        name='sog',
        num_gaussians=num_learned_gaussians,
        amps_l1_reg=amps_l1_reg,
        use_means_spread_regularizer=use_means_spread_regularizer,
        centers_min=[MIN_DATA_VALUE] * NUM_DIMENSIONS, centers_max=[MAX_DATA_VALUE] * NUM_DIMENSIONS,
    )
    out = sog_layer(inp)
    out = Rename('output')(out)

    model = tf.keras.Model(
        inputs=inp,
        outputs=out)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer,
                  loss={'output': 'mse'},
                  metrics={'output': 'mse'})  # this allows returning only the mse without other losses
    return model


def run_for_num_true_gaussians(true_num_gaussians):

    for dataset_gen_index in range(NUM_DATASET_GENERATIONS_PER_TRUE_NUM_GAUSSIANS):
        print(f'Generating data for {true_num_gaussians} gaussians [#{dataset_gen_index}]...')
        results_per_true_num_gaussians = []
        mus, sigmas, amps = generate_random_parameters(
            MIN_DATA_VALUE, MAX_DATA_VALUE, true_num_gaussians, NUM_DIMENSIONS)
        ground_truth_model = ft.partial(evaluate_by_params, mus=mus, sigmas=sigmas, amps=amps)

        _X_train = np.random.uniform(low=MIN_DATA_VALUE, high=MAX_DATA_VALUE,
                                     size=(NUM_TRAIN_EXAMPLES, NUM_DIMENSIONS)).astype(np.float32)
        _y_train = ground_truth_model(_X_train)
        _X_test = np.random.uniform(low=MIN_DATA_VALUE, high=MAX_DATA_VALUE,
                                    size=(NUM_TEST_EXAMPLES, NUM_DIMENSIONS)).astype(np.float32)
        _y_test = ground_truth_model(_X_test)

        results_no_reg = []
        results_amps_reg = []
        results_centers_reg = []
        results_both_regs = []

        for num_learned_gaussians in NUM_LEARNED_GAUSSIANS:
            tf.keras.backend.clear_session()

            train_dataset = build_dataset(_X_train, _y_train)
            test_dataset = build_dataset(_X_test, _y_test, eval_mode=True)

            print(f'  - Fitting {NUM_RANDOM_INITIALIZATIONS} times (randomly initialized) with {num_learned_gaussians}'
                  ' learned gaussians...')
            res = get_loss_distribution_for_multiple_runs(
                NUM_RANDOM_INITIALIZATIONS, NUM_EPOCHS, train_dataset, test_dataset,
                num_learned_gaussians=num_learned_gaussians,
                amps_l1_reg=0.0, use_means_spread_regularizer=False)
            res['k'] = num_learned_gaussians
            results_no_reg.append(res)

            res = get_loss_distribution_for_multiple_runs(
                NUM_RANDOM_INITIALIZATIONS, NUM_EPOCHS, train_dataset, test_dataset,
                num_learned_gaussians=num_learned_gaussians,
                amps_l1_reg=0.1, use_means_spread_regularizer=False)
            res['k'] = num_learned_gaussians
            results_amps_reg.append(res)

            res = get_loss_distribution_for_multiple_runs(
                NUM_RANDOM_INITIALIZATIONS, NUM_EPOCHS, train_dataset, test_dataset,
                num_learned_gaussians=num_learned_gaussians,
                amps_l1_reg=0.0, use_means_spread_regularizer=True)
            res['k'] = num_learned_gaussians
            results_centers_reg.append(res)

            res = get_loss_distribution_for_multiple_runs(
                NUM_RANDOM_INITIALIZATIONS, NUM_EPOCHS, train_dataset, test_dataset,
                num_learned_gaussians=num_learned_gaussians,
                amps_l1_reg=0.1, use_means_spread_regularizer=True)
            res['k'] = num_learned_gaussians
            results_both_regs.append(res)

        results_no_reg = pd.DataFrame(results_no_reg)
        results_amps_reg = pd.DataFrame(results_amps_reg)
        results_centers_reg = pd.DataFrame(results_centers_reg)
        results_both_regs = pd.DataFrame(results_both_regs)

        curr_results = (results_no_reg, results_amps_reg, results_centers_reg, results_both_regs)
        results_per_true_num_gaussians.append(curr_results)

        filename = os.path.join(BASE_FILENAME, f'results_per_true_num_gaussians.{true_num_gaussians:02d}.{dataset_gen_index:02d}.pickle')
        pickle.dump(curr_results, open(filename, 'wb'))
        print(f'  - {filename} stored.')


def main(args):
    true_num_gaussians = int(args[1])
    run_for_num_true_gaussians(true_num_gaussians)


if __name__ == '__main__':
    main(sys.argv)
