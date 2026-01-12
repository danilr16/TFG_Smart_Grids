import numpy as np
import keras_tuner as kt
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesRandomSearch(kt.RandomSearch):
    def run_trial(self, trial, x, y, n_splits=5, batch_size=32, epochs=10, callbacks=None, **kwargs):
        time_series = TimeSeriesSplit(n_splits=n_splits)

        metrics_fold = []
        for fold, (train_index, test_index) in enumerate(time_series.split(x)):
            model = self.hypermodel.build(trial.hyperparameters)
            if hasattr(x, 'iloc'):
                x_train_fold, x_test_fold = x.iloc[train_index], x.iloc[test_index]
            else:
                x_train_fold, x_test_fold = x[train_index], x[test_index]

            y_train_fold = {}
            y_test_fold = {}
            for key, data in y.items():
                if hasattr(data, 'iloc'):
                    y_train_data = data.iloc[train_index]
                    y_test_data = data.iloc[test_index]
                else:
                    y_train_data = data[train_index]
                    y_test_data = data[test_index]

                normalizer_y = MinMaxScaler()
                y_train_values = y_train_data.values if hasattr(y_train_data, 'values') else y_train_data
                y_test_values = y_test_data.values if hasattr(y_test_data, 'values') else y_test_data

                y_train_fold[key] = normalizer_y.fit_transform(y_train_values)
                y_test_fold[key] = normalizer_y.transform(y_test_values)

            model.fit(x_train_fold, y_train_fold, validation_data=(x_test_fold, y_test_fold),
                      batch_size=batch_size, epochs=epochs, verbose=0, callbacks=callbacks)
            metrics = model.evaluate(x_test_fold, y_test_fold, verbose=0, return_dict=True)
            metrics_fold.append(metrics)

            print(f'Trial {trial.trial_id} - Fold {fold + 1}:')
            for metric, value in metrics.items():
                print(f'\t {metric}: {value:.5f}')

        avg_metrics = {}
        for metric_name in metrics_fold[0].keys():
            values = [metrics[metric_name] for metrics in metrics_fold]
            key_name = f'val_{metric_name}'
            avg_metrics[key_name] = np.mean(values)
        print(f"--- Promedio Trial {trial.trial_id}: val_loss = {avg_metrics['val_loss']:.5f} ---\n")
        self.oracle.update_trial(trial.trial_id, avg_metrics)


def make_build_model(input_dim, num_voltages, norm_layer):
    def build_model(hp):
        inputs = tf.keras.Input(shape=(input_dim,), name='input_x')
        x_norm = norm_layer(inputs)

        activation = hp.Choice('activation', ['relu', 'elu'])
        dropout_rate = hp.Float('dropout', 0.0, 0.3, step=0.1)
        use_batch_norm = hp.Boolean('use_batch_norm')
        learning_rate = hp.Float('lr', 1e-4, 1e-2, sampling='log')
        optimizer = hp.Choice('optimizer', values=['sgd', 'adam'])
        if optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        n_hidden_q = hp.Int('n_hidden_q', 1, 3)
        n_neurons_q = hp.Int('n_neurons_q', 32, 128, step=32)

        x1 = x_norm
        for _ in range(n_hidden_q):
            x1 = tf.keras.layers.Dense(n_neurons_q)(x1)
            if use_batch_norm:
                x1 = tf.keras.layers.BatchNormalization()(x1)
            x1 = tf.keras.layers.Activation(activation)(x1)
            if dropout_rate > 0:
                x1 = tf.keras.layers.Dropout(dropout_rate)(x1)

        out_qgen = tf.keras.layers.Dense(1, activation='linear', name='out_qgen7')(x1)
        merged = tf.keras.layers.Concatenate()([x_norm, out_qgen])

        n_hidden_v = hp.Int('n_hidden_v', 1, 2)
        n_neurons_v = hp.Int('n_neurons_v', 64, 256, step=32)

        x2 = merged
        for _ in range(n_hidden_v):
            x2 = tf.keras.layers.Dense(n_neurons_v)(x2)
            if use_batch_norm:
                x2 = tf.keras.layers.BatchNormalization()(x2)
            x2 = tf.keras.layers.Activation(activation)(x2)
            if dropout_rate > 0:
                x2 = tf.keras.layers.Dropout(dropout_rate)(x2)

        out_v = tf.keras.layers.Dense(num_voltages, activation='linear', name='out_volt')(x2)

        n_hidden_p = hp.Int('n_hidden_p', 2, 5)
        n_neurons_p = hp.Int('n_neurons_p', 64, 256, step=64)

        x3 = merged
        for _ in range(n_hidden_p):
            x3 = tf.keras.layers.Dense(n_neurons_p)(x3)
            if use_batch_norm:
                x3 = tf.keras.layers.BatchNormalization()(x3)
            x3 = tf.keras.layers.Activation(activation)(x3)
            if dropout_rate > 0:
                x3 = tf.keras.layers.Dropout(dropout_rate)(x3)

        out_perd = tf.keras.layers.Dense(1, activation='linear', name='out_perd')(x3)

        model = tf.keras.Model(inputs=inputs, outputs=[out_v, out_perd, out_qgen])

        model.compile(
            optimizer=optimizer,
            loss_weights={'out_qgen7': 1.0, 'out_volt': 1.0, 'out_perd': 2.0},
            loss={'out_qgen7': 'mse', 'out_volt': 'mse', 'out_perd': 'mse'},
            metrics={'out_qgen7': ['mae', 'mape', tf.keras.metrics.R2Score(name='r2_score')],
                     'out_volt': ['mae', 'mape', tf.keras.metrics.R2Score(name='r2_score')],
                     'out_perd': ['mae', 'mape', tf.keras.metrics.R2Score(name='r2_score')]},
        )
        return model
    return build_model