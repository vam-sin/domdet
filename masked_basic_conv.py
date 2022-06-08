import os
import pickle
import sys
import re
import pandas as pd
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, recall_score, precision_score
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
import wandb
from wandb.keras import WandbCallback

from no_mask_transformer import METRICS, weighted_cross_entropy
from data_generator import DataGenerator

# hp = {'batch_size': 3,
#       'k_size': 5,
#       'filters': 32,
#       'dense_layers': 1,
#       'learning_rate': 0.00001,
#       'max_res': 300,
#       'n_features': 4080,
#       'conv_layers': 3,
#       'epochs': 100,
#       }


def get_hyperparameters(path='conv_hyperparameters.csv'):
    # load hyper params from csv select one set of HPs to evaluate
    # Designed for array job grid-search over HPs
    row_index = int(sys.argv[1]) -1
    hyper_df = pd.read_csv(path)
    hp = hyper_df.loc[row_index].to_dict()
    hp = {k: v for k, v in hp.items() if not re.search('[A-Za-z]*_e\d+', k)} # remove the rows of the data frame that contain performance metrics from previous runs
    print('loaded csv hyperparams')
    return hp, row_index

def select_f1_threshold(precision, recall, thresholds):
    f1_scores = 2 * recall * precision / (recall + precision)
    return thresholds[np.argmax(f1_scores)]

class EvaluateOnTestCallback(keras.callbacks.Callback):
    def __init__(self, generator, *args, **kwargs):
        super(EvaluateOnTestCallback, self).__init__(*args, **kwargs)
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        scores = self.model.evaluate_on_test(self.generator)
        for k, v in scores.items():
            wandb.log({f'test combined {k}': v})


class MaskedConvNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(MaskedConvNet, self).__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.precision_tracker = tf.keras.metrics.Mean(name="prec")
        self.recall_tracker = tf.keras.metrics.Mean(name="rec")

    def train_step(self, data):
        (x, mask), y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.masked_weighted_cross_entropy(y, y_pred, mask)
            self.loss_tracker.update_state(loss)
        self.update_scores(y, y_pred, mask)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def masked_weighted_cross_entropy(self, y_true, y_pred, mask):
        weighting = 0.2
        loss_pos = weighting * y_true * tf.math.log(y_pred)
        loss_neg = (1 - y_true) * tf.math.log(1 - y_pred)
        loss = -1 * (loss_pos + loss_neg) * mask
        return tf.reduce_sum(loss)

    def update_scores(self, y_true, y_pred, mask):
        self.precision_tracker.update_state(self.precision(y_true, y_pred, mask))
        self.recall_tracker.update_state(self.recall(y_true, y_pred))
        # self.recall_tracker = recall_score(f_y, f_p)

    def precision(self, y_true, y_pred, mask):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred * mask, 0, 1)))
        precision_keras = true_positives / (predicted_positives + K.epsilon())
        return precision_keras

    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_keras = true_positives / (possible_positives + K.epsilon())
        return recall_keras

    def evaluate_on_test(self, test_generator):
        """
        Average of multiple batches of AUC ROC score are not the same as
        as the AUC ROC score of all the data computed at once
        but computing on the entire test set at once is expensive
        this function breaks the test set into batches of batches and takes the average of those
        """
        batches_of_batches = 10
        normalizing_weight = 0
        running_auc_roc = 0
        running_recall = 0
        running_precision = 0
        running_prauc = 0
        running_pred = np.array([])
        running_y = np.array([])
        for step, ((x, mask), y) in enumerate(test_generator):
            if (step % batches_of_batches == 0 and step != 0) or step ==len(test_generator)-1:
                weight = len(running_y)
                print(' ')
                print(weight, 'residues')
                normalizing_weight += weight
                roc_auc = roc_auc_score(running_y, running_pred)
                wandb.log({"test mini roc auc": roc_auc})
                running_auc_roc += roc_auc * weight
                precision, recall, thresholds = precision_recall_curve(running_y, running_pred)
                classification_threshold = select_f1_threshold(precision, recall, thresholds)
                binarized_pred = running_pred >= classification_threshold
                prauc = auc(recall, precision)
                wandb.log({"test mini prauc": prauc})
                running_prauc += prauc * weight
                rec = recall_score(running_y, binarized_pred)
                wandb.log({"test mini recall": rec})
                running_recall += rec * weight
                prec = precision_score(running_y, binarized_pred)
                wandb.log({"test mini precision": prec})
                running_precision += prec * weight
                print('batch auc roc score', round(running_auc_roc / normalizing_weight, 4))
                print(' ')
                running_y = np.array([])
                running_pred = np.array([])

            y_pred = self.predict(x)
            y_pred = y_pred[mask.astype(bool)]
            y = y[mask.astype(bool)]
            running_pred = np.append(running_pred, y_pred)
            running_y = np.append(running_y, y)

        scores = {
            'roc_auc': running_auc_roc / normalizing_weight,
            'pr_auc': running_prauc / normalizing_weight,
            'recall': running_recall / normalizing_weight,
            'precision': running_precision / normalizing_weight

        }
        return scores


if __name__=="__main__":
    wandb.init(entity="cath", project="domdet")
    train_dir = 'features/processed/train-val/'
    test_dir = 'features/processed/test/'
    model_save_dir = 'conv_logs/'
    hp, row_index = get_hyperparameters(path='conv_hyperparameters.csv')

    for k in ['max_res', 'epochs', 'filters', 'n_features', 'k_size', 'conv_layers', 'dense_layers', 'batch_size']:
        hp[k] = int(hp[k])

    wandb.config.update(hp)
    wandb.config.model = 'basic convnet'
    input = keras.layers.Input(shape=(hp['max_res'], hp['n_features']))
    input_mask = keras.layers.Input(shape=(hp['max_res'], hp['n_features']))
    conv1 = keras.layers.Conv1D(hp['filters'],
                                kernel_size=int(hp['k_size']),
                                data_format='channels_last',
                                strides=1,
                                padding="same",
                                activation='ELU',
                                input_shape=(hp['max_res'], hp['n_features']),
                                kernel_regularizer=None)
    conv2 = keras.layers.Conv1D(int(hp['filters']),
                                kernel_size=int(hp['k_size']),
                                data_format='channels_last',
                                strides=1,
                                padding="same",
                                activation='ELU',
                                input_shape=(int(hp['max_res']), int(hp['filters'])),
                                kernel_regularizer=None)
    dense = keras.layers.Dense(hp['filters'], activation='ELU')
    x = conv1(input)
    for _ in range(hp['conv_layers'] - 1):
        x = conv2(x)
    for _ in range(hp['dense_layers']):
        x = dense(x)

    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = MaskedConvNet(inputs=input, outputs=output)
    model.compile(loss=weighted_cross_entropy, optimizer=keras.optimizers.Adam(learning_rate=hp['learning_rate'], clipnorm=1.0))
    training_generator = DataGenerator(train_dir, batchSize=hp['batch_size'], max_res=hp['max_res'])
    validation_generator = DataGenerator(test_dir, batchSize=hp['batch_size'], max_res=hp['max_res'])
    history = model.fit(training_generator, epochs=hp['epochs'], callbacks=[WandbCallback(monitor='loss'), EvaluateOnTestCallback(validation_generator)])

