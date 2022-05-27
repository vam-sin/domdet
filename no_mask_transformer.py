import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score

from ppi_transformer import PositionEmbedding
from data_generator import DataGenerator

METRICS = [
    # keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    # keras.metrics.FalseNegatives(name='fn'),
    # keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

class TransformerBlock(layers.Layer):
    def __init__(self, key_dim, value_dim, num_heads, output_shape, rate=0.1, maxlen=1000):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim,
                                             output_shape=output_shape, name='multiheadattn')
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)




    def call(self, inputs, training):
        attn_output = self.att(query=inputs, value=inputs, key=inputs) # this calls the layer with positional arguments (query, value, key)
        attn_output = self.dropout1(attn_output, training=training)
        return self.layernorm1(inputs + attn_output)

def network_builder(hyperp, maxlen=1000, n_features=384):
    combined_input_size = hyperp['n_positional'] + n_features
    inputs = layers.Input(shape=(maxlen, n_features,))
    positional_embedding_layer = PositionEmbedding(maxlen, hyperp['n_positional'])
    positional_embedding = positional_embedding_layer(inputs)
    input_w_position = layers.Concatenate()([inputs, positional_embedding])
    transformer_block = TransformerBlock(key_dim=hyperp['key_dim'], num_heads=hyperp['num_heads'],
                                         value_dim=hyperp['value_dim'], output_shape=combined_input_size)
    x = transformer_block(input_w_position)
    if hyperp['dropout']:
        x = layers.Dropout(0.1)(x)
    x = layers.Dense(hyperp['hidden_nodes'], activation="relu")(x)
    if hyperp['dropout']:
        x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=[inputs], outputs=outputs)
    return model


def weighted_cross_entropy(y_true, y_pred):
    weighting = 0.2
    loss_pos = weighting * y_true * tf.math.log(y_pred)
    loss_neg = (1 - y_true) * tf.math.log(1 - y_pred)
    loss = -1 * (loss_pos + loss_neg)
    return tf.reduce_sum(loss)

if __name__=="__main__":
    # args = sys.argv
    # row_index = int(args[1])
    max_res = 300
    train_dir = 'features/processed/train-val/'
    test_dir = 'features/processed/test/'
    model_save_dir = 'logs/array/'

    # hyperparams = pd.read_csv('transformer_hyperparams.csv')
    # completed = [int(file.split('model_')[-1]) for file in os.listdir(model_save_dir)]
    row_index = 0
    # while row_index in completed:
    #     row_index = np.random.randint(len(hyperparams))
    # hyperparams = hyperparams.loc[row_index]
    hyperparams = {'num_heads': 1,
                 'value_dim': 16,
                 'n_positional': 8,
                 'batch_size': 3,
                 'key_dim': 8,
                 'hidden_nodes': 20,
                 'hidden_layers': 1,
                 'learning_rate': 0.00001,
                 'dropout': False}

    training_generator = DataGenerator(train_dir, batchSize=hyperparams['batch_size'], max_res=max_res)
    validation_generator = DataGenerator(test_dir, batchSize=hyperparams['batch_size'], max_res=max_res)

    model = network_builder(hyperparams,  maxlen=max_res, n_features=4080)
    model.compile(loss=weighted_cross_entropy,
                  metrics=METRICS,
                  optimizer=keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'], clipnorm=1.0))
    model_save_dir = 'logs/array_weight5/'
    os.makedirs(model_save_dir, exist_ok=True)
    filepath = f"{model_save_dir}model_{row_index}"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        restore_best_weights=False,
    )

    history = model.fit(training_generator, epochs=1000, batch_size=hyperparams['batch_size'],
                        # sample_weight=train_sample_weights,
                        callbacks=[checkpoint_callback, es])
    with open(os.path.join(filepath, 'history.pickle'), 'wb') as file_handle:
        pickle.dump(history.history, file_handle)