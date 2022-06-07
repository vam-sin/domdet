import os
import pickle
import tensorflow as tf
from tensorflow import keras
from no_mask_transformer import METRICS, weighted_cross_entropy
from data_generator import DataGenerator


def masked_weighted_cross_entropy(y_true, y_pred):
    weighting  = 0.2
    tf.print(y_true.shape)
    y, masks = tf.split(y_true, 2, axis=1)
    tf.print(y.shape)
    loss_pos = weighting * y * tf.math.log(y_pred)
    loss_neg = (1 - y) * tf.math.log(1 - y_pred)
    loss = -1 * (loss_pos + loss_neg) * masks
    return tf.reduce_sum(loss)

class MaskedConvNet(tf.keras.Model):
    def __init__(self, model_settings, **kwargs):
        super(MaskedConvNet, self).__init__(**kwargs)
        self.model_settings = model_settings
        self.model = build_convnet(model_settings, n_features=4080)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker
            # self.reconstruction_loss_tracker,
            # self.kl_loss_tracker,
        ]


    def train_step(self, data):
        x, y_tup = data
        y, masks = tf.unstack(y_tup, axis=1)
        y_pred = self.model(x)
        with tf.GradientTape() as tape:
            loss = weighted_cross_entropy(y, y_pred)
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(loss)
            # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            # self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            # "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            # "kl_loss": self.kl_loss_tracker.result(),
        }

def build_convnet(hp, n_features):
    inputs = keras.layers.Input(shape=(hp['max_res'], n_features))
    conv1 = keras.layers.Conv1D(hp['filters'],
                                kernel_size=hp['k_size'],
                                data_format='channels_last',
                                strides=1,
                                padding="same",
                                activation='ELU',
                                input_shape=(hp['max_res'], n_features),
                                kernel_regularizer=None)
    conv2 = keras.layers.Conv1D(hp['filters'],
                                kernel_size=hp['k_size'],
                                data_format='channels_last',
                                strides=1,
                                padding="same",
                                activation='ELU',
                                input_shape=(hp['max_res'], hp['filters']),
                                kernel_regularizer=None)
    dense = keras.layers.Dense(hp['filters'], activation='ELU')
    print(hp['max_res'])
    x = conv1(inputs)
    print(x.shape)
    x = conv2(x)
    print(x.shape)
    x = conv2(x)
    print(x.shape)
    for _ in range(hp['dense_layers']):
        x = dense(x)
        print(x.shape)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=output)
    return model




if __name__=="__main__":
    # args = sys.argv
    # row_index = int(args[1])

    train_dir = 'features/processed/train-val/'
    test_dir = 'features/processed/test/'
    model_save_dir = 'conv_logs/'

    hp = {'n_conv_layers': 1,
            'batch_size': 3,
            'k_size': 5,
            'filters':32,
            'dense_layers': 1,
            'learning_rate': 0.00001,
            'max_res':300
                    }
    training_generator = DataGenerator(train_dir, batchSize=hp['batch_size'], max_res=hp['max_res'])
    validation_generator = DataGenerator(test_dir, batchSize=hp['batch_size'], max_res=hp['max_res'])
    model_conv = MaskedConvNet(hp)
    model_conv.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate'], clipnorm=1.0))
    history = model_conv.fit(training_generator, epochs=1)

    model = build_convnet(hp, n_features=4080)
    model.summary()
    model.compile(loss=masked_weighted_cross_entropy,
                  metrics=METRICS,
                  optimizer=keras.optimizers.Adam(learning_rate=hp['learning_rate'], clipnorm=1.0))
    os.makedirs(model_save_dir, exist_ok=True)
    filepath = f"{model_save_dir}model"
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

    history = model.fit(training_generator, validation_data=validation_generator, epochs=1, batch_size=hp['batch_size'],
                        # sample_weight=train_sample_weights,
                        callbacks=[checkpoint_callback, es])
    with open(os.path.join(filepath, 'history.pickle'), 'wb') as file_handle:
        pickle.dump(history.history, file_handle)


'''
2352/2352 [==============================] - 1629s 692ms/step - loss: 154.7525 - tp: 1271286.0000 - fp: 62894.0000 - tn: 782530.0000 - fn: 90.0000 - accuracy: 0.9702 - precision: 0.9529 - recall: 0.9999 - auc: 0.9698 - prc: 0.9621
753/2352 [========>.....................] - ETA: 20:38 - loss: 151.2803 - tp: 411491.0000 - fp: 18285.0000 - tn: 247904.0000 - fn: 20.0000 - accuracy: 0.9730 - precision: 0.9575 - recall: 1.0000 - auc: 0.9696 - prc: 0.9623
'''