import os
import pickle
import tensorflow as tf
from tensorflow import keras
from no_mask_transformer import METRICS, weighted_cross_entropy
from data_generator import DataGenerator


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
    model = build_convnet(hp, n_features=4080)
    model.summary()
    model.compile(loss=weighted_cross_entropy,
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

    history = model.fit(training_generator, epochs=1000, batch_size=hp['batch_size'],
                        # sample_weight=train_sample_weights,
                        callbacks=[checkpoint_callback, es])
    with open(os.path.join(filepath, 'history.pickle'), 'wb') as file_handle:
        pickle.dump(history.history, file_handle)


