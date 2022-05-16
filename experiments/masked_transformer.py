import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # if set to '-1' then will run on CPU
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score



class TransformerBlock(layers.Layer):
    def __init__(self, key_dim, value_dim, num_heads, output_shape, d_rate=0.1, maxlen=1000):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim,
                                             output_shape=output_shape, name='multiheadattn')
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(d_rate)
        self.masking_layer = layers.Embedding(input_dim=maxlen, output_dim=maxlen, mask_zero=True)


    def call(self, inputs, mask_inputs, training):
        mask = self.masking_layer.compute_mask(mask_inputs)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=mask) # this calls the layer with positional arguments (query, value, key)
        attn_output = self.dropout1(attn_output, training=training)
        return self.layernorm1(inputs + attn_output) # out 1



class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.zero_layer = tf.keras.layers.Dense(embed_dim, use_bias=False, trainable=False,
                                                kernel_initializer=tf.keras.initializers.Zeros())


    def call(self, x):
        maxlen = tf.shape(x)[-2]
        x = self.zero_layer(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions



maxlen = 1000  # Only consider the first `maxlen` amino acids of each protein chain
if __name__=="__main__":
    x_train, x_test, y_train, y_test, mask_train, mask_test = get_data_and_mask()

    n_features = 384  # Embedding size for each token
    num_heads = 8
    value_dim = 128
    n_positional = 24
    combined_input_size = n_positional + n_features
    batch_size = 128
    key_dim = 16
    attn_output_dim = combined_input_size
    ffn = 128

    inputs = layers.Input(shape=(maxlen, n_features, ))
    mask_inputs = layers.Input(shape=(maxlen))
    positional_embedding_layer = PositionEmbedding(maxlen, n_positional)
    positional_embedding = positional_embedding_layer(inputs)
    input_w_position = layers.Concatenate()([inputs, positional_embedding])
    transformer_block = TransformerBlock(key_dim=key_dim, num_heads=num_heads,
                                         value_dim=value_dim, output_shape=attn_output_dim)
    x = transformer_block(input_w_position, mask_inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(ffn, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x2 = layers.Dense(ffn, activation='relu')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x+x2)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=[inputs, mask_inputs], outputs=outputs)

    # # model loading:
    # model = keras.models.load_model("logs/friday")
    # model.load_weights('logs/my_checkpoint')

    model.compile(loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, clipnorm=1.0))

    model.summary()
    # keras.utils.plot_model(model, "geometricus_transformer.png", show_shapes=True)
    best_score = 0
    for i in range(3000):
        print(f'epoch {i}')
        model.fit(x=[x_train, mask_train], y=y_train, epochs=1, batch_size=batch_size)
        preds = model.predict([x_test, mask_test])
        roc_score = round(roc_auc_score(y_test.flatten(), preds.flatten()), 4)
        print('roc_score on all: ', roc_score)
        subset_pred = preds[:, :150]
        subset_y = y_test[:, :150]
        roc_score2 = round(roc_auc_score(subset_y.flatten(), subset_pred.flatten()), 4)
        print('roc_score on subset: ', roc_score2)
        if roc_score2 > best_score:
            model.save("logs/sat2_linux_res2")
            model.save_weights('logs/checkpoint_sat1_linux_res2')



