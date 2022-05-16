import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))


class TransformerBlock(layers.Layer):
    def __init__(self, key_dim, value_dim, num_heads, output_shape, rate=0.1, maxlen=1000):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim,
                                             output_shape=output_shape, name='multiheadattn')
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.masking_layer = layers.Embedding(input_dim=maxlen, output_dim=maxlen, mask_zero=True)


    def call(self, inputs, mask_inputs, training):
        mask = self.masking_layer.compute_mask(mask_inputs)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=mask) # this calls the layer with positional arguments (query, value, key)
        attn_output = self.dropout1(attn_output, training=training)
        return self.layernorm1(inputs + attn_output)


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



