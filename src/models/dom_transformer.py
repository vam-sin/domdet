import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os 
import random

train_val_keys = os.listdir('../../data/train-val/')
# print(train_val_keys)
test_keys = os.listdir('../../data/test_final/')

random.seed(4)
random.shuffle(train_val_keys)
train_keys = []
val_keys = []

for i in range(len(train_val_keys)):
  if i <= len(train_val_keys) * 0.2:
    val_keys.append(train_val_keys[i])
  else:
    train_keys.append(train_val_keys[i])

def get_datapoint(set_name, key, mask_bin):
  if set_name == 'train':
    filename = 'features/processed/train-val/' + key + '.npz'
  elif set_name == 'val':
    filename = 'features/processed/train-val/' + key + '.npz'
  elif set_name == 'test':
    filename = 'features/processed/test_final/' + key + '.npz'

  arr = np.load(filename, allow_pickle=True)['arr_0']
  X = arr.item()['X']
  y = arr.item()['y']

  mask = np.zeros((3042, 1), dtype=np.float32)

  y_new = []

  for k in range(len(y)):
    if y[k] == 1:
      y_new.append(0) # non domains
      mask[k] = 1 
    elif y[k] == 2:
      y_new.append(1) # domain
      mask[k] = 1
    else:
      y_new.append(0)

  y_new = np.asarray(y_new)

  if mask_bin: # provide mask as input
    return [X, mask], y
  else: # do not provide mask as input
    return X, y

def generator_from_file(key_lst, batch_size, mask_bin):
  random.shuffle(key_lst)
  i = 0

  while True:
    X_batch = []
    y_batch = []
    mask_batch = []
    for j in range(batch_size):
      if i == len(key_lst):
        random.shuffle(key_lst)
        i = 0

      key = key_lst[i]

      if key in train_keys:
        set_name = 'train'
        X_m, y = get_datapoint(set_name, key, mask_bin)
      elif key in val_keys:
        set_name = 'val'
        X_m, y = get_datapoint(set_name, key, mask_bin)
      elif key in test_keys:
        set_name = 'test'
        X_m, y = get_datapoint(set_name, key, mask_bin)

      i += 1

      X_batch.append(X_m)
      y_batch.append(y)

    X_batch = np.asarray(X_batch)
    y_batch = np.asarray(y_batch)

    yield X_batch, y_batch

bs = 1
mask_bin = False
train_gen = generator_from_file(train_keys, bs, mask_bin)
val_gen = generator_from_file(val_keys, bs, mask_bin)
test_gen = generator_from_file(test_keys, bs, mask_bin)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.zero_layer = tf.keras.layers.Dense(embed_dim, use_bias=False, trainable=False,
                                                kernel_initializer=tf.keras.initializers.Zeros())
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.zero_layer(x)
        return x + positions

maxlen = 3042

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen, 4080,))
embedding_layer = PositionEmbedding(maxlen, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit_generator(train_gen, epochs = 20, steps_per_epoch = len(train_keys), verbose=1, shuffle = False, validation_data = val_gen, validation_steps = len(val_keys))


