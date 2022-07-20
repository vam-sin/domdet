import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import os 
import numpy as np
import random

# att = tf.keras.layers.MultiHeadAttention(num_heads = 2, key_dim = 32)
# transformer block
class TransformerBlock(layers.Layer):
  def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
    super(TransformerBlock, self).__init__()
    self.att = keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
    self.ffn = keras.Sequential([keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim),])
    self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = keras.layers.Dropout(rate)
    self.dropout2 = keras.layers.Dropout(rate)

  def call(self, inputs, training=False):
    attn_output = self.att(inputs, inputs)
    attn_output = self.dropout1(attn_output, training=False)
    out1 = self.layernorm1(inputs + attn_output) # residual link
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=False)

    return self.layernorm2(out1 + ffn_output)

# position embedding + input layer
class TokenAndPositionEmbedding(layers.Layer):
  def __init__(self, maxlen, embed_dim):
    super(TokenAndPositionEmbedding, self).__init__()
    self.token_emb = tf.keras.layers.Dense(embed_dim, activation="relu")
    self.pos_emb = keras.layers.Embedding(input_dim = maxlen, output_dim=embed_dim)

  def call(self, inputs):
    # print(tf.shape(inputs))
    maxlen = tf.shape(inputs)[-2]

    positions = tf.range(start=0, limit=maxlen, delta=1)
    # print(tf.shape(positions))
    position_embeddings = self.pos_emb(positions)
    input_emb = self.token_emb(inputs)

    return input_emb + position_embeddings

# dom transformer model
class domTransformer(keras.Model):
  def __init__(self, num_classes=3, maxlen=3042, embed_dim=32, num_heads=2, ff_dim=32):
    super(domTransformer, self).__init__()
    self.embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
    self.transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    self.dropout1 = layers.Dropout(0.1)
    # self.transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    # self.dropout2 = layers.Dropout(0.1)
    self.ff = layers.Dense(ff_dim, activation = 'relu')
    self.dropout3 = layers.Dropout(0.1)
    self.ff_final = layers.Dense(num_classes, activation="softmax") # 3 classes: 0 padding, 1 non-domain, 2 domain

  def call(self, inputs, training=False):
    x = self.embedding_layer(inputs)
    x = self.transformer_block1(x)
    x = self.dropout1(x, training = training)
    # x = self.transformer_block2(x)
    # x = self.dropout2(x, training = training)
    x = self.ff(x)
    x = self.dropout3(x, training = training)
    x = self.ff_final(x)
    return x

# defining datasets
train_val_keys = os.listdir('/home/vamsi/UCL/projects/domdet/data/processed_tf/train-val')
test_keys = os.listdir('/home/vamsi/UCL/projects/domdet/data/processed_tf/test_final')

random.seed(4)
random.shuffle(train_val_keys)
train_keys = []
val_keys = []

# split train-val into training and validation sets
for i in range(len(train_val_keys)):
  if i <= len(train_val_keys) * 0.2:
    val_keys.append(train_val_keys[i])
  else:
    train_keys.append(train_val_keys[i])

# data generators and required functions
def get_datapoint(set_name, key):
  '''
  provided the set (train/val/test) and the name of the file
  can load the file + generate a mask for the protein chain
  '''
  if set_name == 'train':
    filename = '/home/vamsi/UCL/projects/domdet/data/processed_tf/train-val/' + key
  elif set_name == 'val':
    filename = '/home/vamsi/UCL/projects/domdet/data/processed_tf/train-val/' + key
  elif set_name == 'test':
    filename = '/home/vamsi/UCL/projects/domdet/data/processed_tf/test_final/' + key 

  X_dict = {}

  arr = np.load(filename, allow_pickle=True)['arr_0']
  X_dict['X'] = arr.item()['X']
  y = arr.item()['y']

  mask_vec = []
  y_proc = []

  for i in range(len(y)):
    if y[i] == [0.]: # not in chain
      mask_vec.append([0.])
      y_proc.append([1.,0.,0.])
    elif y[i] == [1.]: # in chain but not domain
      mask_vec.append([1.])
      y_proc.append([0.,1.,0.]) 
    elif y[i] == [2.]: # in chain and domain
      mask_vec.append([1.])
      y_proc.append([0.,0.,1.])

  X_dict["mask_vec"] = np.asarray(mask_vec)

  return X_dict, np.asarray(y_proc)

def generator_from_file(key_lst, batch_size):
  '''
  general generator function used for all three sets
  '''
  random.shuffle(key_lst)
  i = 0

  while True:
    X_batch = []
    y_batch = []
    for j in range(batch_size):
      if i == len(key_lst):
        random.shuffle(key_lst)
        i = 0

      key = key_lst[i]
      # print(key)
      if key in train_keys:
        set_name = 'train'
        X_m, y = get_datapoint(set_name, key)
      elif key in val_keys:
        set_name = 'val'
        X_m, y = get_datapoint(set_name, key)
      elif key in test_keys:
        set_name = 'test'
        X_m, y = get_datapoint(set_name, key)

      i += 1

      X_batch.append(X_m['X'])
      y_batch.append(y)

    X_batch = np.asarray(X_batch)
    y_batch = np.asarray(y_batch)

    yield X_batch, y_batch

# instantiate datasets
bs = 1
train_gen = generator_from_file(train_keys, bs)
val_gen = generator_from_file(val_keys, bs)
test_gen = generator_from_file(test_keys, bs)

for d, l in train_gen:
  print(d.shape, l.shape)
  break

# # model
transformer_model = domTransformer()

class CustomNonPaddingTokenLoss(keras.losses.Loss):
  def __init__(self, name="custom_dom_loss"):
    super().__init__(name=name)
    self.weights = [0.0, 10.0, 0.2]
    self.bs = bs

  def call(self, y_true, y_pred):
    # define loss function
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False, reduction=keras.losses.Reduction.NONE)
    loss = loss_fn(y_true, y_pred)
    tf.print(tf.shape(tf.math.argmax(y_true, axis=2)[0]))

    # get length of chain (used to normalize the loss value for each sample)
    y_true_for_len = tf.math.argmax(y_true, axis=2)
    mask_len = tf.cast((y_true_for_len > 0), dtype=tf.float32)
    chain_len = tf.reduce_sum(mask_len)

    # make loss zero for padding residues
    y_true_for_mask = tf.math.argmax(y_true, axis=2)
    mask = tf.cast((y_true_for_mask == 0), dtype=tf.float32)
    loss_pad = loss * mask * self.weights[0]

    # multiply loss for non-domain residues with their weight
    y_true_for_Ndom = tf.math.argmax(y_true, axis=2)
    mask_Ndom = tf.cast((y_true_for_Ndom == 1), dtype=tf.float32)
    loss_Ndom = loss * mask_Ndom * self.weights[1]

    # multiply loss for domain residues with their weight
    y_true_for_dom = tf.math.argmax(y_true, axis=2)
    mask_dom = tf.cast((y_true_for_dom == 2), dtype=tf.float32)
    loss_dom = loss * mask_dom * self.weights[2]

    # add up losses for domain and non-domain residues
    loss_final = loss_Ndom + loss_dom + loss_pad
    loss_val = tf.reduce_sum(loss_final)

    # normalize per sample loss 
    loss_val_norm = loss_val / chain_len
    # tf.print(loss, summarize=-1)
    # tf.print(loss_dom, summarize=-1)
    # tf.print(y_true, y_pred, loss, loss_pad, loss_dom, loss_Ndom, loss_final, loss_val, loss_val_norm)
  
    return loss_val_norm

def custom_recall(y_true, y_pred):
    y_true_for_len = tf.math.argmax(y_true, axis=2)
    mask_len = tf.cast((y_true_for_len > 0), dtype=tf.float32)
    chain_len = K.cast(tf.reduce_sum(mask_len), "float32")[0]
    tf.print(tf.math.argmax(y_true, axis=2))
    y_true = tf.math.argmax(y_true, axis=2)[0][0:chain_len] - 1 
    y_pred = tf.math.argmax(y_pred, axis=2)[0][0:chain_len] - 1
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

loss = CustomNonPaddingTokenLoss()

mcp_save = keras.callbacks.ModelCheckpoint('saved_models/tf_val_2', save_best_only=True, monitor='val_loss', verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
callbacks_list = [reduce_lr, mcp_save, early_stop]

opt = keras.optimizers.Adam(learning_rate = 1e-5)

transformer_model.compile(optimizer = opt, loss = loss, metrics = [tf.keras.metrics.Precision(), custom_recall, tf.keras.metrics.AUC(), "accuracy"])

history = transformer_model.fit_generator(train_gen, epochs = 50, steps_per_epoch = len(train_keys) // bs, verbose=1, shuffle = False, validation_data = val_gen, validation_steps = len(val_keys) // bs, callbacks = callbacks_list)



'''
0.0825891048 0.315512 0.282815218 0.0451729558 0.0976905674 0.104910783 0.0679256544 0.217530921 0.135812014 0.562793374 0.0212420281 0.312760204 0.059719421 0.0774189606 0.102272652 0.184860393 0.149185538 0.293220729 0.172638655 0.148692682 0.150169179 0.241716132 0.227761313 0.499252379 0.386385679 0.179277971 0.391500562 0.111802034 0.273679584 0.176858276 0.1937543 0.601547837 0.104350165 0.241156057 0.161378518 0.243240148 0.242906496 0.16652064 0.380149066 0.259165496 0.161839232 0.306360811 0.15237464 0.296636313 0.184844926 0.332320839 0.348447591 0.151272044 0.204296157 0.212800011 0.136092246 0.226445869 0.1707789 0.194258109 0.144863516 0.100091413 0.0973429233 0.380114 0.213170931 0.195487127 0.179225802 0.149026498 0.102410793 0.114697158 0.0219910685 0.0408087038 0.0942249745 0.229944751 0.193159625 0.113460876 0.34649688 0.0641478822 0.12169528 0.298173815 0.104877226 0.586996913 0.195667505 0.195185646 0.152563259 0.324662149 0.187961742 0.203839406 0.148927942 0.0960740075 0.0216156896 0.427772373 0.124260604 0.0677333474 0.0911687 0.255744666 0.0626382455 0.134457931 0.0974638686 0.197713718 0.226344869 0.267616361 0.0474724732 0.496437699 0.0901739225 0.267565399 0.161423072 0.19339706 0.44472757 0.154681757 0.171412051 0.235118538 0.252171516 0.104843214 0.19239 0.120260909 0.204878196 0.2460711 0.170671567 0.290526718 0.114121512 0.0858508 0.243322775 0.181331336 0.257100403 0.444946527 0.15279302 0.0649552643 0.0344455689 0.207196757 0.156019107 0.294551313 0.455590695 0.308817238 0.106547706 0.179690987 0.363523304 0.362260312 0.211364865 0.126150325 0.207893759 0.288597435 0.164162323 0.5721578 0.192210719 0.21670495 0.250863612 0.156897306 0.610884607 0.403265923 0.207177803 0.418559849 0.19567591 0.233283803 0.155909836 0.199610636 0.206194192 0.0996662825 0.247138575 0.130480424 0.27273488 0.195528433 0.0725092515 0.368388623 0.150395498 0.265782803 0.131589159 0.0620274059 0.281483322 0.349311382 0.0719438 0.0391575396 0.212009236 0.252367496 0.0920483097 0.1382052 0.166538343 0.242152646 0.320003 0.125139937 0.358436346 0.371333271 0.240854815 0.424678534 0.25845
'''