import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import os 
import numpy as np
import random
from unet_model import unet

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

def get_datapoint(set_name, key):
  if set_name == 'train':
    filename = '../../data/train-val/' + key
  elif set_name == 'val':
    filename = '../../data/train-val/' + key
  elif set_name == 'test':
    filename = '../../data/test_final/' + key 

  X_dict = {}

  arr = np.load(filename, allow_pickle=True)['arr_0']
  X_dict['cmap'] = np.expand_dims(arr.item()['cmap'], axis=0)
  X_dict['ft_vec_1d'] = np.expand_dims(arr.item()['ft_vec_1d'], axis=0)
  y = arr.item()['y']

  return X_dict, y

def generator_from_file(key_lst, batch_size):
  random.shuffle(key_lst)
  i = 0

  while True:
    if i == len(key_lst):
      random.shuffle(key_lst)
      i = 0

    key = key_lst[i]

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

    batch_labels_dict = {}

    batch_labels_dict["out_dom_labels"] = y

    yield X_m, batch_labels_dict

bs = 1
train_gen = generator_from_file(train_keys, bs)
val_gen = generator_from_file(val_keys, bs)
test_gen = generator_from_file(test_keys, bs)

model = unet()

mcp_save = keras.callbacks.ModelCheckpoint('saved_models/unet_simple.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
callbacks_list = [reduce_lr, mcp_save, early_stop]

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"])

history = model.fit_generator(train_gen, epochs = 20, steps_per_epoch = len(train_keys), verbose=1, shuffle = False, validation_data = val_gen, validation_steps = len(val_keys), callbacks = callbacks_list)

'''
error: 13206_train-val.npz
'''
