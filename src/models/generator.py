import numpy as np
import os
import random

train_val_keys = os.listdir('features/processed/train-val/')
test_keys = os.listdir('features/processed/test_final/')

random.seed(4)
train_val_keys = random.shuffle(train_val_keys)
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


def generator_from_file(key_lst, batch_size, mask_bin=False):
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