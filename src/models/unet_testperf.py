import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
import os 
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score
from unet_model import unet

test_keys = os.listdir('../../data/test_final/')

model = keras.models.load_model('saved_models/unet_simple.h5')

y_true_all = []
y_pred_all = []

# metrics 
prec_all = []
rec_all = []
acc_all = []

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

for i in range(len(test_keys)):
	print(i, len(test_keys))
	X, y_true_sample = get_datapoint('test', test_keys[i])
	y_pred_sample = np.squeeze(model.predict(X), axis=0)
	# print(y_pred_sample, y_true_sample)

	y_true_all.append(y_true_sample)
	y_pred_all.append(y_pred_sample)
	prec = precision_score(y_true_sample, np.round(y_pred_sample))
	rec = recall_score(y_true_sample, np.round(y_pred_sample))
	acc = accuracy_score(y_true_sample, np.round(y_pred_sample))
	print(prec, rec, acc)

	prec_all.append(prec)
	rec_all.append(rec)
	acc_all.append(acc)

print("Precision: ", np.mean(prec_all))
print("Recall: ", np.mean(rec_all))
print("Accuracy: ", np.mean(acc_all)) 

