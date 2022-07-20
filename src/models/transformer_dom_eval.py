import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from tensorflow.keras.models import load_model
import os 
import numpy as np
import random
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score

test_keys = os.listdir('../../data/casp_data/test/test_final_sxl')

class CustomNonPaddingTokenLoss(keras.losses.Loss):
  def __init__(self, name="custom_dom_loss"):
    super().__init__(name=name)
    self.weights = [0.0, 1.0, 1e-4]

  def call(self, y_true, y_pred):
    # print(y_true, y_pred)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=keras.losses.Reduction.NONE)
    loss = loss_fn(y_true, y_pred, sample_weight=self.weights)
    # y_true_for_mask = tf.math.argmax(y_true, axis=2)
    # mask = tf.cast((y_true_for_mask > 0), dtype=tf.float32)
    # loss = loss * mask 
    return tf.reduce_sum(loss)

loss_fn = CustomNonPaddingTokenLoss()

model = load_model('saved_models/tf_w7', custom_objects = {'CustomNonPaddingTokenLoss': loss_fn}, compile=False)

y_true_all = []
y_pred_all = []

# metrics 
prec_all = []
rec_all = []
acc_all = []
aucroc_all = []
desc_all = []

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
    filename = '/home/vamsi/UCL/projects/domdet/data/casp_data/test/test_final_sxl/' + key 

  X_dict = {}

  arr = np.load(filename, allow_pickle=True)['arr_0']
  # print(arr.item().keys())
  X_dict['X'] = arr.item()['X']
  y = arr.item()['y']
  desc = arr.item()['desc']

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

  return X_dict, np.asarray(y_proc), desc

for i in range(len(test_keys)):
  if i != 152 and i != 60 and i != 70 and i != 168:
    print(i, len(test_keys))
    X, y_true_sample, desc_sample = get_datapoint('test', test_keys[i])
    # print(X["X"].shape, y_true_sample.shape)
    y_pred_sample = model.predict(np.expand_dims(X["X"], axis=0))
    # print(y_pred_sample.shape)
    len_chain = 0 
    print(y_pred_sample.shape, y_true_sample.shape)
    # print(loss_fn(np.expand_dims(y_true_sample, axis=0), y_pred_sample))

    for j in range(len(y_true_sample)):
    	if y_true_sample[j][0] != 1:
    		len_chain += 1
    	else:
    		break

    # masked
    y_pred_sample = y_pred_sample[0][:len_chain]

    y_true_sample = y_true_sample[:len_chain]
    # for j in range(len_chain):
    # 	print(y_pred_sample[j], y_true_sample[j])

    y_true_sample_probs = []
    for j in range(len(y_true_sample)):
    	y_true_sample_probs.append([y_true_sample[j][1], y_true_sample[j][2]])
    y_true_sample = np.argmax(y_true_sample, axis=1)
    y_true_sample -= 1


    y_pred_sample_probs = []

    for j in range(len(y_pred_sample)):
    	y_pred_sample_probs.append([y_pred_sample[j][1], y_pred_sample[j][2]])
    	y_pred_sample[j][0] = 0

    y_pred_sample = np.argmax(y_pred_sample, axis=1)
    y_pred_sample -= 1

    y_true_all.append(y_true_sample)
    y_pred_all.append(y_pred_sample)
    prec = precision_score(y_true_sample, y_pred_sample)
    rec = recall_score(y_true_sample, y_pred_sample)
    acc = accuracy_score(y_true_sample, y_pred_sample)
    aucroc = roc_auc_score(y_true_sample_probs, y_pred_sample_probs)
    print(prec, rec, acc, aucroc)
    prec_all.append(prec)
    rec_all.append(rec)
    acc_all.append(acc)
    aucroc_all.append(aucroc)
    desc_all.append(desc_sample)
    # break


print("Precision: ", np.mean(prec_all))
print("Recall: ", np.mean(rec_all))
print("Accuracy: ", np.mean(acc_all)) 
print("AUC-ROC Score: ", np.mean(aucroc_all))

ds_dict = {'desc': desc_all, 'precision': prec_all, 'recall': rec_all, 'auc-roc': aucroc_all, 'acc': acc_all}  
ds = pd.DataFrame(ds_dict)

ds.to_csv('results/tf_w7_sxl.csv')
'''4 epochs
self.weights = [0.0, 1.0, 0.2]

Precision:  0.751966753735304
Recall:  0.9999392097264438
Accuracy:  0.7519197618150063
AUC-ROC Score:  0.8036300892984806

(Predicting everything as domain)

self.weights = [0.0, 1.0, 0.02]

Precision:  0.7519760476882652
Recall:  0.9997927791831928
Accuracy:  0.7519056037468898
AUC-ROC Score:  0.7935977760298509

'''

'''tf_w7

OG Domains
Precision:  0.7519799517988427
Recall:  0.999985533453888
Accuracy:  0.7520041145242449
AUC-ROC Score:  0.8165088557386865

SXL Domains +
Precision:  0.8211965897616514
Recall:  0.9999748431528718
Accuracy:  0.8211964804670097
AUC-ROC Score:  0.8405028320383295
'''