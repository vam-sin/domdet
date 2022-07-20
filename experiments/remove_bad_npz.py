import os
import numpy as np


def load_npz(npz_path):
    arr = np.load(npz_path, allow_pickle=True)['arr_0']
    return arr.item()['X'], arr.item()['y']
train_dir, test_dir = '../features/processed/train-val/', '../features/processed/test/'
i= 0
for dir in [train_dir, test_dir]:
    for f in os.listdir(dir):
        try:
            load_npz(dir + f)
            # print(i)
        except:
            print('failed:', f)
            os.remove(dir + f)
        i += 1