import os
import numpy as np
dir  = '../features/processed/train-val/'
for filename in os.listdir(dir):
    try:
        arr = np.load(dir + filename, allow_pickle=True)['arr_0']
        X = arr.item()['X']
        y = arr.item()['y']
        description = arr.item()['desc']
        print(X.shape, y.shape, description)
    except:
        print(filename)

bp=True