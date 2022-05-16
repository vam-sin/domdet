import os
import numpy as np
from tensorflow import keras
import random


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path, batchSize, max_res=None):
        self.dir = data_path
        self.file_list = [f for f in os.listdir(data_path) if '.npz' in f]
        random.seed(13)
        random.shuffle(self.file_list)
        self.batchSize = batchSize
        self.max_res = max_res


    def __len__(self):
        return len(self.file_list) //self.batchSize

    def __getitem__(self, index):
        start_file_num = index*self.batchSize
        end_file_num = (index+1)*self.batchSize
        batch = [self.load_npz(self.dir + self.file_list[i]) for i in range(start_file_num, end_file_num)]
        x = np.stack([chain[0] for chain in batch])
        y = np.stack([chain[1] for chain in batch])
        if self.max_res is not None:
            x, y = x[:, :self.max_res, :], y[:, :self.max_res, :]
        return x, y

    def load_npz(self, npz_path):
        arr = np.load(npz_path, allow_pickle=True)['arr_0']
        return arr.item()['X'], arr.item()['y']
