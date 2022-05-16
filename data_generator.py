import os
import numpy as np
from tensorflow import keras
import random


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path, batchSize):
        self.dir = data_path
        self.file_list = [f for f in os.listdir(data_path) if '.npz' in f]
        random.seed(13)
        random.shuffle(self.file_list)
        self.batchSize = batchSize


    def __len__(self):
        return len(self.file_list) //self.batchSize

    def __getitem__(self, index):
        start_file_num = index*self.batchSize
        end_file_num = (index+1)*self.batchSize
        batch = [self.load_npz(self.dir + self.file_list[i]) for i in range(start_file_num, end_file_num)]
        return np.stack([chain[0] for chain in batch]), np.stack([chain[1] for chain in batch])

    def load_npz(self, npz_path):
        arr = np.load(npz_path, allow_pickle=True)['arr_0']
        return arr.item()['X'], arr.item()['y']
