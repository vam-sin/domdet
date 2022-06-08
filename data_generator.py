import os
import numpy as np
from tensorflow.keras.utils import Sequence
from torch.utils.data import Dataset
import random

# 0 is mask 1 is not domain and 2 is domain

class DataGenerator(Sequence):
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
        masks = self.make_mask(y)
        if self.max_res is not None:
            x, y, masks = x[:, :self.max_res, :], y[:, :self.max_res, :], masks[:, :self.max_res, :]
        # y = np.where(y == 2, -1, y) # use to get y with as: {domain:-1, mask:0, non-domain:1}
        y = (y-1).clip(min=0)
        return (x, masks), y

    def make_mask(self, y):
        return np.where(y==0, 0.0, 1.0)

    def load_npz(self, npz_path):
        arr = np.load(npz_path, allow_pickle=True)['arr_0']
        return arr.item()['X'], arr.item()['y']


class PytorchDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dir, max_res):
        'Initialization'
        self.filenames = [f for f in os.listdir(dir) if '.npz' in f]
        self.passed_ids = []
        self.dir = dir
        self.max_res = max_res

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filenames)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        f_name = self.filenames[index]
        self.passed_ids.append(f_name)
        # Load data and get label
        path = os.path.join(self.dir, f_name)
        x, y = self.load_npz(path)
        mask = self.make_mask(y)
        if self.max_res is not None:
            x, y, mask = x[:self.max_res, :], y[:self.max_res, :], mask[:self.max_res, :]
        x = np.transpose(x, [1, 0])
        y = (y-1).clip(min=0)
        y = np.squeeze(y, axis=-1)
        mask = np.squeeze(mask, axis=-1)
        return (x, mask), y

    def make_mask(self, y):
        return np.where(y==0, 0.0, 1.0)

    def load_npz(self, npz_path):
        arr = np.load(npz_path, allow_pickle=True)['arr_0']
        return arr.item()['X'], arr.item()['y']

