import numpy as np
import torch
from utils import img_resolve as ir
import os
from torch.utils.data import Dataset
from scipy.io import loadmat, savemat

class MyData(Dataset):
    def __init__(self, data_path, Train = True):
        self.Train = Train
        if Train:
            path = os.path.join(data_path, 'train.mat')
        else:
            path = os.path.join(data_path, 'valid.mat')
        self.mat_data = loadmat(path)['data']

    def __getitem__(self, index):
        data = self.mat_data[index]
        return data
    def  __len__(self):
        return self.mat_data.shape[0]

if __name__ == '__main__':
    data_path = r"D:\learn\condata-init\dataset"
    Train_dataset = MyData(data_path,Train=True)
    print(Train_dataset)
    print()