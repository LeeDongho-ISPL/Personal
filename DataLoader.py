import os
import numpy as np
import torch
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import struct
import h5py
import scipy.io as sio
from multiprocessing import freeze_support

freeze_support()

class TrainDataset(Dataset):
    def __init__(self, dataPath="../dataset/train/DIV2K", setName = "DIV2K_R0F0_QF10"):
        super(TrainDataset, self).__init__()
        self.inputImgPath = "%s/%s.h5"%(dataPath, setName)
        self.labelImgPath = "%s/DIV2K_R0F0_original.h5"%(dataPath)

    def __getitem__(self, idx):
        with h5py.File(self.inputImgPath, 'r') as f:
            inputImg = f['data'][idx][:, :, :]/255.
        with h5py.File(self.labelImgPath, 'r') as f:
            labelImg = f['data'][idx][:, :, :]/255.
        return inputImg, labelImg
    def __len__(self):
        with h5py.File(self.labelImgPath, 'r') as f:
            return len(f['data'])

class TestDataset(Dataset):
    def __init__(self, dataPath="../dataset/test", setName = "classic5_QF10"):
        super(TestDataset, self).__init__()
        self.inputImgPaths = glob("%s/%s/*"%(dataPath, setName))
        self.labelImgPaths = glob("%s/%s/*"%(dataPath, setName.split("_")[0]))
#         print(self.inputImgPaths, self.labelImgPaths)
        self.inputImgPaths.sort()
        self.labelImgPaths.sort()
    def __getitem__(self, idx):
        inputImg = Image.open(self.inputImgPaths[idx])
        labelImg = Image.open(self.labelImgPaths[idx])
        if inputImg.mode != 'L':
            inputImg = inputImg.convert('L')
        if labelImg.mode != 'L':
            labelImg = labelImg.convert('L')
        inputImg = np.expand_dims(np.array(inputImg), axis=0) / 255.
        labelImg = np.expand_dims(np.array(labelImg), axis=0) / 255.
        return inputImg, labelImg

    def __len__(self):
        return len(self.labelImgPaths)
        
        
        
  