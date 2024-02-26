import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import scipy.misc
from torch.utils.data.dataloader import DataLoader
from DataLoader_JPEG10 import TestDataset
from utils import AverageMeter, calc_psnr
import numpy as np
from glob import glob
import os
from PIL import Image
from torch.utils.data import Dataset

from models_B3_L5_C128 import Net # 모델 파일 이름 변경, 클래스 이름 변경

class TestDataset(Dataset):
    def __init__(self, dataPath="../dataset/test", fileName = "LIVE1", QF = 10): # 이미지 경로 확인 (Test set)
        super(TestDataset, self).__init__()
        self.inputImgPaths = glob("%s/%s_QF%d/*"%(dataPath, fileName, QF))
        self.labelImgPaths = glob("%s/%s/*"%(dataPath, fileName.split("_")[0]))
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
        return inputImg, labelImg, self.inputImgPaths[idx]

    def __len__(self):
        return len(self.labelImgPaths)

parser = argparse.ArgumentParser()
parser.add_argument('--GPU', type=int, default=0)
args = parser.parse_args()

cudnn.benchmark = True
device = torch.device('cuda:%d'%args.GPU if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print("device: %d"%torch.cuda.current_device())

# 모델 이름 확인
modelNames = ["B3_L5_C128_DIV2K_NWD_QF10"]
models = [Net().to(device)]

# 데이터셋 이름 확인
setNames = ["LIVE1", "classic5"]
QFs = [10]#, 20, 30, 40]

for QF in QFs:                                                                        
    
    for idx, model in enumerate(models):
        
        weightPath = "./weights/test_%s/epoch_49.pth"%(modelNames[idx]) # weight 파일 경로 확인
        state_dict = model.state_dict()
        for n, p in torch.load(weightPath, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        model.eval()
        
        for setName in setNames:
            saveRootPath = "./predImgs/%s/%s_QF%d/"%(modelNames[idx], setName, QF) # 저장될 폴더 생성 확인
            test_dataset = TestDataset(fileName=setName, QF=QF)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size = 1)
            for data in test_dataloader:
                inputImg, labelImg, imgPath = data
                imgPath = imgPath[0].split("/")[-1].split(".")[0]
                savePath = saveRootPath + imgPath + ".png"
                inputImg = inputImg.type(torch.cuda.FloatTensor).to(device)
                labelImg = labelImg.type(torch.cuda.FloatTensor).to(device)

                with torch.no_grad():
                    predImg = model(inputImg)
                predImg = np.array(predImg.cpu().squeeze(0).squeeze(0))*255
#                 PSNR = calc_psnr(predImg, labelImg)

                scipy.misc.imsave(savePath, predImg.astype(np.uint8)) # 이미지 저장
                
        print("QF %d 완료"%QF)

                                                                       