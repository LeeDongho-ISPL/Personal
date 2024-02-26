import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from utils import AverageMeter, calc_psnr
from DataLoader import TrainDataset, TestDataset

testNum = "B3_L5"
from model_MSRD import Net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=3)
    parser.add_argument('--train-file', type=str, default="temp")
    parser.add_argument('--eval-file', type=str, default="temp")
    parser.add_argument('--outputs-dir', type=str,default="weights/test_%s/"%testNum)
    parser.add_argument('--log-dir', type=str,default="logs/log_%s.txt"%testNum)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--GPU', type=int, default=1)
    args = parser.parse_args()
    
    fpLog = open("logs/log_%s.txt"%testNum, 'w') # training loss check
    learning_rate = args.lr
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:%d'%args.GPU if torch.cuda.is_available() else 'cpu')
    
    torch.cuda.set_device(device)
    print("testNum: %s"%testNum)
    print("device: %d"%torch.cuda.current_device())
    torch.manual_seed(args.seed)
    
    model = Net(num_steps = args.num_steps, batchSize=args.batch_size, device = device, channel_in = 1)
    model.to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=0.0000003)

    train_dataset = TrainDataset(setName = "BSD500_JPEG10")
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    sr_test_dataset = TestDataset(setName="classic5_QF10")
    sr_test_dataLoader = DataLoader(dataset=sr_test_dataset, batch_size = 1)
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        
        if (epoch+1)%20 == 0 and learning_rate > 1e-6:
            print("learning rate decay (%f -> %f)"%(learning_rate, learning_rate*0.1))
            learning_rate = learning_rate * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputImg, labelImg = data
        
#                 inputRedImg = inputRedImg.type(torch.cuda.FloatTensor)
                inputImg = inputImg.type(torch.cuda.FloatTensor)
                labelImg = labelImg.type(torch.cuda.FloatTensor)

#                 inputRedImg = inputRedImg.to(device)
                inputImg = inputImg.to(device)
                labelImg = labelImg.to(device)

                predImg = model(inputImg)

                loss = criterion(predImg, labelImg)

                epoch_losses.update(loss.item(), len(inputImg))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputImg))
        # validation check
#         model.set(batchSize=1)
        model.eval()
        validPSNRs = []
        for validData in sr_test_dataLoader:
            inputImg, labelImg = validData
#             inputRedImg = inputRedImg.type(torch.cuda.FloatTensor).to(device)
            inputImg = inputImg.type(torch.cuda.FloatTensor).to(device)
            labelImg = labelImg.type(torch.cuda.FloatTensor).to(device)
            with torch.no_grad():
                predImg = model(inputImg).clamp(0.0, 1.0)
            validPSNRs.append(calc_psnr(predImg, labelImg))
        fpLog.write("%d epoch:\t%f\t%f\n"%(epoch, sum(validPSNRs)/len(validPSNRs), epoch_losses.avg))
        print("%d epoch:\t%f\t%f\n"%(epoch, sum(validPSNRs)/len(validPSNRs), epoch_losses.avg))
#        model.setBatchSize(batchSize=args.batch_size)
        # end (validation check)
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        
    print('%d epoch training complete'%args.num_epochs)
    fpLog.close()

