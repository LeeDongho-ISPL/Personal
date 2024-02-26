from torch import nn

import torch
import torch.nn as nn

import torch.nn.functional as F
from math import sqrt
import numpy as np 
import torch.nn.init as init

def xavier(param):
    init.xavier_uniform_(param)
 
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) *  (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

    
# --------------------------MSRB------------------------------- #

class MSRB_Block(nn.Module):
    def __init__(self, channel_in):
        super(MSRB_Block, self).__init__()
    
        self.relu = nn.ReLU()
        channel = 64

        # self.conv_1 = nn.Sequential(
        #     nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = True),
        #     nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 5, stride = 1, padding = 2, bias = True),
        #     nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 7, stride = 1, padding = 3, bias = True)#DD
        # )

        self.conv_3_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_3_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel * 2, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_5_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.conv_5_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel * 2, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.conv_7_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 7, stride = 1, padding = 3, bias = True)#DD
        self.confusion = nn.Conv2d(in_channels = channel * 5, out_channels = channel, kernel_size = 1, stride = 1, padding = 0, bias = True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))
        output_7_1 = self.relu(self.conv_7_1(x)) #DD

        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        output = torch.cat([output_3_2, output_5_2, output_7_1], 1)#DD
        output = self.confusion(output)
        output = torch.add(output, identity_data)
        return output
#-------------------FeedbackBlock----------------------#


#-------------------------------------------------------------#
class ChannelAttention(nn.Module):
    def __init__(self, n_channels=64, n_layers=2):
        super(ChannelAttention, self).__init__()
        layers = []
        layers.append(nn.AdaptiveAvgPool2d((1, 1))) # Global average pooling
        for idx in range(n_layers):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, bias=True))
        layers.append(nn.Sigmoid())
        self.CA_block = nn.Sequential(*layers)
        
    def forward(self, x):
        CA_map = self.CA_block(x)
        out = x * CA_map.expand_as(x)
        return out

class Net(nn.Module):
    def __init__(self,num_steps , channel_in = 1, n_channels=64, batchSize=1, summary=False, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Net, self).__init__()
        

        self.relu = nn.ReLU()
        self.inputlayer = nn.Conv2d(in_channels=channel_in, out_channels=64, kernel_size=1, stride=1, padding=0)  
        self.num_steps = num_steps
        self.Rblock = self.make_layer(MSRB_Block, 64)
       
        self.ChannelAttention = ChannelAttention(n_layers=2, n_channels=n_channels*1).to(device)  

        #나중에 뺄거면 빼고
        self.recon6 = nn.Conv2d(n_channels, channel_in, kernel_size = 3, stride = 1, padding = 1)
        
  
  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
                if isinstance(m, nn.ConvTranspose2d):   
                    c1, c2, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()  


    def make_layer(self, Rblock, channel_in):
        layers = []
        layers.append(Rblock(channel_in))
        return nn.Sequential(*layers)        

    def forward(self,x):
        num_steps = self.num_steps
        inputlayer = self.relu(self.inputlayer(x)) 

        for i in range(self.num_steps):
            if i < 1:
                h = self.Rblock(inputlayer)
            else:
                h = self.Rblock(h)
        concat = torch.cat([h], 1)
            
   
        ChannelAttention = self.ChannelAttention(concat)
        fout = torch.add(ChannelAttention,inputlayer)               
        out = self.relu(self.recon6(fout))          
      
    
        return out    
    
# model = Net(3)

# test = torch.rand(16, 1, 17, 17)
# print(model)
# model(test)
# print(test.size())