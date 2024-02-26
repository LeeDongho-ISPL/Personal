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

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss
    
class Multi_Dense_Block1(nn.Module):
    def __init__(self, channel_in):
        super(Multi_Dense_Block1, self).__init__()

        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1)     
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1)               
        
    def forward(self, x):

        conv1 = self.relu(self.conv1(x))

        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1)) 
        
        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))   
        
        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5], 1))        

        conv6 = self.relu(self.conv6(cout5_dense))         
               
        return conv6             

class Multi_Dense_Block2(nn.Module):
    def __init__(self, channel_in):
        super(Multi_Dense_Block2, self).__init__()

        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1)     
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1)               
        
    def forward(self, x):

        conv1 = self.relu(self.conv1(x))

        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1)) 
        
        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))   
        
        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5], 1))        

        conv6 = self.relu(self.conv6(cout5_dense))         
               
        return conv6    
    
class Multi_Dense_Block3(nn.Module):
    def __init__(self, channel_in):
        super(Multi_Dense_Block3, self).__init__()

        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1)     
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1)               
        
    def forward(self, x):

        conv1 = self.relu(self.conv1(x))

        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1)) 
        
        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))   
        
        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5], 1))        

        conv6 = self.relu(self.conv6(cout5_dense))         
               
        return conv6            
      
    
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
    def __init__(self, channel_in = 1, n_channels=64, batchSize=1, summary=False, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Net, self).__init__()
        
        self.relu = nn.PReLU()
        self.inputlayer = nn.Conv2d(in_channels=channel_in, out_channels=64, kernel_size=3, stride=1, padding=1)  
        
        self.MDblock1 = self.make_layer(Multi_Dense_Block1, 64)
        self.MDblock2 = self.make_layer(Multi_Dense_Block2, 64)   
        self.MDblock3 = self.make_layer(Multi_Dense_Block3, 64)        

        self.bottleneck1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bottleneck2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bottleneck3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bottleneck4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bottleneck5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)       
        self.bottleneck6 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)   
        self.bottleneck7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)          
        
        self.Dense_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Dense_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.Dense_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False) 
        self.Dense_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)    
        self.Dense_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)   
        self.Dense_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)        
        
        self.recon1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.recon2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.recon3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)   
        self.recon4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.recon5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)        
        self.recon7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)         
#         self.recon5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)          
        
        self.ChannelAttention1 = ChannelAttention(n_layers=2, n_channels=n_channels*1).to(device)  
        self.ChannelAttention2 = ChannelAttention(n_layers=2, n_channels=n_channels*1).to(device) 
        self.ChannelAttention3 = ChannelAttention(n_layers=2, n_channels=n_channels*1).to(device) 
        self.ChannelAttention4 = ChannelAttention(n_layers=2, n_channels=n_channels*1).to(device) 
        self.ChannelAttention5 = ChannelAttention(n_layers=2, n_channels=n_channels*1).to(device) 
        self.ChannelAttention6 = ChannelAttention(n_layers=2, n_channels=n_channels*1).to(device)         
        
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=False),
#             nn.PReLU(),
#             nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=False),
#             nn.PReLU()
#         )

        self.recon6 = nn.Conv2d(in_channels=64, out_channels=channel_in, kernel_size=3, stride=1, padding=1)    
  
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
                    
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)       
        
    def forward(self,x):
        inputlayer = self.relu(self.inputlayer(x))
#         recon1 = self.relu(self.recon1(inputlayer))
#         out = torch.add(recon1,inputlayer) 

        MDblock1 = self.relu(self.MDblock1(inputlayer))  
        concat1 = torch.cat([MDblock1, inputlayer], 1)
        bottleneck1 = self.relu(self.bottleneck1(concat1))  
        out1 = torch.add(bottleneck1,inputlayer)               
        recon1 = self.relu(self.recon1(out1))
        
        MDblock2 = self.relu(self.MDblock2(recon1))  
        concat2 = torch.cat([MDblock2, recon1], 1)
        bottleneck2 = self.relu(self.bottleneck2(concat2))  
        out2 = torch.add(bottleneck2,recon1)               
        recon2 = self.relu(self.recon2(out2))

        MDblock3 = self.relu(self.MDblock3(recon2))  
        concat3 = torch.cat([MDblock3, recon2], 1)
        bottleneck3 = self.relu(self.bottleneck3(concat3))   
        out3 = torch.add(bottleneck3,recon2)               
        recon3 = self.relu(self.recon3(out3))
        
#         concat5 = torch.cat([recon1, recon2], 1)
#         concat6 = torch.cat([recon2, recon3], 1)
#         concat7 = torch.cat([recon3, recon4], 1)
        
#         concat8 = torch.cat([concat5, concat6], 1)   
#         concat9 = torch.cat([concat6, concat7], 1)          
        concat10 = torch.cat([recon1, recon2, recon3, inputlayer], 1)        
        
        bottleneck6 = self.relu(self.bottleneck6(concat10))      
        ChannelAttention6 = self.ChannelAttention6(bottleneck6)
        out6 = torch.add(ChannelAttention6,inputlayer)               
        recon6 = self.relu(self.recon6(out6))          
        
#         deconv = self.deconv(out)
#         recon3 = self.recon3(out4)       
    
        return recon6    
    