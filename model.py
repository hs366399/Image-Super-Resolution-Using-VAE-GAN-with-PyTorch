import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class BasicBlock(nn.Module):
    def __init__(self, channels = 128, stride = 1, padding = 1):
        super(BasicBlock, self).__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        
        self.conv_1 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels,
                                kernel_size = 3, stride = self.stride, padding = self.padding)
        self.bn_1 = nn.BatchNorm2d(self.channels)
        self.prelu_1 = nn.PReLU()
        
        self.conv_2 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels,
                                kernel_size = 3, stride = self.stride, padding = self.padding)
        self.bn_2 = nn.BatchNorm2d(self.channels)
        self.prelu_2 = nn.PReLU()
        
    def forward(self, x):
        identity = x
        x = self.prelu_1(self.bn_1(self.conv_1(x)))
        x = self.bn_2(self.conv_2(x)) + identity     
        return self.prelu_2(x)

class preEncoder(nn.Module):
    def __init__(self, channels = 128, stride = 1, padding = 1):
        super(preEncoder, self).__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        
        self.input_pass = nn.Sequential(
                          nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                          nn.BatchNorm2d(32),
                          nn.PReLU(),
            
                          nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                          nn.BatchNorm2d(32),
                          nn.PReLU(),
            
                          nn.Conv2d(in_channels = 32, out_channels = 64,kernel_size = 3, stride = 1, padding = 1),
                          nn.BatchNorm2d(64),
                          nn.PReLU()
                        )
    
    def forward(self, x):
        return self.input_pass(x)

class Encoder(nn.Module):
    def __init__(self, block):
        super(Encoder, self).__init__()
        self.block = block
        
        self.input_conv = nn.Sequential(
                       nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
                       nn.BatchNorm2d(128),
                       nn.PReLU(),

                       nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1),
                       nn.BatchNorm2d(128),
                       nn.PReLU(),
                       )
        
        self.layer_1 = self.make_layers(1)
        self.layer_2 = self.make_layers(1)
        self.layer_3 = self.make_layers(1)
        
        self.downsample_conv_1 = nn.Sequential(
                                 nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 2),
                                 nn.BatchNorm2d(128),
                                 nn.PReLU()
                                 )
        
        self.downsample_conv_2 = nn.Sequential(
                                 nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 2),
                                 nn.BatchNorm2d(128),
                                 nn.PReLU()
                                 )
        
        self.downsample_conv_3 = nn.Sequential(
                                 nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 2),
                                 nn.BatchNorm2d(128),
                                 nn.PReLU()
                                 )
        
    def make_layers(self, layers):
        res_layers = []
        for i in range(layers):
            res_layers.append(self.block())
        return nn.Sequential(*res_layers)  
        
    def forward(self, x):
        x = self.input_conv(x)
        x = self.layer_1(x)
        x = self.downsample_conv_1(x)
        x = self.layer_2(x)
        x = self.downsample_conv_2(x)
        x = self.layer_3(x)
        x = self.downsample_conv_3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, block):
        super(Decoder, self).__init__()
        self.block = block
        
        self.layer_1 = self.make_layers(1)
        self.layer_2 = self.make_layers(1)
        self.layer_3 = self.make_layers(1)
    
        self.trans_conv_1 = nn.Sequential(
                                 nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2),
                                 nn.BatchNorm2d(128),
                                 nn.PReLU()
                                 )
        
        self.trans_conv_2 = nn.Sequential(
                                 nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2),
                                 nn.BatchNorm2d(128),
                                 nn.PReLU()
                                 )
        
        self.trans_conv_3 = nn.Sequential(
                                 nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2),
                                 nn.BatchNorm2d(128),
                                 nn.PReLU()
                                 )
        
        self.trans_conv_4 = nn.Sequential(
                                 nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 2),
                                 nn.BatchNorm2d(128),
                                 nn.PReLU()
                                 )
        
        self.trans_conv_5 = nn.Sequential(
                                 nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 2),
                                 nn.BatchNorm2d(128),
                                 nn.PReLU()
                                 )
        
        self.output_conv = nn.Sequential(
                       nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 2, padding = 0),
                       nn.BatchNorm2d(64),
                       nn.PReLU(),

                       nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1),
                       nn.BatchNorm2d(32),
                       nn.PReLU(),
            
                       nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 3, padding = 1),
                       )
        
    def make_layers(self, layers):
        res_layers = []
        for i in range(layers):
            res_layers.append(self.block())
        return nn.Sequential(*res_layers) 
        
    def forward(self, x):
        x = self.trans_conv_1(x)
        x = self.layer_1(x)
        x = self.trans_conv_2(x)
        x = self.layer_2(x)
        x = self.trans_conv_3(x)
        x = self.layer_3(x)
        x = self.trans_conv_4(x)
        x = self.trans_conv_5(x)
        x = self.output_conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, block):
        super(Discriminator, self).__init__()
        
        self.block = block
        
        self.layer_1 = self.make_layers(1)
        self.layer_2 = self.make_layers(1)
        self.layer_3 = self.make_layers(1)
        
        self.main_conv_1 = nn.Sequential(
                           nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2),
                           nn.BatchNorm2d(32),
                           nn.PReLU()
                        )

        self.main_conv_2 = nn.Sequential(
                           nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 2),
                           nn.BatchNorm2d(128),
                           nn.PReLU()
                        )
            
        self.main_conv_3 = nn.Sequential(
                           nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 2, padding = 2),
                           nn.BatchNorm2d(128),
                           nn.PReLU()
                        )
                            
        self.main_conv_4 = nn.Sequential(
                           nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 2),
                           nn.BatchNorm2d(128),
                           nn.PReLU()
                        )
                
        self.main_conv_5 = nn.Sequential(
                           nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2),
                           nn.BatchNorm2d(64),
                           nn.PReLU()
                        )
        
        self.main_conv_6 = nn.Sequential(
                           nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2),
                           nn.BatchNorm2d(32),
                           nn.PReLU()
                        )
        
        self.main_conv_7 = nn.Sequential(
                           nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1),
                           nn.BatchNorm2d(16),
                           nn.PReLU()
                        )
            
        self.main_conv_8 = nn.Sequential(
                           nn.Linear(256, 128),
                           nn.BatchNorm1d(128),
                           nn.PReLU(),
            
                           nn.Linear(128, 10),
                           nn.BatchNorm1d(10),
                           nn.PReLU(),
            
                           nn.Linear(10, 2),
                        )
        
    def make_layers(self, layers):
        res_layers = []
        for i in range(layers):
            res_layers.append(self.block())
        return nn.Sequential(*res_layers) 
        
    def forward(self, x):
        x = self.main_conv_1(x)
        x = self.main_conv_2(x)
        x = self.layer_1(x)
        x = self.main_conv_3(x)
        x = self.layer_2(x)
        x = self.main_conv_4(x)
        x = self.layer_3(x)
        x = self.main_conv_5(x)
        x = self.main_conv_6(x)
        x = self.main_conv_7(x)
        x = x.view(x.shape[0], -1)
        x = self.main_conv_8(x)        
        return x
