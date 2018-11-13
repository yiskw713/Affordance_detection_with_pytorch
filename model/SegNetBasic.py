import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

""" 
for the details of SegNet, please refer to this paper:

Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(12), 2481–2495. 
https://doi.org/10.1109/TPAMI.2016.2644615


SegNet Basic is a smaller version of SegNet
Please refer to this repository:
https://github.com/0bserver07/Keras-SegNet-Basic

"""


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x, idx = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        return x, idx


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class SegNetBasic(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.encoder1 = Encoder(in_channel, 64)
        self.encoder2 = Encoder(64, 80)
        self.encoder3 = Encoder(80, 96)
        self.encoder4 = Encoder(96, 128)
        
        self.decoder1 = Decoder(128, 96)
        self.decoder2 = Decoder(96, 80)
        self.decoder3 = Decoder(80, 64)
        self.decoder4 = Decoder(64, out_channel)
        
    def forward(self, x):
        size1 = x.size()
        x, idx1 = self.encoder1(x)

        size2 = x.size()
        x, idx2 = self.encoder2(x)

        size3 = x.size()
        x, idx3 = self.encoder3(x)
        
        size4 = x.size()
        x, idx4 = self.encoder4(x)

        x = F.max_unpool2d(x, idx4, kernel_size=2, stride=2, output_size=size4)
        x = self.decoder1(x)
        
        x = F.max_unpool2d(x, idx3, kernel_size=2, stride=2, output_size=size3)
        x = self.decoder2(x)

        x = F.max_unpool2d(x, idx2, kernel_size=2, stride=2, output_size=size2)
        x = self.decoder3(x)

        x = F.max_unpool2d(x, idx1, kernel_size=2, stride=2, output_size=size1)
        x = self.decoder4(x)

        return x
