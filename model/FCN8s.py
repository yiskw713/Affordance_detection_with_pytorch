import torch
import torch.nn as nn
import torchvision


"""
for the details of FCN8s, please refer to this paper:

Shelhamer, E., Long, J., & Darrell, T. (2017). 
Fully Convolutional Networks for Semantic Segmentation. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(4), 640–651. 
https://doi.org/10.1109/TPAMI.2016.2572683

"""


class DeconvBn_2(nn.Module):
    """ Deconvolution(stride=2) => Batch Normilization """
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        return self.bn(self.deconv(x))



class DeconvBn_8(nn.Module):
    """ Deconvolution(stride=8) => Batch Normilization """
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=8, stride=8, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        return self.bn(self.deconv(x))



class FCN8s(nn.Module):
    """ Fully Convolutional Network """
    
    def __init__(self, in_channel, n_classes):
        super().__init__()
        vgg = torchvision.models.vgg16_bn(pretrained=True).features
        
        # confirm the architecture of vgg16 by "print(vgg)"
        
        self.pool3 = vgg[:24]
        self.pool4 = vgg[24:34]
        self.pool5 = vgg[34:]
        
        self.deconv_bn1 = DeconvBn_2(512, 512)
        self.deconv_bn2 = DeconvBn_2(512, 256)
        self.deconv_bn3 = DeconvBn_8(256, n_classes)
        
    def forward(self, x):
        # vgg16
        x3 = self.pool3(x)     # output size => (N, 256, H/8, W/8)
        x4 = self.pool4(x3)    # output size => (N, 512, H/16, W/16)
        x5 = self.pool5(x4)    # output size => (N, 512, H/32, W/32)

        score = self.deconv_bn1(x5)
        score = self.deconv_bn2(x4 + score)
        score = self.deconv_bn3(x3 + score)
        
        return score