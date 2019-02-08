import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


"""
for the details of UNet, please refer to this paper:

Ronneberger, O., Fischer, P., & Brox, T. (2015). 
U-net: Convolutional networks for biomedical image segmentation. 
In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) (Vol. 9351, pp. 234–241). Springer Verlag. 
https://doi.org/10.1007/978-3-319-24574-4_28

"""


class DoubleConv(nn.Module):
    """ (Conv => BatchNorm => ReLU) * 2 """
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x


class Down(nn.Module):
    """ MaxPooling => DoubleConv """
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x


class Up(nn.Module):
    """ UpSampling => concat => DoubleConv """
    
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.double_conv = DoubleConv(in_channel+out_channel, out_channel) # after concat
        
    def forward(self, x, skipped_layer):
        """ the size of x is the same as the input from skipped layer """
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skipped_layer], dim=1)
        x = self.double_conv(x)
        return x



class UNet(nn.Module):
    """ the size of input is (3, 240, 320) and that of output is the same """
    
    def __init__(self, in_channel, n_classes):
        super().__init__()
        
        self.double_conv = DoubleConv(in_channel, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up4 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up1 = Up(64, 32)
        self.conv = nn.Conv2d(32, n_classes, 1)
        
    def forward(self, x):
        # the left side of U-Net
        x1 = self.double_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        
        # the right side of U-Net
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.conv(x)
        
        return x
