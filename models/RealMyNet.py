import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import resnet

ResNet = resnet.ResNet
Bottleneck = resnet.Bottleneck

def CSA(pretrained=False, root='~/models', **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
   
    return model

csa_block = CSA()

class RealMyNet(nn.Module):
    def __init__(self,in_channels=1,num_classes=1, init_features=64):
        super(RealMyNet, self).__init__()

        batchNorm_momentum = 0.1
        num_features = init_features

        # Encoder
        self.conv11 = nn.Conv2d(in_chaniels, num_features, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)
        self.conv13 = nn.Conv2d(num_features, num_features, kernel_size=1, padding=0)
        self.bn13 = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(num_features, num_features*2, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(num_features*2, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(num_features*2, num_features*2, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(num_features*2, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(num_features*2, num_features*4, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(num_features*4, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(num_features*4, num_features*4, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(num_features*4, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(num_features*4, num_features*4, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(num_features*4, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(num_features*4, num_features*8, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)

        # Decoder
        self.conv53d = nn.Conv2d(num_features*8*2, num_features*8, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv2d(num_features*8*2, num_features*8, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(num_features*8, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(num_features*8, num_features*4, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(num_features*4, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv2d(num_features*4*2, num_features*4, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(num_features*4, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(num_features*4, num_features*4, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(num_features*4, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(num_features*4,  num_features*2, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(num_features*2, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(num_features*2*2, num_features*2, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(num_features*2, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(num_features, num_classes, kernel_size=3, padding=1)
        
        # CSA Module
        self.down1 = csa_block.layer1
        self.down2 = csa_block.layer2
        self.down3 = csa_block.layer3
        self.down4 = csa_block.layer4
        self.up1 = csa_block.layer5
        self.up2 = csa_block.layer6
        self.up3 = csa_block.layer7
        self.up4 = csa_block.layer8


    def forward(self, x):
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv13(x11)))
        x13 = F.relu(self.bn12(self.conv13(x12)))
        x1p, id1 = F.max_pool2d(x13, kernel_size=2, stride=2,return_indices=True)
        # x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2,return_indices=True)
        
        x22 = self.down1(x1p)
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        x33 = self.down2(x2p)
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        x43 = self.down3(x3p)
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)
        
        x53 = self.down4(x4p)
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)
        
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x5d = torch.cat((x5d, x53), dim=1)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x4d = torch.cat((x4d, x43), dim=1)
        x41d = self.up1(x4d)

        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x3d = torch.cat((x3d, x33), dim=1)
        x31d = self.up2(x3d)

        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x2d = torch.cat((x2d, x22), dim=1)
        x21d = self.up3(x2d)

        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x1d = torch.cat((x1d, x12), dim=1)
        x12d = self.up4(x1d)
        x11d = self.conv11d(x12d)

        return x11d
        
if __name__ == '__main__':
    model = RealMyNet(in_channels=3, num_classes=1)
    input_tensor = torch.randn(32, 3, 512, 512)
    out = model(input_tensor)
    print("Output shape:", out.shape)
    