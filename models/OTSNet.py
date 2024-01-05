import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class PFC(nn.Module):
    def __init__(self,channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(3, channels, kernel_size, padding=  kernel_size // 2),
                    #nn.Conv2d(3, channels, kernel_size=3, padding= 1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        print('- Input Layer:', x.shape)
        residual = x
        x = self.depthwise(x)
        print('- Depthwise:', x.shape)
        x += residual
        print('- Residual:', x.shape)
        x = self.pointwise(x)
        print('- Pointwise:', x.shape)
        return x

class OTSNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=1, init_features=32):
        super(OTSNet, self).__init__()
        self.pfc = PFC(32)
        
        batchNorm_momentum = 0.1
        num_features = init_features
        
        self.conv11 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)
        
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

    def forward(self, x):
        print(x.shape)
        # x11 = F.relu(self.bn11(self.conv11(x)))
        # print(x11.shape)
        # x12 = F.relu(self.bn12(self.conv12(x11)))
        # print(x12.shape)
        x12 = self.pfc(x)
        print(x12.shape)
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2,return_indices=True)

        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)

        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x5d = torch.cat((x5d, x53), dim=1)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x4d = torch.cat((x4d, x43), dim=1)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x3d = torch.cat((x3d, x33), dim=1)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x2d = torch.cat((x2d, x22), dim=1)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x1d = torch.cat((x1d, x12), dim=1)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d
    
    
if __name__ == '__main__':
    model = OTSNet(in_channels=3, num_classes=1)
    input_tensor = torch.randn(32, 3, 512, 512)
    out = model(input_tensor)
    print("Output shape:", out.shape)
    