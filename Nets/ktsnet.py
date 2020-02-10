import torch
import torch.nn as nn
import torch.nn.functional as F
# from AttentionModule import *
from modules.vnet_parts import *
from modules.residual_modules import *


class KTSNet(nn.Module):

    def __init__(self, nclass=3):
        super(KTSNet, self).__init__()
        self.inputTrans = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.backbone = nn.Sequential(
            ResidualUnit3d(32, 64),
            make_nResidualUnits3d(64, 64, depth=3, reduction=4),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            ResidualUnit3d(64, 128),
            make_nResidualUnits3d(128, 128, depth=4, reduction=4),
            DilatedResidualUnit3d(128, 128),
            make_nDilatedResidualUnits3d(128, 128, depth=3, reduction=4)
        )
        self.ppm = PPM(128, x_size=16)

        self.outputTrans = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=4),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.Conv3d(64, nclass, kernel_size=1)
        )

    def forward(self, x):
        x = self.inputTrans(x)
        print(x.size())
        x = self.backbone(x)
        x = self.ppm(x)
        x = self.outputTrans(x)
        return x


if __name__ == '__main__':
    ktsnet = KTSNet()
    x = torch.random.randn((2,1,128,128,128))
    y = ktsnet(x)
    print(y.shape)
