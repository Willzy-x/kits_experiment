import torch
import torch.nn as nn
import torch.nn.functional as F

# 3d version
''' return #depth residual units '''


def make_nResidualUnits3d(inchannel, outchannel, depth, reduction=2):
    units = []
    # handle unequal channels
    if inchannel != outchannel:
        for i in range(depth):
            if i == depth // 2:
                units.append(ResidualUnit3d(inchannel, outchannel, reduction))
            else:
                units.append(ResidualUnit3d(inchannel, inchannel, reduction))

    else:
        for _ in range(depth):
            units.append(ResidualUnit3d(inchannel, outchannel, reduction))

    return nn.Sequential(*units)


def make_nDilatedResidualUnits3d(inchannel, outchannel, depth, reduction=2):
    units = []
    # handle unequal channels
    if inchannel != outchannel:
        for i in range(depth):
            if i == depth // 2:
                units.append(DilatedResidualUnit3d(inchannel, outchannel, reduction))
            else:
                units.append(DilatedResidualUnit3d(inchannel, inchannel, reduction))

    else:
        for _ in range(depth):
            units.append(DilatedResidualUnit3d(inchannel, outchannel, reduction))

    return nn.Sequential(*units)


# Basic ResidualBlock
class ResidualUnit3d(nn.Module):
    """docstring for ResidualUnit3d."""

    def __init__(self, inchannel, outchannel, stride=1, reduction=2):
        super(ResidualUnit3d, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, inchannel // reduction, kernel_size=1),
            nn.BatchNorm3d(inchannel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv3d(inchannel // reduction, inchannel // reduction, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(inchannel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv3d(inchannel // reduction, outchannel, kernel_size=1),
            nn.BatchNorm3d(outchannel)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1),
                nn.BatchNorm3d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        print("left", out.shape)
        out_1 = self.shortcut(x)
        print("short", out_1.shape)
        out += out_1
        out = F.relu(out)

        return out


class DilatedResidualUnit3d(nn.Module):

    def __init__(self, inChan, outChan, dilation=2, reduction=2, stride=1):
        super(DilatedResidualUnit3d, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(inChan, inChan // reduction, kernel_size=1),
            nn.BatchNorm3d(inChan // reduction),
            nn.ReLU(inplace=True),
            nn.Conv3d(inChan // reduction, inChan // reduction, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inChan // reduction),
            nn.ReLU(inplace=True),
            nn.Conv3d(inChan // reduction, outChan, kernel_size=1),
            nn.BatchNorm3d(outChan)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inChan != outChan:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inChan, outChan, kernel_size=1, stride=1),
                nn.BatchNorm3d(outChan)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)

        out = F.relu(out)
        return out


class PPM(nn.Module):

    def __init__(self, inChan, sizes=(2, 4, 8, 16), x_size=32):
        super(PPM, self).__init__()
        self.ppms = []
        for _, size in enumerate(sizes):
            self.ppms.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool3d(output_size=2),
                    nn.Conv3d(inChan, inChan, kernel_size=1),
                    nn.BatchNorm3d(inChan),
                    nn.ReLU(inplace=True),
                    nn.Upsample(x_size, mode='trilinear'),
                    nn.BatchNorm3d(inChan),
                    nn.ReLU(inplace=True)
                )
            )
        self.outputTrans = nn.Conv3d(inChan * (len(sizes)+1), inChan, kernel_size=1)

    def forward(self, x):
        x_ori = x
        for _, module in enumerate(self.ppms):
            temp_x = module(x_ori)
            x = torch.cat((x, temp_x), dim=1)

        x = self.outputTrans(x)
        return x


class GEFM(nn.Module):

    def __init__(self, m, k):
        super(GEFM, self).__init__()
        blocks = []
        for i in range(m):
            ops = []
            for j in range(k):
                ops.append(
                    nn.Sequential(
                        nn.Conv3d()
                    )
                )