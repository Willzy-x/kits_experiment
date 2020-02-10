import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from AttentionModule import *

def passthrough(x, **kwargs):
    return x

# pass through
class Identity(nn.Module):
    def forward(self, x):
        return x

class ReLUConv(nn.Module):
    def __init__(self, nchan):
        super(ReLUConv, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(num_features=nchan)

    def forward(self, x):
        out = self.relu1(self.in1(self.conv1(x)))
        return out


# Basic ResidualBlock
class ResidualUnit3d(nn.Module):
    """docstring for ResidualUnit3d."""
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualUnit3d, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(outchannel)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.InstanceNorm3d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)

        out = F.relu(out)

        return out
        

# 3d version
''' return #depth residual units '''
def _make_nResidualUnits3d(inchannel, outchannel, depth):
    units = []
    # handle unequal channels
    if inchannel != outchannel:
        for i in range(depth):
            if i == depth // 2:
                units.append(ResidualUnit3d(inchannel, outchannel))
            else:
                units.append(ResidualUnit3d(inchannel, inchannel))

    else:
        for _ in range(depth):
            units.append(ResidualUnit3d(inchannel, outchannel))

    return nn.Sequential(*units)


class InputTransition(nn.Module):
    def __init__(self, outChans):
        super(InputTransition, self).__init__()
        # Normalize the data at the beginning?
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(num_features=outChans)
        self.relu1 = nn.ReLU(inplace=True)

        self.outChans = outChans

        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm3d(num_features=outChans)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.relu1(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        #split input in to 24 channels
        x_list = []
        for i in range(self.outChans):
            x_list.append(x)
        
        xs = reduce(lambda x,y: torch.cat((x, y), dim=1), x_list)
        # x30 = torch.cat((x, x, x, x, x, x, x, x,
        #                  x, x, x, x, x, x, x, x,
        #                  x, x, x, x, x, x, x, x), dim=1)
        out = self.relu1(torch.add(out, xs))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nResiduals, dropout=False):
        super(DownTransition, self).__init__()
        # outChans = 2*inChans
        self.down_conv = ResidualUnit3d(inchannel=inChans,
                    outchannel=outChans, stride=2)
        self.do1 = passthrough
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nResidualUnits3d(inchannel=outChans, 
                    outchannel=outChans, depth=nResiduals)
    
    def forward(self, x):
        out = self.down_conv(x)
        out = self.do1(out)
        out = self.ops(out)
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(outChans//2 + outChans, outChans, kernel_size=3, padding=1)
        # self.in1 = nn.InstanceNorm3d(outChans // 2)
        self.in2 = nn.InstanceNorm3d(outChans)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        # self.do2 = passthrough
        # self.relu1 = nn.ReLU(outChans // 2, inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        if dropout:
            self.do1 = nn.Dropout3d()

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.up_conv(x)
        # out = self.relu1(self.in1(self.up_conv(x)))
        if not out.size() == skipxdo.size():
            out = F.interpolate(out, size=skipxdo.size()[2:5], mode='trilinear')

        xcat = torch.cat((out, skipxdo), 1)
        out = self.relu2(self.in2(self.conv2(xcat)))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, nclass, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, nclass, kernel_size=1)
        self.in1 = nn.InstanceNorm3d(nclass)
        self.relu1 = nn.ReLU(inplace=True) # 3 labels
        # self.conv2 = nn.Conv3d(nclass, nclass, kernel_size=1)
        
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax
        self.nclass = nclass
        

    def forward(self, x):
        # convolve 32 down to nclass channels
        out = self.relu1(self.in1(self.conv1(x)))
        # out = self.conv2(out)
        # make channels the last axis
        out = self.softmax(out, dim=1)
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        # out = out.view(out.numel() // self.nclass, self.nclass) # 3 labels
        # out = self.softmax(out) # dim?
        # treat channel 0 as the predicted output
        return out


class SupervisionTransition(nn.Module):
    def __init__(self, inChans, nclass, nll):
        super(SupervisionTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, nclass, kernel_size=1)

        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax
        self.nclass = nclass

    def forward(self, x):
        # convolve inChans down to nclass channels
        out = self.conv1(x)
        out = self.softmax(out, dim=1)
        return out

class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu, nll=False, attention='DcpA', nclass=2):
        super(VNet, self).__init__()
        # self.isStart = False # ver. 4
        #-------- Encoders ------------------------------------------------------------------
        self.in_tr = InputTransition(24) # 16
        self.down_tr30 = DownTransition(24, 60, 1) # 16
        self.down_tr60 = DownTransition(60, 120, 2) # 32
        self.down_tr120 = DownTransition(120, 240, 3, dropout=True) # from 3 to 2 # 64
        self.down_tr240 = DownTransition(240, 320, 4, dropout=True)
        self.down_tr320 = DownTransition(320, 320, 5, dropout=True)

        #-------- Attention module ----------------------------------------------------------
        if attention == 'Dcpa':
            self.attention = DecoupledAttention3d(320, 320)
        else:
            self.attention = Identity()

        #-------- Decoders ------------------------------------------------------------------
        self.up_tr320 = UpTransition(320, 320, dropout=True)
        self.up_tr240 = UpTransition(320, 240, dropout=True) # 128 128
        self.up_tr120 = UpTransition(240, 120) # 128 64
        self.up_tr60 = UpTransition(120, 60) # 64 32
        self.up_tr30 = UpTransition(60, 24)
        self.out_tr = OutputTransition(24, nclass=nclass, nll=nll) # 32

        #-------- Multi-resolution output transition ----------------------------------------
        self.c1 = SupervisionTransition(320, nclass, nll=nll)
        self.c2 = SupervisionTransition(240, nclass, nll=nll)
        self.c3 = SupervisionTransition(120, nclass, nll=nll)
        self.c4 = SupervisionTransition(60, nclass, nll=nll)
        self.c5 = SupervisionTransition(24, nclass, nll=nll)
    
    def forward(self, x):
        #if self.isStart: # ver. 4
        #    x = self.attention(x)
        out_list = []
        tar_size = x.size()
        out30 = self.in_tr(x)
        out60 = self.down_tr30(out30)
        out120 = self.down_tr60(out60)
        out240 = self.down_tr120(out120)
        out320 = self.down_tr240(out240)
        out320a = self.down_tr320(out320)
        # attention calculation
        # if not self.isStart: # ver. 4
        out320a = self.attention(out320a)
        # -----------------------------
        out = self.up_tr320(out320a, out320)
        out_list.append(self.c1(out))
        out = self.up_tr240(out320, out240)
        out_list.append(self.c2(out))
        out = self.up_tr120(out, out120)
        out_list.append(self.c3(out))
        out = self.up_tr60(out, out60)
        out_list.append(self.c4(out))
        out = self.up_tr30(out, out30)
        out_list.append(self.c5(out))
        out = self.out_tr(out)
        out_list.append(out)
        return out_list

if __name__ == "__main__":
    x = torch.randn([2, 1, 32, 32, 32])
    net = VNet(elu=False, nll=False, attention=None)
    y = net(x)
    print(y[1].size())
    
