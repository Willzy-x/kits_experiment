import torch
import torch.nn as nn
import torch.nn.functional as F
from AttentionModule import *


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# pass through
class Identity(nn.Module):
    def forward(self, x):
        return x

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        # Normalize the data at the beginning?
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        #split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), dim=1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        if not out.size() == skipxdo.size():
            out = F.interpolate(out, size=skipxdo.size()[2:5], mode='trilinear')

        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, nclass, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, nclass, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nclass)
        self.conv2 = nn.Conv3d(nclass, nclass, kernel_size=1)
        self.relu1 = ELUCons(elu, nclass) # 3 labels
        '''
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax
        self.nclass = nclass
        '''

    def forward(self, x):
        # convolve 32 down to nclass channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        # out = out.view(out.numel() // self.nclass, self.nclass) # 3 labels
        # out = self.softmax(out) # dim?
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False, attention='DcpA'):
        super(VNet, self).__init__()
        # self.isStart = False # ver. 4
        self.in_tr = InputTransition(16, elu) # 16
        self.down_tr32 = DownTransition(16, 1, elu) # 16
        self.down_tr64 = DownTransition(32, 2, elu) # 32
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True) # from 3 to 2 # 64
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        if attention == 'DbA':
            self.attention = DoubleAttention3d(128, 128, reduced_dim=128)
        elif attention == 'DcpA':
            self.attention = DecoupledAttention3d(256, 256)
        elif attention == 'ResA':
            self.attention = ResidualAttention3d(128, 128)
        elif attention == 'BAM':
            self.attention = BAM3d(128)
        elif attention == "CBAM":
            self.attention = CBAM3d(128)
        elif attention == "CbA":
            self.isStart = True # ver. 4
            self.attention = CrossbarAttention3d(1, 1, 128, 128)
        elif attention == "Pica":
            self.isStart = True # ver. 4
            self.attention =  PicanetG3d([32, 128, 128], 1)
        else:
            self.attention = Identity()
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True) # 128 128
        self.up_tr64 = UpTransition(128, 64, 1, elu) # 128 64
        self.up_tr32 = UpTransition(64, 32, 1, elu) # 64 32
        self.out_tr = OutputTransition(32, nclass=2, elu=elu, nll=nll) # 32
    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        #if self.isStart: # ver. 4
        #    x = self.attention(x)
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        # attention calculation
        # if not self.isStart: # ver. 4
        out256 = self.attention(out256)
        # -----------------------------
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
