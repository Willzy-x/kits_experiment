import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.vnet_parts import *
from modules.acm import *


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, nclass=2, elu=True, nll=False, attention='DcpA'):
        super(VNet, self).__init__()
        # self.isStart = False # ver. 4
        self.in_tr = InputTransition(16, elu) # 16
        self.down_tr32 = DownTransition(16, 1, elu) # 16
        self.down_tr64 = DownTransition(32, 2, elu) # 32
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True) # from 3 to 2 # 64
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.acm1 = AdaptiveContextModule3d(256, 1)
        self.acm2 = AdaptiveContextModule3d(256, 2)
        self.trans = nn.Conv3d(256*2, 256, 1)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True) # 128 128
        self.up_tr64 = UpTransition(128, 64, 1, elu) # 128 64
        self.up_tr32 = UpTransition(64, 32, 1, elu) # 64 32
        self.out_tr = OutputTransition(32, nclass=nclass, elu=elu, nll=nll) # 32
    # The network topology as described in the diagram
    # in the VNet paper

    def forward(self, x):
        #if self.isStart: # ver. 4
        #    x = self.attention(x)
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out256_1 = self.acm1(out256)
        out256_2 = self.acm2(out256)
        out256 = self.trans(torch.cat((out256_1, out256_2), 1)) + out256
        # -----------------------------
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out