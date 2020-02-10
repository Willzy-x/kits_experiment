import torch
import torch.nn as nn
import torch.nn.functional as F
from AttentionModule import *
from modules.vnet_parts import *


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, nclass=2, elu=True, nll=False, attention='DcpA', reduce_rate=2):
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
        elif attention == "nDcpA":
            self.attention = New_DCPA(256, 256)
        elif attention == "SKMDcpa":
            self.attention = SKM_DcpA(256, 256)
        # elif attention == "CbA":
        #     self.isStart = True # ver. 4
        #     self.attention = CrossbarAttention3d(1, 1, 128, 128)
        elif attention == "Pica":
            self.isStart = True # ver. 4
            self.attention =  PicanetG3d([32, 128, 128], 1)
        else:
            self.attention = Identity()
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True, reduction_rate=reduce_rate)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True, reduction_rate=reduce_rate) # 128 128
        self.up_tr64 = UpTransition(128, 64, 1, elu, reduction_rate=reduce_rate) # 128 64
        self.up_tr32 = UpTransition(64, 32, 1, elu, reduction_rate=reduce_rate) # 64 32
        self.out_tr = OutputTransition(32, nclass=nclass, elu=elu, nll=nll) # 32
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
