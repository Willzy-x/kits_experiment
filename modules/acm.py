import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveContextModule3d(nn.Module):

    def __init__(self, inChan, s):
        super(AdaptiveContextModule3d, self).__init__()
        self.upBranch1 = nn.Conv3d(inChan, inChan, kernel_size=1)
        self.downAdaptivePool = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=s),
            nn.Conv3d(inChan, inChan, kernel_size=1)
        )
        self.upBranch2 = nn.Conv3d(inChan, pow(s, 3), kernel_size=1)
        self.s= s

    def forward(self, x):
        input_size = x.size()
        up1 = self.upBranch1(x)
        down = self.downAdaptivePool(x)
        up2 = F.adaptive_avg_pool3d(up1, output_size=1)
        up2 = up1 + up2
        up2 = self.upBranch2(up2)
        aff = up2.view(input_size[0], -1, pow(self.s, 3))
        down = down.view(input_size[0], pow(self.s, 3), -1)
        out = torch.einsum('bxs,bsy->bxy', aff, down)
        out = out.view(input_size)
        out += up1
        return out


if __name__ == '__main__':
    x = torch.rand([2,4,16,16,16])
    print(x.size())
    acm = AdaptiveContextModule3d(4, 2)
    y = acm(x)
    print(y.size())
