import torch.nn.functional as F
import torch.nn as nn
import torch

def entropy(pk, qk=None):
    
    pk = 1.0 * pk / torch.sum(pk)
    if qk is not None:
        qk = 1.0 * qk / torch.sum(qk)
        S = torch.sum(pk * torch.log(pk/qk))
    else:
        S = -torch.sum(pk * torch.log(pk))
    return S

if __name__ == "__main__":
    x1 = torch.randn([1,2,4,5,6])
    print(entropy(x1))