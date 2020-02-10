import torch 
import numpy as np

# https://github.com/pytorch/pytorch/issues/1249


def dice_coeff(pred, target, cpu=True):
    if cpu:
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

    smooth = 1e-5
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
