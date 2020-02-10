import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
# from network.lib.surface_distance import compute_surface_distances, compute_average_surface_distance


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


def get_prediction(logits):
    size = logits.size()
    if size[1] > 1:
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    else:
        size = list(size)
        # delete channel dim to prevent
        size = [size[0]] + size[2:]
        preds = torch.round(torch.sigmoid(logits)).long().reshape(size)
    return preds


def make_same_size(logits, target):
    assert isinstance(logits, torch.Tensor), "model output {}".format(type(logits))
    size = logits.size()
    if logits.size() != target.size():
        if len(size) == 5:
            logits = F.interpolate(logits, target.size()[2:], align_corners=False, mode='trilinear')
        elif len(size) == 4:
            logits = F.interpolate(logits, target.size()[2:], align_corners=False, mode='bilinear')
        else:
            raise Exception("Invalid size of logits : {}".format(size))
    return logits


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_numpy(tensor_list):
    res = []
    for tensor in tensor_list:
        if isinstance(tensor, torch.Tensor):
            res.append(tensor.detach().cpu().numpy())
        else:
            res.append(tensor)
    return res


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def get_slice(imgs, d):
    res = []
    for img in imgs:
        res.append(img[d])
    return res

class DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.cuda.FloatTensor), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.cuda.FloatTensor), requires_grad=False)

    def forward(self, logits, target):
        # target = target[:, None]
        logits = make_same_size(logits, target)

        size = logits.size()
        N, nclass = size[0], size[1]
        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.cuda.FloatTensor)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        # N x C
        inter = inter.view(N, nclass, -1).sum(2)
        union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
    
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        weighted_dice = dice * self.class_weights
        # sum of all class, mean of all batch
        #
        # use which one ? weighted_dice.mean(), weighted_dice.sum(1).mean(), weighted_dice.sum(0).mean()
        return 1-weighted_dice.mean()