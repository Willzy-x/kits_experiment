import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import yaml
import math
import os
from skimage import transform
from PIL import Image
import torch
from torch.backends import cudnn

from starter_code.utils import load_case


def crop_non_label_slice(img, seg):
    """ The slice dimension tends to be the first dimension. """
    new_img = []
    new_seg = []
    for i in range(seg.shape[0]):
        count = len(np.unique(seg[i, :, :]))
        # add the slices with labels
        if count > 1:
            new_img.append(np.copy(img[i, :, :]))
            new_seg.append(np.copy(seg[i, :, :]))

    new_img = np.array(new_img)
    new_seg = np.array(new_seg)

    return new_img, new_seg


def save_nii(img_data, img_affine, path, slice_tag=None, file_type="", seg=False):
    img = nib.Nifti1Image(img_data, img_affine)
    # seg = nib.Nifti1Image(seg, seg_affine)
    if not seg:
        nib.save(img, os.path.join(path, file_type + str(slice_tag) + '_img.nii'))
    else:
        nib.save(img, os.path.join(path, file_type + str(slice_tag) + '_seg.nii'))


def crop_multi_slices(img, seg=None, new_slice=32):
    """ img and seg should be in the sam shape """
    img_list = []
    seg_list = []

    num_slice = img.shape[0]
    # if the #slice is too small to divide
    if num_slice < new_slice:
        padd = np.zeros((new_slice - num_slice, img.shape[1], img.shape[2]))
        img = np.concatenate((img, padd))
        if seg is not None:
            seg = np.concatenate((seg, padd))
        num_slice = new_slice

    count1 = math.ceil(num_slice / new_slice)
    count2 = num_slice // new_slice
    # if the ndarray cannot be divided equally: overlap = True
    overlap = False
    # compute how many overlap slices
    diff = 0

    for i in range(count1):
        img_list.append(img[i * new_slice:(i + 1) * new_slice, :, :])
        if seg is not None:
            seg_list.append(seg[i * new_slice:(i + 1) * new_slice, :, :])

    if count1 == count2:
        pass
    else:
        # pop the last element whose #slice != new_slice
        img_list.pop()
        if seg is not None:
            seg_list.pop()
        overlap = True
        # add a new one backward to cover the whole image
        img_list.append(img[-new_slice:, :, :])
        if seg is not None:
            seg_list.append(seg[-new_slice:, :, :])
        diff = new_slice - (num_slice - count2 * new_slice)

    return img_list, seg_list, overlap, diff


def crop_as_appointed_size(img, seg, h=-1, w=-1):
    ori_h, ori_w = img.shape[1], img.shape[2]

    check_h, check_w = False, False
    h_check_label = []
    w_check_label = []
    for i in range(ori_h):
        count = len(np.unique(seg[:, i, :]))
        if count != 1 and not check_h:
            h_check_label.append(i)
            check_h = True
        elif count == 1 and check_h:
            h_check_label.append(i - 1)
            check_h = False

    new_check_h = []
    new_check_h.append(h_check_label[0])
    new_check_h.append(h_check_label[-1])

    for i in range(ori_w):
        count = len(np.unique(seg[:, :, i]))
        if count != 1 and not check_w:
            w_check_label.append(i)
            check_w = True
        elif count == 1 and check_w:
            w_check_label.append(i - 1)
            check_w = False

    new_check_w = []
    new_check_w.append(w_check_label[0])
    new_check_w.append(w_check_label[-1])

    new_h = new_check_h[-1] - new_check_h[0]
    new_w = new_check_w[-1] - new_check_w[0]
    print("new: ", new_h, new_w)

    if h != -1 and h >= new_h:
        delta_h = h - new_h
        isUp = True
        for i in range(delta_h):
            if isUp and new_check_h[0] > 0:
                isUp = False
                new_check_h[0] -= 1
            elif not isUp and new_check_h[-1] < ori_h:
                isUp = True
                new_check_h[-1] += 1
            elif new_check_h[0] <= 0:
                new_check_h[-1] += 1
            elif new_check_h[-1] >= ori_h:
                new_check_h[0] -= 1
            else:
                print('vertical full!')

    if w != -1 and w >= new_w:
        delta_w = w - new_w
        isLeft = True
        for i in range(delta_w):
            if isLeft and new_check_w[0] > 0:
                isLeft = False
                new_check_w[0] -= 1
            elif not isLeft and new_check_w[-1] < ori_w:
                isLeft = True
                new_check_w[-1] += 1
            elif new_check_w[0] <= 0:
                new_check_w[-1] += 1
            elif new_check_w[-1] >= ori_w:
                new_check_w[0] -= 1
            else:
                print('horizontal full!')

    new_img = []
    new_seg = []
    for i in range(img.shape[0]):
        temp_sli = img[i, :, :]
        temp_seg = seg[i, :, :]
        temp_sli = Image.fromarray(temp_sli)
        temp_seg = Image.fromarray(temp_seg)
        # w1, h1, w2, h2
        box = (new_check_w[0], new_check_h[0], new_check_w[1], new_check_h[1])
        temp_sli = temp_sli.crop(box)
        temp_seg = temp_seg.crop(box)
        new_img.append(np.array(temp_sli))
        new_seg.append(np.array(temp_seg))

    new_img = np.array(new_img)
    new_seg = np.array(new_seg)

    return new_img, new_seg


def resize_data3d(img, seg, slice=32, h=256, w=256, preserve_seg=False):
    img = transform.resize(img, (slice, h, w), mode='reflect', preserve_range=True)
    if not preserve_seg:
        seg = transform.resize(seg, (slice, h, w), mode='reflect', preserve_range=True).astype('int8')

    return img, seg


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def restore_slice(img_list, ori_slice, new_slice=32, drop=1):
    temp_img_list = list(img_list)
    diff = len(img_list) * new_slice - ori_slice

    if diff == 0:
        img = np.vstack(temp_img_list)
        return img
    else:
        if drop == 1:
            temp_img_list[-2] = temp_img_list[-2][0:new_slice - diff, :, :].copy()
        elif drop == 2:
            temp_img_list[-1] = temp_img_list[-1][diff:new_slice, :, :].copy()

    img = np.vstack(temp_img_list)
    return img


def drop_slices(img, seg, drop_method="odd"):
    num_slice = img.shape[0]
    new_img = []
    new_seg = []
    if drop_method == "odd":
        drop = 1
    else:
        drop = 0

    for s in range(num_slice):
        if (s + drop) % 2 == 0:
            new_img.append(img[s, :, :])
            new_seg.append(seg[s, :, :])
        else:
            pass

    new_img = np.asarray(new_img)
    new_seg = np.asarray(new_seg)


def get_random_patches_records(img, patch_size, patch_num):
    """ Generate records for random patches. Records contains a list of (x, y, z) to be extracted
    :param img:  ndarray shape (D, H, W)
    :param patch_size: list (d, h, w)
    :param patch_num : list (z_num, y_num, x_num)
    :return: a list of patch records contains (x, y, z) coords
    """
    img_z, img_y, img_x = img.shape
    z_range = img_z - patch_size[0]
    y_range = img_y - patch_size[1]
    x_range = img_x - patch_size[2]
    patches = []

    if z_range <= 0 and y_range <= 0 and x_range <= 0:
        print("image too small !!! image shape : {}, patch shape : {}".format(img.shape, patch_size))
        return img

    if z_range <= 0 or y_range <= 0 or x_range <= 0:
        print("image is small !!! image shape : {}, patch shape : {}".format(img.shape, patch_size))

    for i in range(patch_num):
        if x_range <= 0:
            rand_x = 0
        else:
            rand_x = np.random.randint(0, x_range)

        if y_range <= 0:
            rand_y = 0
        else:
            rand_y = np.random.randint(0, y_range)

        if z_range <= 0:
            rand_z = 0
        else:
            rand_z = np.random.randint(0, z_range)

        patches.append((rand_x, rand_y, rand_z))
    return patches


def get_random_patches(img, patch_size, patch_records):
    """ Generate patches for normal image
    :param img:  ndarray shape (D, H, W)
    :param patch_size: list (d, h, w)
    :param patch_records: list [(x, y, z), ...]
    :return: img with shape the same as patch_size
    """
    img_patches = []
    for i, patch_record in enumerate(patch_records):
        rand_x, rand_y, rand_z = patch_record
        img_patch = img[rand_z:rand_z + patch_size[0], rand_y:rand_y + patch_size[1], rand_x:rand_x + patch_size[2]]
        if img_patch.shape != patch_size:
            empty_img_patch = np.zeros((patch_size))
            z, y, x = img_patch.shape
            empty_img_patch[:z, :y, :x] = img_patch[:, :, :]
            img_patch = empty_img_patch
        img_patches.append(img_patch)
    return np.array(img_patches)


def clip_img(img, low_bound, high_bound):
    """ clip a image into the range of [low_bound, high_bound]
    
    : param img: a image array
    : type img: ndarray
    : param low_bound: the lower clip boundary
    : type low_bound: int or float
    : param high_bound: the higher clip boundary
    : type high_bound: int or float
    : return new_img: clipped image array
    : type new_img: ndarray
    """
    new_img = img.copy()

    new_img[np.where(new_img <= low_bound)] = low_bound
    new_img[np.where(new_img >= high_bound)] = high_bound

    return new_img


def normalize(img, method='maxmin'):
    if method == 'maxmin':
        min_img = np.amin(img)
        max_img = np.amax(img)
        return (img - min_img) / (max_img - min_img)
    elif method == 'feature':
        mu = np.mean(img)
        std = np.std(img)
        return (img - mu) / std
    else:
        max_img = np.amax(img)
        return img / max_img


# ------------- Compute Dice coefficient ------------------------------------------------------
def evaluate_dice(predictions, gt, cpu=False, smooth=1e-5):
    if cpu:
        predictions = predictions.cpu().numpy()
        gt = gt.cpu().numpy()

    # Handle case of softmax output
    if len(predictions.shape) == 5:
        predictions = np.argmax(predictions, axis=1)

    # Check predictions for type and dimensions
    if not isinstance(predictions, (np.ndarray, nib.Nifti1Image)):
        raise ValueError("Predictions must by a numpy array or Nifti1Image")
    if isinstance(predictions, nib.Nifti1Image):
        predictions = predictions.get_data()

    if not np.issubdtype(predictions.dtype, np.integer):
        predictions = np.round(predictions)
    predictions = predictions.astype(np.uint8)

    if not predictions.shape == gt.shape:
        raise ValueError(
            ("Predictions have shape {} "
             "which do not match ground truth shape of {}").format(
                predictions.shape, gt.shape
            )
        )

    try:
        # Compute tumor+kidney Dice
        tk_pd = np.greater(predictions, 0)
        tk_gt = np.greater(gt, 0)
        tk_dice = 2 * np.logical_and(tk_pd, tk_gt).sum() / (
                tk_pd.sum() + tk_gt.sum() + smooth
        )
        if np.isinf(tk_dice):
            return [0.0, 0.0]
    except ZeroDivisionError:
        return [0.0, 0.0]

    try:
        # Compute tumor Dice
        tu_pd = np.greater(predictions, 1)
        tu_gt = np.greater(gt, 1)
        tu_dice = 2 * np.logical_and(tu_pd, tu_gt).sum() / (
                tu_pd.sum() + tu_gt.sum() + smooth
        )
        if np.isinf(tu_dice):
            return [tk_dice, 0.0]
    except ZeroDivisionError:
        return [tk_dice, 0.0]

    return [tk_dice, tu_dice]


def diceIoU(lab, tar, cpu=False):
    if cpu:
        lab = lab.cpu().numpy()
        tar = tar.data.cpu().numpy()
    a = np.unique(tar)
    dice = []
    # use lab instead of tar because tar sometimes doesn't have the whole catagory

    for i in range(len(a) - 1):
        tmp_lab = lab.copy()
        tmp_tar = tar.copy()
        tmp_lab[tmp_lab < a[i + 1]] = 0  #
        tmp_tar[tmp_tar < a[i + 1]] = 0
        tmp_lab[tmp_lab >= a[i + 1]] = 1  #
        tmp_tar[tmp_tar >= a[i + 1]] = 1

        interset = (tmp_tar * tmp_lab).sum()
        pre = tmp_lab.sum()
        gt = tmp_tar.sum()
        dice.append(2.0 * interset / (pre + gt))  # list of 2 Dice

    if len(a) <= 1:
        return [0.0, 0.0]
    elif len(a) <= 2:
        dice.append(0.0)

    return dice


# -------------- adding 2 lists with the same length-------------------------------------------
def list_add(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return c


def mirror_when_test(sample_data, axes=(0, 1, 2)):
    mirrored_axes = [0, 0, 0]
    vol = sample_data.numpy().copy()
    if 0 in axes and np.random.uniform() < 0.5:
        vol[:, :, :] = vol[:, :, ::-1]
        mirrored_axes[0] = 1
    if 1 in axes and np.random.uniform() < 0.5:
        vol[:, :, :, :] = vol[:, :, :, ::-1]
        mirrored_axes[1] = 1
    if 2 in axes and len(vol.shape) == 4:
        if np.random.uniform() < 0.5:
            vol[:, :, :, :, :] = vol[:, :, :, :, ::-1]
        mirrored_axes[2] = 1
    return torch.from_numpy(vol), mirrored_axes


def reverse_mirror(sample_seg, mirrored_axes):
    vol = sample_seg.numpy().copy()
    if mirrored_axes[0] == 1:
        vol[:, :, :] = vol[:, :, ::-1]
    if mirrored_axes[1] == 1:
        vol[:, :, :, :] = vol[:, :, :, ::-1]
    if mirrored_axes[2] == 1:
        vol[:, :, :, :, :] = vol[:, :, :, :, ::-1]
    return torch.from_numpy(vol)


def compute_mean(nEpochs, epoch_list, data):
    '''
    docstring for compute mean
    Parameters:
    nEpochs: int, show how many epochs in training/test process.
    epoch_list: list, 1 epoch may contains multiple training/test times.
    data: list, contains dice data, can be lists nested in a list, lists inside have the same length as epoch_list

    Return:
    new_data: list, also can be lists nested in a list, lists inside has the length nEpochs.
    '''
    # whether dice contains only 1 list
    # mutli dice lists:
    new_data = []
    count_epochs = {}
    if isinstance(data[0], list):
        pass
    # only 1 dice list:
    else:
        # transform 1 dice list into a multi-dice list
        data = [data]

    # store the means of dices in muliple lists
    for j in range(len(data)):
        new_data.append([])
        for i in range(nEpochs):
            new_data[j].append(0.0)

    # computing the mean of dices
    for i in range(len(epoch_list)):
        epoch_index = epoch_list[i]
        # i.e epoch_index = ith Epoch
        if count_epochs.__contains__(epoch_index):
            count_epochs[epoch_index] += 1
        else:
            count_epochs[epoch_index] = 1

        for j in range(len(data)):
            try:
                new_data[j][epoch_index - 1] += data[j][i]
            except TypeError:
                print("Wrong type matching!", type(new_data[j][epoch_index - 1]), type(data[j][i]))
                exit(-1)

    for i in range(nEpochs):
        for j in range(len(data)):
            new_data[j][i] /= count_epochs[i + 1]

    return new_data
