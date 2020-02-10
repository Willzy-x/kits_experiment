import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import argparse
import Nets.vnet_ker3 as vnet
from utils1 import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchbiomed.datasets as ds
from torch.backends import cudnn
from functools import reduce
from starter_code.evaluation import evaluate
# use the official evaluate function
# from starter_code import evaluation
from starter_code.utils import load_volume, load_segmentation
# from utils.vis_tools import draw_contour_volume
# which gpu to use (node01: 0~2, node02: 0~7, node03: 0~7)
# torch.cuda.set_device(2)
from utils.kits2019_dataloader_3d import Kits2019DataLoader3D
from utils.sliding_window import reconstruct_patches, reconstruct_labels
from sklearn.feature_extraction import image


def predict_volume_slide_window(model, configs, model_path, case_id, patch_size=(160, 160, 128), strides=(40, 40, 20),
                                suffix='nii', do_mirror=True):
    # ---------- Load Trained Model ------------------------------------------------------
    model.eval()
    model = nn.parallel.DataParallel(model, device_ids=range(0, 1))
    # ---------- visualize pipeline ------------------------------------------------------
    case_num = str(case_id).zfill(5)
    print('Loading case_' + case_num)
    # ---------- Load volume -------------------------------------------------------------
    if suffix == 'nii':
        img = load_volume(case_id)
        img_shape = img.shape
        img_data = img.get_fdata()[None, None, :, :, :]
    else:
        img_data = Kits2019DataLoader3D.load_patient(os.path.join('/home/data_share/npy_data/', case_num))[0][0]
        img_data = img_data[None, None, :, :, :]
    c = np.zeros((img_data.shape))
    # ---------- Preprocess --------------------------------------------------------------
    img_data = torch.tensor(img_data, dtype=torch.float32)
    img_shape = img_data.size()
    full_pred = torch.zeros((img_shape))
    background_color = img_data.min()
    # ------------------------------------------------------------------------------------
    d, h, w = img_shape[2], img_shape[3], img_shape[4]
    with torch.no_grad():
        i = 0
        while i < d:
            j = 0
            while j < h:
                k = 0
                while k < w:
                    # ---------- Predict segmentation mask -----------------------------------------------
                    data_window = img_data[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]

                    if data_window.shape[2:] != patch_size:
                        empty_data_window = torch.zeros((patch_size))
                        empty_data_window = empty_data_window[None, None, :, :, :] + background_color
                        dw, hw, ww = data_window.shape[2:]
                        empty_data_window[:, :, :dw, :hw, :ww] = data_window[:, :, :, :, :]

                    mirrored_data, mirrored_axes = mirror_when_test(data_window)

                    output = model(data_window)
                    output = F.interpolate(output, size=patch_size, mode='trilinear')

                    pred = output.permute(0, 2, 3, 4, 1).contiguous()
                    pred = pred.view(patch_size[0], patch_size[1], patch_size[2], -1)

                    if not configs['dice']:
                        pred = F.log_softmax(pred, dim=-1)  # dim?
                    else:
                        pred = F.softmax(pred, dim=-1)

                    pred = torch.argmax(pred, dim=-1)
                    pred = pred[None, None, :, :, :]

                    full_pred[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += \
                        pred[:, :, :min(d - i, patch_size[0]), :min(h - j, patch_size[1]),
                        :min(w - k, patch_size[2])].float().cpu()

                    c[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += 1

                    if do_mirror:
                        mirrored_data = mirrored_data.float()
                        output = model(mirrored_data)

                        output = F.interpolate(output, size=patch_size, mode='trilinear')

                        pred = output.permute(0, 2, 3, 4, 1).contiguous()
                        pred = pred.view(patch_size[0], patch_size[1], patch_size[2], -1)

                        if not configs['dice']:
                            pred = F.log_softmax(pred, dim=-1)  # dim?
                        else:
                            pred = F.softmax(pred, dim=-1)

                        pred = torch.argmax(pred, dim=-1)
                        pred = pred[None, None, :, :, :].cpu()
                        pred = reverse_mirror(pred, mirrored_axes)

                        full_pred[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += \
                            pred[:, :, :min(d - i, patch_size[0]), :min(h - j, patch_size[1]),
                            :min(w - k, patch_size[2])].float()

                        c[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += 1

                    if k + patch_size[2] >= w:
                        break
                    else:
                        k += strides[2]
                if j + patch_size[1] >= h:
                    break
                else:
                    j += strides[1]
            if i + patch_size[0] >= d:
                break
            else:
                i += strides[0]

        full_pred = np.round(full_pred.numpy() / c)

    return full_pred.squeeze()


# load test set
def predict_volume_split(model, configs, model_path, case_id, suffix='nii'):
    # ---------- Load Trained Model ------------------------------------------------------
    model.eval()
    model = nn.parallel.DataParallel(model, device_ids=range(0, 1))
    # ---------- visualize pipeline ------------------------------------------------------
    case_num = str(case_id).zfill(5)
    case_pred = []
    tensor_list = []
    print('Loading case_' + case_num)
    # ---------- Load volume -------------------------------------------------------------
    if suffix == 'nii':
        img = load_volume(case_id)
        img_shape = img.shape
        print("image shape:", img_shape)
        # ---------- Preprocess --------------------------------------------------------------
        img_data = img.get_fdata()
    else:
        img_data = Kits2019DataLoader3D.load_patient(
            os.path.join('/home/data_share/npy_data/', case_num))[0][0]
        img_shape = img_data.shape
        # img_data = img_data[None, None, :, :, :]
    # img_data = clip_img(img=img_data, low_bound=-200, high_bound=400)
    # img_data = normalize(img=img_data, method='feature')
    img_list, _, _, _ = crop_multi_slices(img_data, new_slice=32)
    print("loading tensor...")
    for j in range(len(img_list)):
        img_arr = img_list[j][None, None, :, :, :]
        tensor_list.append(torch.tensor(img_arr, dtype=torch.float32))
    img_tensor_shape = tensor_list[0].size()
    # ---------- Predict segmentation mask -----------------------------------------------
    with torch.no_grad():
        for k in range(len(tensor_list)):
            prediction = model(tensor_list[k])

            if isinstance(prediction, list):
                prediction = prediction[-1]

            pred_shape = prediction.size()

            if pred_shape != img_tensor_shape:
                prediction = F.interpolate(
                    prediction, size=img_tensor_shape[2:5], mode='trilinear')
                pred_shape = img_tensor_shape

            prediction = prediction.permute(0, 2, 3, 4, 1).contiguous()
            prediction = prediction.view(pred_shape[2],
                                         pred_shape[3], pred_shape[4], -1)  # 2 labels

            prediction = F.log_softmax(prediction, dim=-1)  # dim?
            prediction = prediction.cpu().numpy()
            print("before", prediction.shape)

            prediction = np.argmax(prediction, axis=-1)
            print("after", prediction.shape)

            case_pred.append(prediction)

    case_result = restore_slice(case_pred, ori_slice=img_shape[0], new_slice=32)
    return case_result


def predict_volume(model, configs, model_path, case_id, suffix='nii'):
    # ---------- Load Trained Model ------------------------------------------------------
    model.eval()
    model = nn.parallel.DataParallel(model, device_ids=range(0, 1))
    # ---------- visualize pipeline ------------------------------------------------------
    case_num = str(case_id).zfill(5)
    case_pred = []
    tensor_list = []
    print('Loading case_' + case_num)
    # ---------- Load volume -------------------------------------------------------------
    if suffix == 'nii':
        img = load_volume(case_id)
        img_shape = img.shape
        # print("image shape:", img_shape)
        img_data = img.get_fdata()[None, None, :, :, :]
    else:
        img_data = Kits2019DataLoader3D.load_patient(os.path.join('/home/data_share/npy_data/', case_num))[0][0]
        img_data = img_data[None, None, :, :, :]
    # ---------- Preprocess --------------------------------------------------------------
    img_data = torch.tensor(img_data, dtype=torch.float32)
    img_shape = img_data.size()
    # ---------- Predict segmentation mask -----------------------------------------------
    with torch.no_grad():
        prediction = model(img_data)

        if isinstance(prediction, list):
            prediction = prediction[-1]

        pred_shape = prediction.size()

        if pred_shape != img_shape:
            prediction = F.interpolate(
                prediction, size=img_shape[2:5], mode='trilinear')
            pred_shape = img_shape

        prediction = prediction.permute(0, 2, 3, 4, 1).contiguous()
        prediction = prediction.view(pred_shape[2],
                                     pred_shape[3], pred_shape[4], -1)  # 2 labels

        if not configs['dice']:
            prediction = F.log_softmax(prediction, dim=-1)  # dim?
        else:
            prediction = F.softmax(prediction, dim=-1)
        prediction = prediction.cpu().numpy()
        print("before", prediction.shape)

        prediction = np.argmax(prediction, axis=-1)
        print("after", prediction.shape)

    return prediction


def predict_volume_sw(model, configs, model_path, case_id, patch_size=(160, 160, 128), strides=(20, 20, 10),
                      suffix='nii', batch_size=4, mirror=False):
    # ---------- Load Trained Model ------------------------------------------------------
    model.eval()
    model = nn.parallel.DataParallel(model, device_ids=range(0, 1))
    # ---------- visualize pipeline ------------------------------------------------------
    case_num = str(case_id).zfill(5)
    print('Loading case_' + case_num)
    # ---------- Load volume -------------------------------------------------------------
    if suffix == 'nii':
        img = load_volume(case_id)
        img_shape = img.shape
        # print("image shape:", img_shape)
        img_data = img.get_fdata()
    else:
        img_data = Kits2019DataLoader3D.load_patient(os.path.join('/home/data_share/npy_data/', case_num))[0][0]
        img_data = img_data
    # ---------- Preprocess --------------------------------------------------------------
    img_shape = img_data.shape
    if mirror:
        mirrored = np.flip(img_data.copy(), axis=(0, 1, 2))
        m_patches = image.extract_patches(mirrored, patch_size, strides)
        m_patches_shape = m_patches.shape

    patches = image.extract_patches(img_data, patch_size, strides)
    patches_shape = patches.shape

    patches = patches.reshape((-1, 1, patch_size[0], patch_size[1], patch_size[2]))
    patches = torch.tensor(patches, dtype=torch.float32)
    # ---------- Predict segmentation mask -----------------------------------------------
    with torch.no_grad():
        pred_list = []
        nb_batches = int(np.ceil(patches.shape[0] / float(batch_size)))  # batch size
        for batch_id in range(nb_batches):
            batch_index_1, batch_index_2 = batch_id * batch_size, (batch_id + 1) * batch_size
            data = patches[batch_index_1:batch_index_2]
            prediction = model(data)

            if isinstance(prediction, list):
                prediction = prediction[-1]

            pred_shape = prediction.size()

            if pred_shape[2:] != patch_size:
                prediction = F.interpolate(
                    prediction, size=patch_size, mode='trilinear')
                pred_shape[2:] = patch_size

            prediction = prediction.permute(0, 2, 3, 4, 1).contiguous()
            prediction = prediction.view(pred_shape[0], pred_shape[2],
                                         pred_shape[3], pred_shape[4], -1)  # 2 labels

            if not configs['dice']:
                prediction = F.log_softmax(prediction, dim=-1)  # dim?
            else:
                prediction = F.softmax(prediction, dim=-1)

            prediction = prediction.cpu().numpy()
            print("before", prediction.shape)

            prediction = np.argmax(prediction, axis=-1)
            print("after", prediction.shape)
            pred_list.append(prediction)

        final_result = reduce(lambda x, y: np.concatenate((x, y), axis=0), pred_list)
        final_result = final_result.reshape(patches_shape)

        if mirror:
            m_patches = m_patches.reshape((-1, 1, patch_size[0], patch_size[1], patch_size[2]))
            m_patches = torch.tensor(m_patches, dtype=torch.float32)
            pred_list = []
            nb_batches = int(np.ceil(m_patches.shape[0] / float(batch_size)))  # batch size
            for batch_id in range(nb_batches):
                batch_index_1, batch_index_2 = batch_id * batch_size, (batch_id + 1) * batch_size
                data = m_patches[batch_index_1:batch_index_2]
                prediction = model(data)

                if isinstance(prediction, list):
                    prediction = prediction[-1]

                pred_shape = prediction.size()

                if pred_shape[2:] != patch_size:
                    prediction = F.interpolate(
                        prediction, size=patch_size, mode='trilinear')
                    pred_shape[2:] = patch_size

                prediction = prediction.permute(0, 2, 3, 4, 1).contiguous()
                prediction = prediction.view(pred_shape[0], pred_shape[2],
                                             pred_shape[3], pred_shape[4], -1)  # 2 labels

                if not configs['dice']:
                    prediction = F.log_softmax(prediction, dim=-1)  # dim?
                else:
                    prediction = F.softmax(prediction, dim=-1)

                prediction = prediction.cpu().numpy()
                print("before", prediction.shape)

                prediction = np.argmax(prediction, axis=-1)
                print("after", prediction.shape)
                pred_list.append(prediction)

            m_final_result = reduce(lambda x, y: np.concatenate((x, y), axis=0), pred_list)
            m_final_result = m_final_result.reshape(m_patches_shape)

            prediction = reconstruct_labels(final_result, img_shape, 3, strides, m_final_result)

        else:
            prediction = reconstruct_labels(final_result, img_shape, 3, strides)
    print(prediction.shape)

    return prediction
