import pandas as pd
import os
import torch
import time
import argparse
from visdom import Visdom
import matplotlib.pyplot as plt
import requests
import numpy as np
from sklearn.feature_extraction import image
from utils1 import *
from utils.kits2019_dataloader_3d import Kits2019DataLoader3D
from utils.sliding_window import reconstruct_patches, mirror_img, reconstruct_labels
from starter_code.utils import load_volume, load_segmentation

if __name__ == '__main__':
    patch_size = (160, 160, 64)
    strides = [20, 20, 10]
    do_mirror = True
    # --------- Parse arguments ----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_num', type=int, default=201,
                        help='case id to visualize.')
    args = parser.parse_args()
    # --------- Connect to visdom server -------------------------------------------------
    viz = Visdom(server='http://127.0.0.1', port=8097, env='test')
    assert viz.check_connection()
    # img_data = load_volume(args.case_num).get_fdata()
    img_data = \
        Kits2019DataLoader3D.load_patient(os.path.join('/home/data_share/npy_data/', str(args.case_num).zfill(5)))[0][1]
    img_shape = img_data.shape
    print(img_data.dtype)
    assert len(np.unique(img_data)) == 3
    print(img_shape)
    plt.imshow(img_data[:, :, img_shape[-1] // 2], cmap='gray')
    viz.matplot(
        plt,
        opts={
            'title': 'case_original',
            'showlegend': True
        }
    )
    mirrored = np.flip(img_data.copy(), axis=(0, 1, 2))
    # print(mirrored.shape)
    # plt.imshow(mirrored[img_shape[-1] // 2, :, :], cmap='gray')
    # viz.matplot(
    #     plt,
    #     opts={
    #         'title': 'case_mirrored',
    #         'showlegend': True
    #     }
    # )
    patches = image.extract_patches(img_data, patch_size, strides)
    print(patches.shape)
    m_patches = image.extract_patches(mirrored, patch_size, strides)
    print(m_patches.shape)
    # patches = patches.reshape((-1, patch_size[0], patch_size[1], patch_size[2]))
    recovered = reconstruct_labels(patches, img_shape, 3, strides, m_patches)
    plt.imshow(recovered[:, :, img_shape[-1] // 2], cmap='gray')
    viz.matplot(
        plt,
        opts={
            'title': 'case_recovered',
            'showlegend': True
        }
    )
    '''
    img_data = img_data[None, None, :, :, :]
    c = torch.zeros((img_data.shape))
    # ---------- Preprocess --------------------------------------------------------------
    img_shape = img_data.shape
    full_pred = torch.zeros((img_shape))
    background_color = img_data.min()
    img_data = torch.tensor(img_data, dtype=torch.float32)
    plt.imshow(img_data.squeeze().numpy()[:, :, img_shape[-1] // 2], cmap='gray')
    viz.matplot(
        plt,
        opts={
            'title': 'case_original',
            'showlegend': True
        }
    )
    # ------------------------------------------------------------------------------------
    d, h, w = img_shape[2], img_shape[3], img_shape[4]

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
                full_pred[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += \
                    data_window[:, :, :min(d - i, patch_size[0]), :min(h - j, patch_size[1]), :min(w - k, patch_size[2])].float().cpu()
                # plt.imshow(data_window.squeeze().numpy()[:, :, patch_size[2] // 2], cmap='gray')
                # viz.matplot(
                #     plt,
                #     opts={
                #         'title': 'case_' + str(args.case_num) + '_' + str(i).zfill(3) + str(j).zfill(3) + str(k).zfill(3),
                #         'showlegend': True
                #     }
                # )
                c[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += 1

                if do_mirror:
                    mirrored_data = mirrored_data.float()
                    reversed_data = reverse_mirror(mirrored_data, mirrored_axes)

                    full_pred[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += \
                        reversed_data[:, :, :min(d - i, patch_size[0]), :min(h - j, patch_size[1]), :min(w - k, patch_size[2])].float()
                    # plt.imshow(reversed_data.squeeze().numpy()[:, :, patch_size[2] // 2], cmap='gray')
                    # viz.matplot(
                    #     plt,
                    #     opts={
                    #         'title': 'case_mirrored_' + str(args.case_num) + '_' + str(i).zfill(3) + str(j).zfill(3) + str(
                    #                 k).zfill(3),
                    #         'showlegend': True
                    #     }
                    # )
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

    full_pred = (full_pred / c).squeeze().numpy()
    plt.imshow(full_pred[:, :, img_shape[-1] // 2], cmap='gray')
    viz.matplot(
        plt,
        opts={
            'title': 'case_recovered',
            'showlegend': True
        }
    )

'''
