import numpy as np
import os
from visdom import Visdom
import matplotlib.pyplot as plt
import requests
from utils.generate_bbox import *

data_path = "/home/data_share/npy_data/"
start = 0
end = 209
save_path = "/home/data_share/kidneys/"

# viz = Visdom(server='http://127.0.0.1', port=8097, env='test')
# assert viz.check_connection()

for i in range(0, end+1):
    case_id = str(i).zfill(5) + '.npy'
    data = np.load(os.path.join(data_path, case_id))
    img, seg = data[0], data[1]
    img_coord = get_bbox_3d(seg)

    img_data = img[img_coord[2][0]:img_coord[2][1], img_coord[1][0]:img_coord[1][1], img_coord[0][0]:img_coord[0][1]]
    seg_data = seg[img_coord[2][0]:img_coord[2][1], img_coord[1][0]:img_coord[1][1], img_coord[0][0]:img_coord[0][1]]

    minIndex, maxIndex = -1, -1
    seg_next, img_next = [], []
    endIndex = img_data.shape[0]
    for index in range(seg_data.shape[0]):
        if np.count_nonzero(np.unique(seg_data[index, :, :])) != 0:
            if index == endIndex - 1:
                break
            if np.count_nonzero(np.unique(seg_data[index+1, :, :])) == 0:
                minIndex = index + 1

        else:
            if np.count_nonzero(np.unique(seg_data[index+1, :, :])) != 0:
                maxIndex = index + 1

    # 2 kidneys
    if min == -1 or max == -1:
        patient_data = np.concatenate([img_data[None], seg_data[None]]).astype(np.float32)
        np.save(os.path.join(save_path, case_id), patient_data)
        print("1 Kidney")
    # 1 kidney
    else:
        kidney_img_1, kidney_seg_1 = img_data[0:minIndex, :, :], seg_data[0:minIndex, :, :]
        kidney_img_2, kidney_seg_2 = img_data[maxIndex:endIndex, :, :], seg_data[maxIndex:endIndex, :, :]
        patient_data_1 = np.concatenate([kidney_img_1[None], kidney_seg_1[None]]).astype(np.float32)
        patient_data_2 = np.concatenate([kidney_img_2[None], kidney_seg_2[None]]).astype(np.float32)
        np.save(os.path.join(save_path, case_id + '_0'), patient_data_1)
        np.save(os.path.join(save_path, case_id + '_1'), patient_data_2)
        print("2 Kidneys")

    print("Finished case_" + case_id)

'''
    plt.imshow(kidney_img_1[:, :, kidney_img_1.shape[-1] // 2], cmap='gray')
    viz.matplot(
        plt,
        opts={
            'title': 'case_img1',
            'showlegend': True
        }
    )
    plt.imshow(kidney_seg_1[:, :, kidney_seg_1.shape[-1] // 2], cmap='gray')
    viz.matplot(
        plt,
        opts={
            'title': 'case_label1',
            'showlegend': True
        }
    )
    plt.imshow(kidney_img_2[:, :, kidney_img_2.shape[-1] // 2], cmap='gray')
    viz.matplot(
        plt,
        opts={
            'title': 'case_img2',
            'showlegend': True
        }
    )
    plt.imshow(kidney_seg_2[:, :, kidney_seg_2.shape[-1] // 2], cmap='gray')
    viz.matplot(
        plt,
        opts={
            'title': 'case_label2',
            'showlegend': True
        }
    )
'''