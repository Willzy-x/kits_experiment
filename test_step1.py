import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse
import Nets.vnet2 as vnet
from utils1 import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchbiomed.datasets as ds
#import torchbiomed.transforms as biotransforms
#import torchbiomed.loss as bioloss
#import dataset as ds
from torch.backends import cudnn
from starter_code.evaluation import evaluate
# use the official evaluate function
# from starter_code import evaluation
from starter_code.utils import load_volume, load_segmentation
# from utils.vis_tools import draw_contour_volume
# which gpu to use (node01: 0~2, node02: 0~7, node03: 0~7)
#torch.cuda.set_device(2)

# load test set
def predict_volume_split(model, configs, model_path, case_id):
    #---------- Load Trained Model ------------------------------------------------------
    model.eval()
    model = nn.parallel.DataParallel(model, device_ids=range(0, 2))
    #---------- visualize pipeline ------------------------------------------------------
    case_num = str(case_id).zfill(5)
    case_pred = []
    tensor_list = []
    print('Loading case_' + case_num)
    #---------- Load volume -------------------------------------------------------------
    img = load_volume(case_id)
    img_shape = img.shape
    print("image shape:", img_shape)
    #---------- Preprocess --------------------------------------------------------------
    img_data = img.get_fdata()
    # img_data = clip_img(img=img_data, low_bound=-200, high_bound=400)
    # img_data = normalize(img=img_data, method='feature')
    img_list, _, _, _ = crop_multi_slices(img_data, new_slice=32)
    print("loading tensor...")
    for j in range(len(img_list)):
        img_arr = img_list[j][None, None, :, :, :]
        tensor_list.append(torch.tensor(img_arr, dtype=torch.float32))
    img_tensor_shape = tensor_list[0].size()
    #---------- Predict segmentation mask -----------------------------------------------
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
                                         pred_shape[3], pred_shape[4],  -1)  # 2 labels

            prediction = F.log_softmax(prediction, dim=-1)  # dim?
            prediction = prediction.cpu().numpy()
            print("before", prediction.shape)

            prediction = np.argmax(prediction, axis=-1)
            print("after", prediction.shape)

            case_pred.append(prediction)

    case_result = restore_slice(case_pred, ori_slice=img_shape[0])
    return case_result


def predict_volume(model, configs, model_path, case_id):
    #---------- Load Trained Model ------------------------------------------------------
    model.eval()
    model = nn.parallel.DataParallel(model, device_ids=range(0, 2))
    #---------- visualize pipeline ------------------------------------------------------
    case_num = str(case_id).zfill(5)
    case_pred = []
    tensor_list = []
    print('Loading case_' + case_num)
    #---------- Load volume -------------------------------------------------------------
    img = load_volume(case_id)
    img_shape = img.shape
    print("image shape:", img_shape)
    #---------- Preprocess --------------------------------------------------------------
    img_data = img.get_fdata()
    img_shape = img_data.shape
    #---------- Predict segmentation mask -----------------------------------------------
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
                                        pred_shape[3], pred_shape[4],  -1)  # 2 labels

        prediction = F.log_softmax(prediction, dim=1)  # dim?
        prediction = prediction.cpu().numpy()
        print("before", prediction.shape)

        prediction = np.argmax(prediction, axis=1)
        print("after", prediction.shape)

    return prediction


def main():
    #--------- Parse arguments ----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, default=189,
                        help='case id to test.')
    parser.add_argument('--end_id', type=int, default=209,
                        help='case id to test.')
    parser.add_argument('--test_op', type=str, default='case',
                        help='test operations[case/patch]')
    parser.add_argument('--model_path', type=str, default='./models/Dcpa/',
                        help='Path to load trained model and configs.')
    parser.add_argument('--model_name', type=str, default='vnet_step1_277.tar',
                        help='Name to load trained model.')
    args = parser.parse_args()
    #---------- Load model --------------------------------------------------------------
    # model_name = 'vnet_step1_277.tar'
    # loading configs
    configs = get_config(args.model_path + '/config.yaml')
    model = vnet.VNet(elu=False, nll=True,
                      attention=configs['attention'], dropout=True)
    if args.model_name.endswith('.pkl'):
        model = torch.load(os.path.join(args.model_path, args.model_name))
    else:
        checkpoint = torch.load(os.path.join(args.model_path, args.model_name))['model_state_dict']
        model.load_state_dict(
            {k.replace('module.', ''): v for k, v in checkpoint.items()})
    model.cuda()

    # print('Loading Vnet...')
    # net = torch.load(os.path.join(args.model_path, model_name))

    if args.test_op == 'case':
        testF = open(os.path.join(args.model_path, args.model_name+'.csv'), 'w')
        testF.write('Case,Kidney_Dice\n ')

        dice_sum = [0.0, 0.0]
        for i in range(args.start_id, args.end_id + 1):
            pred = predict_volume_split(model, configs, args.model_path, i)

            dice = evaluate(i, pred)

            case_id = str(i).zfill(5)

            testF.write('{},{}\n'.format(case_id, dice[0]))

            dice_sum = list_add(dice_sum, dice)

        tk_avg = dice_sum[0] / (args.end_id - args.start_id + 1)

        testF.write('{},{}'.format('avg', tk_avg))

    testF.close()

if __name__ == "__main__":
    main()
