import torch
import nibabel as nib
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse
import Nets.vnet_ker3 as vnet
from utils1 import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchbiomed.datasets as ds
#import torchbiomed.transforms as biotransforms
#import torchbiomed.loss as bioloss
#import dataset as ds
from torch.backends import cudnn
from starter_code.visualize import visualize
# use the official evaluate function
# from starter_code import evaluation
from starter_code.utils import load_volume, load_segmentation
from utils.vis_tools import draw_contour_volume, visualize_patch
# which gpu to use (node01: 0~2, node02: 0~7, node03: 0~7)
#torch.cuda.set_device(2)
# colors
RED_RGB     = [255, 0, 0]
GREEN_RGB   = [0, 255, 0]
BLUE_RGB    = [0, 0, 255]
CYAN_RGB    = [0, 255, 255]
MAGENTA_RGB = [255, 0, 255]

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

            F.interpolate(prediction, size=img_tensor_shape[2:5], mode='trilinear')

            prediction = prediction.permute(0, 2, 3, 4, 1).contiguous()
            prediction = prediction.view(pred_shape[2], 
                                         pred_shape[3], pred_shape[4],  -1) # 2 labels
            
            prediction = F.log_softmax(prediction, dim=-1) # dim?
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
    model = nn.parallel.DataParallel(model, device_ids=range(0, 1))
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
    img_data = img.get_fdata()[None, None, :, :, :]
    img_data = torch.tensor(img_data, dtype=torch.float32)
    img_shape = img_data.size()
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

        if not configs['dice']:
            prediction = F.log_softmax(prediction, dim=-1)  # dim?
        else:
            prediction = F.softmax(prediction, dim=-1)
        prediction = prediction.cpu().numpy()
        print("before", prediction.shape)

        prediction = np.argmax(prediction, axis=-1)
        print("after", prediction.shape)

    return prediction

def predict_patch(model, configs, img_data):
    #---------- Load Trained Model ------------------------------------------------------
    model.eval()
    model = nn.parallel.DataParallel(model, device_ids=range(0, 1))
    #---------- visualize pipeline ------------------------------------------------------
    img_data = torch.tensor(img_data, dtype=torch.float32)
    img_shape = img_data.size()
    # compute
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

        if not configs['dice']:
            prediction = F.log_softmax(prediction, dim=-1)  # dim?
        else:
            prediction = F.softmax(prediction, dim=-1)
        prediction = prediction.cpu().numpy()
        print("before", prediction.shape)

        prediction = np.argmax(prediction, axis=-1)
        print("after", prediction.shape)

    return prediction


def visualize_case(case_result, model_path, case_id):
    visualize(case_id, case_result, os.path.join(model_path, 'pics', str(case_id).zfill(5)+'_pred'), 
        ori=False, k_color=MAGENTA_RGB, t_color=CYAN_RGB)


def visualize_ori(cid, model_path):
    seg = load_segmentation(cid)
    visualize(cid, seg, os.path.join(model_path, 'pics', str(cid).zfill(5)+'_gt'), ori=True)


def visualize_from_pred(cid, model_path, path='./predictions'):
    file_name = 'prediction_00' + str(cid) + '.nii.gz'
    pred = nib.load(os.path.join(path, file_name))
    pred = pred.get_fdata().copy()
    
    visualize(cid, pred, os.path.join(model_path, 'pics'), ori=False)     


def visual_patch(cid, model, model_path, configs, gt=False, patch_path="./set/test"):
    case_num = "case_" + str(cid).zfill(5)
    patch_path = os.path.join(patch_path, case_num, "none")
    files = os.listdir(patch_path)
    files.sort()
    print(files)
    case_path = os.path.join(model_path, 'pics', case_num)
    if not os.path.exists(case_path):
        os.mkdir(case_path)
    #-------- Load patches --------------------------------------------------------------
    volumes, segs = [], []
    for file in files:
        if file.startswith("img"):
            temp_img = nib.load(os.path.join(patch_path, file))
            volumes.append(temp_img.get_data()[None, None, :,:,:].astype(np.float))

        elif file.startswith("seg"):
            temp_img = nib.load(os.path.join(patch_path, file))
            segs.append(temp_img.get_data().astype(np.int))
        
        else:
            pass
    #--------- Output the pictures ------------------------------------------------------
    if gt:
        for i, vol in enumerate(volumes):
            visualize_patch(vol[0][0], segs[i], destination=os.path.join(
                case_path, "patch_gt_"+str(i).zfill(2)), k_color=RED_RGB, t_color=BLUE_RGB)
    else:
        for i, vol in enumerate(volumes):
            pred = predict_patch(model, configs, vol)
            visualize_patch(vol[0][0], pred, destination=os.path.join(
                case_path, "patch_pred_"+str(i).zfill(2)), k_color=MAGENTA_RGB, t_color=CYAN_RGB)
    

def main():
    # which case to visualize
    #--------- Parse arguments ----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_id', type=int, default=0, 
                        help='case id to visualize.')
    parser.add_argument('--vis_op', type=str, default='cal',
                        help='visualize operations[pred/gt/cal]')
    parser.add_argument('--model_path', type=str, default='./models/Dcpa/',
                        help='Path to load trained model and configs.')
    parser.add_argument('--model_name', type=str, default='vnet_step1_277.tar',
                        help='Name to load trained model.')
    args = parser.parse_args()
    #---------- Load model --------------------------------------------------------------
    # model_name = 'vnet_step1_277.tar'
    # loading configs
    configs = get_config(args.model_path + '/config.yaml')
    model = vnet.VNet(elu=False, nll=True, attention=configs['attention'], nclass=3)#, dropout=True)
    if args.model_name.endswith('.pkl'):
        model = torch.load(os.path.join(args.model_path, args.model_name))
    else:
        checkpoint = torch.load(os.path.join(args.model_path, args.model_name))['model_state_dict']
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
    model.cuda()
    
    # print('Loading Vnet...')
    # net = torch.load(os.path.join(args.model_path, model_name))

    if args.vis_op == 'cal':
        pred = predict_volume_split(model, configs, args.model_path, args.case_id)
        visualize_case(pred, args.model_path, args.case_id)

    elif args.vis_op == 'gt':
        visualize_ori(args.case_id, args.model_path)

    elif args.vis_op == 'cont':
        pred = predict_volume_split(model, configs, args.model_path, args.case_id)
        draw_contour_volume(args.case_id, pred, args.model_path)

    elif args.vis_op == 'patch_gt':
        visual_patch(args.case_id, model, args.model_path, configs, gt=True)

    elif args.vis_op == 'patch_pred':
        visual_patch(args.case_id, model, args.model_path, configs, gt=False)

    else:
        visualize_from_pred(args.case_id)


if __name__ == "__main__":
    main() 
