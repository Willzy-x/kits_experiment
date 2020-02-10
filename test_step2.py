import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import argparse
import Nets.vnet_ker3 as vnet
from utils1 import *
from functools import reduce
from starter_code.evaluation import evaluate
# use the official evaluate function
# from starter_code import evaluation
from starter_code.utils import load_volume, load_segmentation
# from utils.vis_tools import draw_contour_volume
# which gpu to use (node01: 0~2, node02: 0~7, node03: 0~7)
# torch.cuda.set_device(2)
from utils.kits2019_dataloader_3d import Kits2019DataLoader3D
from sklearn.feature_extraction import image
from visual import predict_patch
from test_methods import *
from utils.vis_tools import visualize_patch
from train_and_test import test_nll, test_dice


def main():
    # --------- Parse arguments ----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, default=189,
                        help='case id to test.')
    parser.add_argument('--end_id', type=int, default=209,
                        help='case id to test.')
    parser.add_argument('--test_op', type=str, default='case', choices=["case", "patch"],
                        help='test operations')
    parser.add_argument('--model_path', type=str, default='./models/Dcpa/',
                        help='Path to load trained model and configs.')
    parser.add_argument('--sliding_window', type=bool, default=False, help='Whether to use sliding window to test.')
    parser.add_argument('--model_name', type=str, default='vnet_step1_277.tar',
                        help='Name to load trained model.')
    parser.add_argument('--test_path', type=str, default='./set/test',
                        help='Path to load test patches.')
    parser.add_argument('--data_suffix', type=str, default='nii', help='Which type of data to load.')
    parser.add_argument('--strides', type=int, nargs=3, help='step for sliding window.')
    args = parser.parse_args()
    # ---------- Load model --------------------------------------------------------------
    # loading configs
    configs = get_config(args.model_path + '/config.yaml')
    model = vnet.VNet(elu=False, nll=True,
                      attention=configs['attention'], nclass=3)
    if args.model_name.endswith('.pkl'):
        model = torch.load(os.path.join(args.model_path, args.model_name))
    else:
        checkpoint = torch.load(os.path.join(args.model_path, args.model_name))['model_state_dict']
        model.load_state_dict(
            {k.replace('module.', ''): v for k, v in checkpoint.items()})
    model.cuda()

    if args.test_op == 'case':
        testF = open(os.path.join(args.model_path, args.model_name + '_case.csv'), 'w')
        testF.write('Case,Kidney_Dice,Tumor_Dice\n ')

        dice_sum = [0.0, 0.0]

        print('starting no crop...')
        for i in range(args.start_id, args.end_id + 1):
            if args.sliding_window:
                # pred = predict_volume_slide_window(model, configs, args.model_path, i, suffix=args.data_suffix)
                pred = predict_volume_sw(model, configs, args.model_path, i, patch_size=(160, 160, 64),
                                         strides=(20, 20, 10), suffix=args.data_suffix, mirror=True)
                print("Using sliding window method...")
            else:
                pred = predict_volume(model, configs, args.model_path, i, args.data_suffix)
                print("Using no sliding window method...")
            if args.data_suffix == 'npy':
                seg_data = Kits2019DataLoader3D.load_patient(os.path.join(
                    '/home/data_share/npy_data/', str(i).zfill(5)))[0][1]

            else:
                seg_data = load_segmentation(i).get_fdata()

            assert len(np.unique(seg_data)) == 3
            dice = evaluate_dice(pred, seg_data)

            case_id = str(i).zfill(5)

            testF.write('{},{},{}\n'.format(case_id, dice[0], dice[1]))
            testF.flush()

            dice_sum = list_add(dice_sum, dice)

        # else:
        #     print('starting crop...')
        #     for i in range(args.start_id, args.end_id + 1):
        #         volume, seg = load_case(i)
        #         volume = volume.get_data().astype(np.float)
        #         seg = seg.get_data()
        #         volume, seg = crop_non_label_slice(volume, seg)
        #         volume, seg = crop_as_appointed_size(volume, seg)
        #         assert len(np.unique(seg)) == 3
        #         volume = volume[None, None, :, :, :]
        #         pred = predict_patch(model, configs, volume)
        #         dice = evaluate_dice(pred, seg)
        #
        #         case_id = str(i).zfill(5)
        #
        #         testF.write('{},{},{}\n'.format(case_id, dice[0], dice[1]))
        #         testF.flush()
        #
        #         dice_sum = list_add(dice_sum, dice)

        tk_avg = dice_sum[0] / (args.end_id - args.start_id + 1)
        tu_avg = dice_sum[1] / (args.end_id - args.start_id + 1)
        testF.write('{},{},{}'.format('avg', tk_avg, tu_avg))
        testF.flush()

    else:
        testF = open(os.path.join(args.model_path,
                                  args.model_name + '_patch.csv'), 'w')
        testF.write('Case,Kidney_Dice,Tumor_Dice\n ')
        dice_sum = [0.0, 0.0]
        files = os.listdir(args.test_path)
        files.sort()

        for file in files:
            data_path = os.path.join(args.test_path, file, 'none')
            patches = os.listdir(data_path)
            patches.sort()
            case_dice = [0.0, 0.0]
            case_imgs, case_segs = [], []
            for patch in patches:
                if patch.startswith("img"):
                    temp_img = nib.load(os.path.join(data_path, patch))
                    case_imgs.append(temp_img.get_data()[
                                     None, None, :, :, :].astype(np.float))

                elif patch.startswith("seg"):
                    temp_img = nib.load(os.path.join(data_path, patch))
                    case_segs.append(temp_img.get_data().astype(np.int))

                else:
                    pass

            for i, patch in enumerate(case_imgs):
                pred = predict_patch(model, configs, patch)
                print(pred)
                temp_dice = evaluate_dice(pred, case_segs[i])
                case_dice = list_add(case_dice, temp_dice)

            case_dice = [c / len(case_imgs) for c in case_dice]
            dice_sum = list_add(dice_sum, case_dice)
            testF.write("{},{},{}\n".format(file, case_dice[0], case_dice[1]))
            testF.flush()

        dice_avg = [c / len(files) for c in dice_sum]
        testF.write('{},{},{}'.format('avg', dice_avg[0], dice_avg[1]))
        testF.flush()

    testF.close()


if __name__ == "__main__":
    main()
