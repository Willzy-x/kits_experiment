# from local import *
import time
import argparse
import torch
from apex import amp

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
# import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import sys
import math

import shutil

import setproctitle
# choose which model to use
import Nets.vnet_ker3 as vnet
from functools import reduce
import operator

# from torch.backends import cudnn

from utils import lr_scheduler
from utils.kits2019_dataloader_3d import Kits2019DataLoader3D, get_split_deterministic, \
    get_train_transform, get_list_of_patients
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
import utils.loss as dloss
from utils1 import *
from train_and_test import train_bg, test_bg, train_bg_dice, test_bg_dice

CUDA_VISIBLE_DEVICES = 4, 5, 6, 7
torch.backends.cudnn.benchmark = True




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def datestr():
    now = time.localtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def noop(x):
    return x


def model():
    model = main()
    return model


def main():
    # --------- Parse arguments ---------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config_0.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--dice', action='store_true')
    # be aware of this argument!!!
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')

    # 1e-8 works well for lung masks but seems to prevent
    # rapid learning for nodule masks
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    # ---------- get the config file(config.yaml) --------------------------------------------
    config = get_config(args.config)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = os.path.join('./work', (datestr() + ' ' + config['filename']))
    nll = True
    if config['dice']:
        nll = False

    weight_decay = config['weight_decay']
    num_threads_for_kits19 = config['num_of_threads']
    patch_size = (160, 160, 128)
    num_batch_per_epoch = config['num_batch_per_epoch']
    setproctitle.setproctitle(args.save)
    start_epoch = 1
    # -------- Record best kidney segmentation dice -------------------------------------------
    best_tk = 0.0
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    # Embed attention module
    model = vnet.VNet(elu=False, nll=nll,
                      attention=config['attention'], nclass=3)  # mark
    batch_size = config['ngpu'] * config['batchSz']
    save_iter = config['model_save_iter']
    # batch_size = args.ngpu*args.batchSz
    gpu_ids = range(config['ngpu'])
    # print(gpu_ids)
    model.apply(weights_init)
    # ------- Resume training from saved model -----------------------------------------------
    if config['resume']:
        if os.path.isfile(config['resume']):
            print("=> loading checkpoint '{}'".format(config['resume']))
            checkpoint = torch.load(config['resume'])
            # .tar files
            if config['resume'].endswith('.tar'):
                # print(checkpoint, "tar")
                start_epoch = checkpoint['epoch']
                best_tk = checkpoint['best_tk']
                checkpoint_model = checkpoint['model_state_dict']
                model.load_state_dict(
                    {k.replace('module.', ''): v for k, v in checkpoint_model.items()})
            # .pkl files for the whole model
            else:
                # print(checkpoint, "pkl")
                model.load_state_dict(checkpoint.state_dict())
            print("=> loaded checkpoint (epoch {})".format(
                checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config['resume']))
            exit(-1)
    else:
        pass
    # ------- Which loss function to use ------------------------------------------------------
    if nll:
        training = train_bg
        validate = test_bg
        # class_balance = True
    else:
        training = train_bg_dice
        validate = test_bg_dice
        # class_balance = False
    # -----------------------------------------------------------------------------------------
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    # -------- Set on GPU ---------------------------------------------------------------------
    if args.cuda:
        model = model.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    # create the output directory
    os.makedirs(args.save)
    # save the config file to the output folder
    shutil.copy(args.config, os.path.join(args.save, 'config.yaml'))

    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # ------ Load Training and Validation set --------------------------------------------
    preprocessed_folders = "/home/data_share/npy_data/"
    patients = get_list_of_patients(
        preprocessed_data_folder=preprocessed_folders)
    # split num_split cross-validation sets
    # train, val = get_split_deterministic(
    #     patients, fold=0, num_splits=5, random_state=12345)
    train, val = patients[0:147], patients[147:189]

    # VALIDATION DATA CANNOT BE LOADED IN CASE DUE TO THE LARGE SHAPE...
    # PRINT VALIDATION CASES FOR LATER TEST USE!!
    print("Validation cases:\n", val)
    # set max shape for validation set 
    shapes = [Kits2019DataLoader3D.load_patient(
        i)[0].shape[1:] for i in val]
    max_shape = np.max(shapes, 0)
    max_shape = np.max((max_shape, patch_size), 0)
    # data loading + augmentation
    dataloader_train = Kits2019DataLoader3D(
        train, batch_size, patch_size, num_threads_for_kits19)
    dataloader_validation = Kits2019DataLoader3D(
        val, batch_size * 2, patch_size, num_threads_for_kits19)
    tr_transforms = get_train_transform(patch_size, prob=config['prob'])
    # whether to use single/multiThreadedAugmenter ------------------------------------------
    if num_threads_for_kits19 > 1:
        tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, 
                                        num_processes=num_threads_for_kits19,
                                        num_cached_per_queue=3,seeds=None, pin_memory=True)
        val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                         num_processes=max(1, num_threads_for_kits19//2), 
                                         num_cached_per_queue=1, seeds=None, pin_memory=False)
        
        tr_gen.restart()
        val_gen.restart()
    else:
        tr_gen = SingleThreadedAugmenter(dataloader_train, transform=tr_transforms)
        val_gen = SingleThreadedAugmenter(dataloader_validation, transform=None)
    # ------- Set learning rate scheduler ----------------------------------------------------
    lr_schdl = lr_scheduler.LR_Scheduler(mode=config['lr_policy'], base_lr=config['lr'],
                                         num_epochs=config['nEpochs'], iters_per_epoch=num_batch_per_epoch,
                                         lr_step=config['step_size'], warmup_epochs=config['warmup_epochs'])
    
    # ------ Choose Optimizer ----------------------------------------------------------------
    if config['opt'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                              momentum=0.99, weight_decay=weight_decay)
    elif config['opt'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    elif config['opt'] == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    lr_plateu = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True, threshold=1e-3, patience=5)
    # ------- Apex Mixed Precision Acceleration ----------------------------------------------
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = nn.parallel.DataParallel(model, device_ids=gpu_ids)
    # ------- Save training data -------------------------------------------------------------
    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    trainF.write('Epoch,Loss,Kidney_Dice,Tumor_Dice\n')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    testF.write('Epoch,Loss,Kidney_Dice,Tumor_Dice\n ')
    # ------- Training Pipeline --------------------------------------------------------------
    for epoch in range(start_epoch, config['nEpochs'] + start_epoch):
        torch.cuda.empty_cache()
        training(args, epoch, model, tr_gen, optimizer, trainF, config, lr_schdl)
        torch.cuda.empty_cache()
        print('==>lr decay to:', optimizer.param_groups[0]['lr'])
        print('testing validation set...')
        composite_dice = validate(args, epoch, model, val_gen, optimizer, testF, config, lr_plateu)
        torch.cuda.empty_cache()
        # save model with best result and routinely
        if composite_dice > best_tk or epoch % config['model_save_iter'] == 0:
            # model_name = 'vnet_epoch_step1_' + str(epoch) + '.pkl'
            model_name = 'vnet_step1_' + str(epoch) + '.tar'
            # torch.save(model, os.path.join(args.save, model_name))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_tk': best_tk
            }, os.path.join(args.save, model_name))
            best_tk = composite_dice
    # ----------------------------------------------------------------------------------------
    trainF.close()
    testF.close()


# ------------- One epoch of training process -------------------------------------------------
if __name__ == '__main__':
    main()
