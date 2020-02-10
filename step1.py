# from local import *
import time
import argparse
import torch
from apex import amp

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torchbiomed.datasets as dset
import torchbiomed.transforms as biotransforms
import torchbiomed.loss as bioloss

import os
import sys
import math

import shutil

import setproctitle
# choose which model to use
import Nets.vnet2 as vnet
from functools import reduce
import operator
# from torch.backends import cudnn

from utils import lr_scheduler
import utils.loss as dloss
from utils1 import *

# CUDA_VISIBLE_DEVICES = 4,5,6,7
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
    #--------- Parse arguments ---------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml', 
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
    # load training set and test set from the directory
    train_dir = config['train_dir']
    test_dir = config['test_dir']

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = os.path.join('./work', (datestr() + ' ' + config['filename']))
    nll = True
    if config['dice']:
        nll = False
    
    weight_decay = config['weight_decay']
    setproctitle.setproctitle(args.save)
    start_epoch = 1
    #-------- Record best kidney segmentation dice -------------------------------------------
    best_tk = 0.0
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    # Embed attention module
    model = vnet.VNet(elu=False, nll=nll, attention=config['attention']) # mark
    batch_size = config['ngpu'] * config['batchSz']
    save_iter = config['model_save_iter']
    # batch_size = args.ngpu*args.batchSz
    gpu_ids = range(config['ngpu'])
    # print(gpu_ids)
    
    #------- Resume training from saved model -----------------------------------------------
    if config['resume']:
        if os.path.isfile(config['resume']):
            print("=> loading checkpoint '{}'".format(config['resume']))
            checkpoint = torch.load(config['resume'])
            # .tar files
            if config['resume'].endswith('.tar'):
                start_epoch = checkpoint['epoch']
                best_tk = checkpoint['best_tk']
                checkpoint = checkpoint['model_state_dict']
                model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
            # .pkl files for the whole model
            else:
                model.load_state_dict(checkpoint.state_dict())    
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config['resume']))
    else:
        pass
    #------- Which loss function to use ------------------------------------------------------
    if nll:
        train = train_nll
        test = test_nll
        # class_balance = True
    else:
        train = train_dice
        test = test_dice
        # class_balance = False
    #-----------------------------------------------------------------------------------------
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    #-------- Set on GPU ---------------------------------------------------------------------
    if args.cuda:
        model = model.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    # create the output directory
    os.makedirs(args.save)
    # save the config file to the output folder
    shutil.copy(args.config, os.path.join(args.save, 'config.yaml'))

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # ------ Load Training set --------------------------------------------
    print("loading training set...")
    trainset = dset.Dataset(train_dir)
    assert len(np.unique(trainset.img_labels)) == 2
    print("Training set loaded. Length:" ,len(trainset))
    trainLoader = DataLoader(trainset, batch_size=batch_size, 
                            shuffle=True, num_workers=6, pin_memory=True)
    niter_per_epoch = len(trainset) // batch_size
    print("loading test set")
    # ------ Load Validation Set ------------------------------------------
    print("loading validation set...")
    testset = dset.Dataset(test_dir)
    assert len(np.unique(testset.img_labels)) == 2
    print("Validation set loaded. Length:" ,len(testset))
    testLoader = DataLoader(testset, batch_size=batch_size, 
                            shuffle=True, num_workers=6, pin_memory=True)
    #------- Set learning rate scheduler ----------------------------------------------------
    lr_schdl = lr_scheduler.LR_Scheduler(mode=config['lr_policy'], base_lr=config['lr'], 
        num_epochs=config['nEpochs'], iters_per_epoch=niter_per_epoch,
        lr_step=config['step_size'], warmup_epochs=0)
    # ------ Choose Optimizer ----------------------------------------------------------------
    if config['opt'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                              momentum=0.99, weight_decay=weight_decay)
    elif config['opt'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    elif config['opt'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    #-------- Apex Mixed Precision Acceleration ----------------------------------------------
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model.apply(weights_init)
    # ------- Save training data -------------------------------------------------------------
    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    trainF.write('Epoch,Loss,Kidney_Dice\n')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    testF.write('Epoch,Loss,Kidney_Dice\n ')
    # ------- Training Pipeline --------------------------------------------------------------
    for epoch in range(start_epoch, config['nEpochs'] + start_epoch):
        torch.cuda.empty_cache()
        # adjust_opt(config['opt'], optimizer, epoch, config['lr_policy'], config['lr'], config['step_size'])
        train(args, epoch, model, trainLoader, optimizer, trainF, config, lr_schdl)
        torch.cuda.empty_cache()
        print('==>lr decay to:', optimizer.param_groups[0]['lr'])
        print('testing single set...')
        tk_dice = test(args, epoch, model, testLoader, optimizer, testF, config)
        torch.cuda.empty_cache()
        # save model with best result and routinely
        if tk_dice > best_tk or epoch % config['model_save_iter'] == 0:
            # model_name = 'vnet_epoch_step1_' + str(epoch) + '.pkl'
            model_name = 'vnet_step1_' + str(epoch) + '.tar'
            # torch.save(model, os.path.join(args.save, model_name))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_tk': best_tk
            }, os.path.join(args.save, model_name))
            best_tk = tk_dice
    # ----------------------------------------------------------------------------------------
    trainF.close()
    testF.close()

#------------ Adjust learning rate -----------------------------------------------------------
def adjust_opt(optAlg, optimizer, epoch, learning_policy, learning_rate, step_size):
    """ add learning_policy option """
    if learning_policy == 'step':
        lr = learning_rate / (10**(epoch // step_size))
    else:
        lr = learning_rate * (0.985**epoch) # learning rate decay
        
    print('==>learning rate decay to', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#------------- One epoch of training process -------------------------------------------------
def train_nll(args, epoch, model, trainLoader, optimizer, trainF, config, scheduler):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    # nIter_per_epoch = nTrain // batch_size
    # dice_loss = dloss.DiceLoss(nclass=2)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)

        output = output.permute(0, 2, 3, 4, 1).contiguous()
        output = output.view(output.numel() // 2, 2) # 2 labels
        output = F.log_softmax(output, dim=-1) # dim marked
        
        target = target.view(target.numel())
        
        # add CrossEntropyLoss
        loss = F.nll_loss(output, target)

        # loss.backward() becomes:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # loss.backward()
        optimizer.step()
        # update learning rate
        scheduler(optimizer, i=batch_idx, epoch=epoch)
        print('==>lr decay to:', optimizer.param_groups[0]['lr'])

        nProcessed += len(data)
        
        pred = torch.argmax(output, dim=-1)  # get the index of the max log-probability
        print(output.size(), pred.size(), target.size())
        dice = diceIoU(pred, target, cpu=True)

        incorrect = pred.ne(target.data).cpu().sum()
        partialEpoch = (int)(epoch + batch_idx / len(trainLoader))
        loss_data = loss.detach().data.cpu().numpy()
        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Kidney_Dice: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss_data, dice[0]))
#
        trainF.write('{},{},{}\n'.format(partialEpoch,loss_data, dice[0]))
        trainF.flush()

# #-------------- adding 2 lists with the same length-------------------------------------------
# def list_add(a, b):
#     c = []
#     for i in range(len(a)):
#         c.append(a[i] + b[i])
#     return c

#------------- One epoch of test process -----------------------------------------------------
def test_nll(args, epoch, model, testLoader, optimizer, testF, config):
    model.eval()
    test_loss = 0
    dice = [0.0, 0.0]
    # no gradient computation
    # dice_loss = dloss.DiceLoss(nclass=2)
    with torch.no_grad():
        for data, target in testLoader:
            if args.cuda:
                data, target = data.cuda(), target.type(torch.LongTensor).cuda()
            data, target = Variable(data, requires_grad=False), Variable(target)
            output = model(data)
    
            output = output.permute(0, 2, 3, 4, 1).contiguous()
            output = output.view(output.numel() // 2, 2) # 2 labels
            output = F.log_softmax(output, dim=-1) # dim?
            
            target = target.view(target.numel())

            test_loss += F.nll_loss(output, target).detach().data.cpu().numpy()
            
            pred = torch.argmax(output, dim=-1)  # get the index of the max log-probability

            d = diceIoU(pred, target, cpu=True)
            dice[0] += d[0]

    test_loss /= len(testLoader)  # loss function already averages over batch size
    dice = [c/len(testLoader) for c in dice] # average dice on every sample
    print('\nTest set: Average loss: {:.4f}, Kidney_Dice: {:.6f}\n'.format(
            test_loss, dice[0]))
    testF.write('{},{},{}\n'.format(epoch, test_loss, dice[0]))
    testF.flush()
    return dice[0]


#------------- One epoch of training process -------------------------------------------------
def train_dice(args, epoch, model, trainLoader, optimizer, trainF, config, scheduler):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    # nIter_per_epoch = nTrain // batch_size
    dice_loss = dloss.DiceLoss(nclass=2)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # output is already the result of softmax function
        # may add CrossEntropy here
        output = model(data)
        
        # tar_numeled = False
#-------------- Process multi supervision in a model -----------------------------------------
        if isinstance(output, list):
            loss1 = []
            loss2 = []
            tar_size = target.size()
            tar_numeled = target.view(target.numel())
            for i, out in enumerate(output):
                if i < (len(output) - 1):
                    out = F.interpolate(
                        out, size=tar_size[2:5], mode='trilinear') 
                loss1.append(dice_loss(out, target))

                out = out.permute(0, 2, 3, 4, 1).contiguous()
                out = out.view(out.numel()//2, 2)
                loss2.append(F.cross_entropy(out, tar_numeled))
            
            loss = reduce(lambda x, y: x+y, loss1)
            loss += reduce(lambda x, y: x+y, loss2)

            target = target[:, 0, :, :, :]
            pred = torch.argmax(output[-1], dim=1)

            assert ((len(target.size()) == 4) and (len(pred.size()) == 4))
        
        else: 
            loss = dice_loss(output, target)

            output = output.permute(0, 2, 3, 4, 1).contiguous()
            output = output.view(output.numel()//2, 2)
            target = target.view(target.numel())
            loss += F.cross_entropy(output, target)

            # get the index of the max log-probability
            pred = torch.argmax(output, dim=-1)
# ----------- apex mixed precision ------------------------------------------------
        # loss.backward() becomes:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        optimizer.step()
        # update learning rate
        scheduler(optimizer, i=batch_idx, epoch=epoch)
        print('==>lr decay to:', optimizer.param_groups[0]['lr'])
        nProcessed += len(data)
        # print(output.size(), pred.size(), target.size())
        dice = diceIoU(pred, target, cpu=True)

        incorrect = pred.ne(target.data).cpu().sum()
        partialEpoch = (int)(epoch + batch_idx / len(trainLoader))
        loss_data = loss.detach().data.cpu().numpy()
        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Kidney_Dice: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss_data, dice[0]))
#
        trainF.write('{},{},{}\n'.format(partialEpoch,loss_data, dice[0]))
        trainF.flush()


#------------- One epoch of test process -----------------------------------------------------
def test_dice(args, epoch, model, testLoader, optimizer, testF, config):
    model.eval()
    test_loss = 0
    dice = [0.0, 0.0]
    # no gradient computation
    dice_loss = dloss.DiceLoss(nclass=2)
    with torch.no_grad():
        for data, target in testLoader:
            if args.cuda:
                data, target = data.cuda(), target.type(torch.LongTensor).cuda()
            data, target = Variable(
                data, requires_grad=False), Variable(target)
            output = model(data)
#-------------- Process multi supervision in a model -----------------------------------------
            if isinstance(output, list):
                loss1 = []
                loss2 = []
                tar_numeled = target.view(target.numel())
                tar_size = target.size()
                for i, out in enumerate(output):
                    if i < (len(output) - 1):
                        out = F.interpolate(out, size=tar_size[2:5], mode='trilinear')
                    loss1.append(
                        dice_loss(out, target).detach().data.cpu().numpy())

                    out = out.permute(0, 2, 3, 4, 1).contiguous()
                    out = out.view(out.numel()//2, 2)
                    loss2.append(F.cross_entropy(
                        out, tar_numeled).detach().data.cpu().numpy())

                test_loss += reduce(lambda x, y: x+y, loss1)
                test_loss += reduce(lambda x, y: x+y, loss2)

                target = target[:, 0, :, :, :]
                pred = torch.argmax(output[-1], dim=1)

                assert len(target.size()) == len(pred.size())


            else:
                test_loss += dice_loss(output,
                                        target).detach().data.cpu().numpy()
            
                output = output.permute(0, 2, 3, 4, 1).contiguous()
                output = output.view(output.numel()//2, 2)
                target = target.view(target.numel())

                test_loss += F.cross_entropy(output,
                                            target).detach().data.cpu().numpy()

                # get the index of the max probability
                pred = torch.argmax(output, dim=-1)
                
            d = diceIoU(pred, target, cpu=True)
            dice[0] += d[0]

    # loss function already averages over batch size
    test_loss /= len(testLoader)
    dice = [c/len(testLoader) for c in dice]  # average dice on every sample
    print('\nTest set: Average loss: {:.4f}, Kidney_Dice: {:.6f}\n'.format(
        test_loss, dice[0]))
#
    testF.write('{},{},{}\n'.format(epoch, test_loss, dice[0]))
    testF.flush()
    return dice[0]

if __name__ == '__main__':
    main()
