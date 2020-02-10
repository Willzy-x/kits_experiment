import torch
from apex import amp

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss

# ------- Loss function from nnunet ------------------------------------------------------
try:
    dice_args = {"batch_dice": False, "do_bg": True, "smooth": 1.,
                 "square": False}
    ce_args = {"weight": None}
    dice_and_ce_loss = DC_and_CE_loss(dice_args, ce_args)
except NameError:
    pass

from functools import reduce
from utils import lr_scheduler
import utils.loss as dloss
from utils.adversarial_attacks import FGSM_Attack
from utils1 import *


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
        output = output.view(output.numel() // 3, 3)  # 3 labels
        output = F.log_softmax(output, dim=-1)  # dim marked

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

        nProcessed += len(data)

        # get the index of the max log-probability
        pred = torch.argmax(output, dim=-1)
        print(output.size(), pred.size(), target.size())
        dice = evaluate_dice(pred, target, cpu=True)

        incorrect = pred.ne(target.data).cpu().sum()
        partialEpoch = (int)(epoch + batch_idx / len(trainLoader))
        loss_data = loss.detach().data.cpu().numpy()
        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice:{:.6}'.format(
            partialEpoch, nProcessed, nTrain, 100. *
                                              batch_idx / len(trainLoader),
            loss_data, dice[0], dice[1]))
        #
        trainF.write('{},{},{},{}\n'.format(
            partialEpoch, loss_data, dice[0], dice[1]))
        trainF.flush()


# -------------- adding 2 lists with the same length-------------------------------------------
def list_add(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return c


# ------------- One epoch of test process -----------------------------------------------------
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
            data, target = Variable(
                data, requires_grad=False), Variable(target)
            output = model(data)

            output = output.permute(0, 2, 3, 4, 1).contiguous()
            output = output.view(output.numel() // 3, 3)  # 2 labels
            output = F.log_softmax(output, dim=-1)  # dim?

            target = target.view(target.numel())

            test_loss += F.nll_loss(output, target).detach().data.cpu().numpy()

            # get the index of the max log-probability
            pred = torch.argmax(output, dim=-1)

            d = evaluate_dice(pred, target, cpu=True)
            dice[0] += d[0]
            dice[1] += d[1]

    # loss function already averages over batch size
    test_loss /= len(testLoader)
    dice = [c / len(testLoader) for c in dice]  # average dice on every sample
    print('\nTest set: Average loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice: {:.6f}\n'.format(
        test_loss, dice[0], dice[1]))
    testF.write('{},{},{},{}\n'.format(epoch, test_loss, dice[0], dice[1]))
    testF.flush()
    return dice[0] + dice[1]


# ------------- One epoch of training process -------------------------------------------------
def train_dice(args, epoch, model, trainLoader, optimizer, trainF, config, scheduler):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    # nIter_per_epoch = nTrain // batch_size
    dice_loss = dloss.DiceLoss(nclass=3)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # output is already the result of softmax function
        # may add CrossEntropy here
        output = model(data)

        # tar_numeled = False
        # -------------- Process multi supervision in a model -----------------------------------------
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
                out = out.view(out.numel() // 3, 3)
                loss2.append(F.cross_entropy(out, tar_numeled))

            loss = reduce(lambda x, y: x + y, loss1)
            loss += reduce(lambda x, y: x + y, loss2)

            target = target[:, 0, :, :, :]
            pred = torch.argmax(output[-1], dim=1)

            assert len(target.size()) == len(pred.size())

        else:
            output = F.softmax(output, dim=1)
            loss = dice_loss(output, target)

            # output = output.permute(0, 2, 3, 4, 1).contiguous()
            # output = output.view(output.numel()//3, 3)
            # target = target.view(target.numel())
            target = target.view(target.size(0), target.size(
                2), target.size(3), target.size(4))
            # loss += F.cross_entropy(output, target)

            # get the index of the max log-probability
            pred = torch.argmax(output, dim=1)
        # ----------- apex mixed precision ------------------------------------------------
        # loss.backward() becomes:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        # update learning rate
        scheduler(optimizer, i=batch_idx, epoch=epoch)
        nProcessed += len(data)
        # print(output.size(), pred.size(), target.size())
        dice = evaluate_dice(pred, target, cpu=True)

        incorrect = pred.ne(target.data).cpu().sum()
        partialEpoch = (int)(epoch + batch_idx / len(trainLoader))
        loss_data = loss.detach().data.cpu().numpy()
        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. *
                                              batch_idx / len(trainLoader),
            loss_data, dice[0], dice[1]))
        #
        trainF.write('{},{},{},{}\n'.format(
            partialEpoch, loss_data, dice[0], dice[1]))
        trainF.flush()


# ------------- One epoch of test process -----------------------------------------------------
def test_dice(args, epoch, model, testLoader, optimizer, testF, config):
    model.eval()
    test_loss = 0
    dice = [0.0, 0.0]
    # no gradient computation
    dice_loss = dloss.DiceLoss(nclass=3)
    with torch.no_grad():
        for data, target in testLoader:
            if args.cuda:
                data, target = data.cuda(), target.type(torch.LongTensor).cuda()
            data, target = Variable(
                data, requires_grad=False), Variable(target)
            output = model(data)
            # -------------- Process multi supervision in a model -----------------------------------------
            if isinstance(output, list):
                loss1 = []
                loss2 = []
                tar_numeled = target.view(target.numel())
                tar_size = target.size()
                for i, out in enumerate(output):
                    if i < (len(output) - 1):
                        out = F.interpolate(
                            out, size=tar_size[2:5], mode='trilinear')
                    loss1.append(
                        dice_loss(out, target).detach().data.cpu().numpy())

                    out = out.permute(0, 2, 3, 4, 1).contiguous()
                    out = out.view(out.numel() // 3, 3)
                    loss2.append(F.cross_entropy(
                        out, tar_numeled).detach().data.cpu().numpy())

                test_loss += reduce(lambda x, y: x + y, loss1)
                test_loss += reduce(lambda x, y: x + y, loss2)

                target = target[:, 0, :, :, :]
                pred = torch.argmax(output[-1], dim=1)

                assert len(target.size()) == len(pred.size())

            else:
                output = F.softmax(output, dim=1)
                test_loss += dice_loss(output,
                                       target).detach().data.cpu().numpy()

                # output = output.permute(0, 2, 3, 4, 1).contiguous()
                # output = output.view(output.numel()//3, 3)
                # target = target.view(target.numel())
                target = target.view(target.size(0), target.size(
                    2), target.size(3), target.size(4))

                # test_loss += F.cross_entropy(output,
                #                             target).detach().data.cpu().numpy()

                # get the index of the max probability
                pred = torch.argmax(output, dim=1)

            d = evaluate_dice(pred, target, cpu=True)
            dice[0] += d[0]
            dice[1] += d[1]

    # loss function already averages over batch size
    test_loss /= len(testLoader)
    dice = [c / len(testLoader) for c in dice]  # average dice on every sample
    print('\nTest set: Average loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice: {:.6f}\n'.format(
        test_loss, dice[0], dice[1]))
    #
    testF.write('{},{},{},{}\n'.format(epoch, test_loss, dice[0], dice[1]))
    testF.flush()
    return (dice[0] + dice[1])


def train_bg(args, epoch, model, tr_gen, optimizer, trainF, config, scheduler):
    model.train()
    nProcessed = 0
    num_batch = config['num_batch_per_epoch']
    batch_count = 0
    for b in range(num_batch):
        batch = next(tr_gen)
        if args.cuda:
            data, target = torch.tensor(batch['data'], dtype=torch.float32).cuda(), \
                           torch.tensor(batch['seg'], dtype=torch.long).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        output = output.permute(0, 2, 3, 4, 1).contiguous()
        output = output.view(output.numel() // 3, 3)  # 3 labels
        output = F.log_softmax(output, dim=-1)  # dim marked

        target = target.view(target.numel())

        # add CrossEntropyLoss
        loss = F.nll_loss(output, target)

        # loss.backward() becomes:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # loss.backward()
        optimizer.step()
        # update learning rate
        scheduler(optimizer, i=batch_count, epoch=epoch)
        batch_count += 1

        pred = torch.argmax(output, dim=-1)
        # print(output.size(), pred.size(), target.size())
        dice = evaluate_dice(pred, target, cpu=True)
        partialEpoch = (int)(epoch + batch_count /
                             num_batch)
        loss_data = loss.detach().data.cpu().numpy()
        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice:{:.6}'.format(
            partialEpoch, batch_count, num_batch, 100. *
                                                  batch_count / num_batch,
            loss_data, dice[0], dice[1]))
        #
        trainF.write('{},{},{},{}\n'.format(
            partialEpoch, loss_data, dice[0], dice[1]))
        trainF.flush()


def test_bg(args, epoch, model, val_gen, optimizer, testF, config, lr_schdl):
    model.eval()
    test_loss = 0
    dice = [0.0, 0.0]
    # no gradient computation
    batch_count = 0
    num_batch = config['num_batch_per_epoch'] // 5  # 5 cross_val
    with torch.no_grad():
        for b in range(num_batch):
            batch = next(val_gen)
            # print("\nLoad batch {}, shape: {}".format(b, batch['data'].shape))
            if args.cuda:
                data, target = torch.tensor(batch['data'], dtype=torch.float32).cuda(), \
                               torch.tensor(batch['seg'], dtype=torch.long).cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)

            output = output.permute(0, 2, 3, 4, 1).contiguous()
            output = output.view(output.numel() // 3, 3)  # 3 labels
            output = F.log_softmax(output, dim=-1)  # dim marked

            target = target.view(target.numel())

            # add CrossEntropyLoss
            test_loss += F.nll_loss(output,
                                    target).detach().data.cpu().numpy()

            pred = torch.argmax(output, dim=-1)
            # print(output.size(), pred.size(), target.size())
            temp_dice = evaluate_dice(pred, target, cpu=True)
            # print("\nBatch: {}, tk_dice: {}, tu_dice: {}, completed.".format(
            # b, temp_dice[0], temp_dice[1]))
            dice[0] += temp_dice[0]
            dice[1] += temp_dice[1]

    test_loss /= num_batch

    dice = [c / num_batch for c in dice]  # average dice on every sample
    print('\nTest set: Average loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice: {:.6f}\n'.format(
        test_loss, dice[0], dice[1]))
    testF.write('{},{},{},{}\n'.format(epoch, test_loss, dice[0], dice[1]))
    testF.flush()
    lr_schdl.step(test_loss)
    return (dice[0] + dice[1])


def train_adv(args, epoch, model, trainLoader, optimizer, trainF, config, scheduler):
    model.train()
    attack = FGSM_Attack(model, F.nll_loss)
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    # nIter_per_epoch = nTrain // batch_size
    # dice_loss = dloss.DiceLoss(nclass=2)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        _, output_adv, output = attack.fgsm(data, target, softmax=F.log_softmax)

        output = output.permute(0, 2, 3, 4, 1).contiguous()
        output = output.view(output.numel() // 3, 3)  # 3 labels
        # output = F.log_softmax(output, dim=-1)  # dim marked
        output_adv = output_adv.permute(0, 2, 3, 4, 1).contiguous()
        output_adv = output_adv.view(output.numel() // 3, 3)  # 3 labels

        target = target.view(target.numel())

        # add CrossEntropyLoss
        loss = F.nll_loss(output, target)
        adv_loss = F.nll_loss(output_adv, target)

        # loss.backward() becomes:
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        loss.backward()
        adv_loss.backward()
        optimizer.step()
        # update learning rate
        scheduler(optimizer, i=batch_idx, epoch=epoch)

        nProcessed += len(data)

        # get the index of the max log-probability
        pred = torch.argmax(output, dim=-1)
        # print(output.size(), pred.size(), target.size())
        dice = evaluate_dice(pred, target, cpu=True)

        incorrect = pred.ne(target.data).cpu().sum()
        partialEpoch = (int)(epoch + batch_idx / len(trainLoader))
        loss_data = loss.detach().data.cpu().numpy()
        adv_loss_data = adv_loss.detach().data.cpu().numpy()
        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice:{:.6}, Adv_loss: {:.4f}'.format(
                partialEpoch, nProcessed, nTrain, 100. *
                                                  batch_idx / len(trainLoader),
                loss_data, dice[0], dice[1], adv_loss_data))
        #
        trainF.write('{},{},{},{},{}\n'.format(
            partialEpoch, loss_data, dice[0], dice[1], adv_loss_data))
        trainF.flush()


def train_adv(args, epoch, model, trainLoader, optimizer, trainF, config, scheduler):
    model.train()
    attack = FGSM_Attack(model, F.nll_loss)
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    # nIter_per_epoch = nTrain // batch_size
    # dice_loss = dloss.DiceLoss(nclass=2)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        _, output_adv, output = attack.fgsm(data, target, softmax=F.log_softmax)

        output = output.permute(0, 2, 3, 4, 1).contiguous()
        output = output.view(output.numel() // 3, 3)  # 3 labels
        # output = F.log_softmax(output, dim=-1)  # dim marked
        output_adv = output_adv.permute(0, 2, 3, 4, 1).contiguous()
        output_adv = output_adv.view(output.numel() // 3, 3)  # 3 labels

        target = target.view(target.numel())

        # add CrossEntropyLoss
        loss = F.nll_loss(output, target)
        adv_loss = F.nll_loss(output_adv, target)

        # loss.backward() becomes:
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        loss.backward()
        adv_loss.backward()
        optimizer.step()
        # update learning rate
        scheduler(optimizer, i=batch_idx, epoch=epoch)

        nProcessed += len(data)

        # get the index of the max log-probability
        pred = torch.argmax(output, dim=-1)
        # print(output.size(), pred.size(), target.size())
        dice = evaluate_dice(pred, target, cpu=True)

        incorrect = pred.ne(target.data).cpu().sum()
        partialEpoch = (int)(epoch + batch_idx / len(trainLoader))
        loss_data = loss.detach().data.cpu().numpy()
        adv_loss_data = adv_loss.detach().data.cpu().numpy()
        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice:{:.6}, Adv_loss: {:.4f}'.format(
                partialEpoch, nProcessed, nTrain, 100. *
                                                  batch_idx / len(trainLoader),
                loss_data, dice[0], dice[1], adv_loss_data))
        #
        trainF.write('{},{},{},{},{}\n'.format(
            partialEpoch, loss_data, dice[0], dice[1], adv_loss_data))
        trainF.flush()


def train_bg_dice(args, epoch, model, tr_gen, optimizer, trainF, config, scheduler):
    model.train()
    nProcessed = 0
    num_batch = config['num_batch_per_epoch']
    batch_count = 0
    for b in range(num_batch):
        batch = next(tr_gen)
        if args.cuda:
            data, target = torch.tensor(batch['data'], dtype=torch.float32).cuda(), \
                           torch.tensor(batch['seg'], dtype=torch.long).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # add CrossEntropyLoss And DiceLoss
        loss = dice_and_ce_loss(output, target)
        # # loss.backward() becomes:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # loss.backward()
        optimizer.step()
        # update learning rate
        scheduler(optimizer, i=batch_count, epoch=epoch)
        batch_count += 1

        pred = torch.argmax(output, dim=1)
        # print(output.size(), pred.size(), target.size())
        dice = evaluate_dice(pred, target.squeeze(), cpu=True)
        partialEpoch = (int)(epoch + batch_count /
                             num_batch)
        loss_data = loss.detach().data.cpu().numpy()
        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice:{:.6}'.format(
            partialEpoch, batch_count, num_batch, 100. *
                                                  batch_count / num_batch,
            loss_data, dice[0], dice[1]))
        #
        trainF.write('{},{},{},{}\n'.format(
            partialEpoch, loss_data, dice[0], dice[1]))
        trainF.flush()


def test_bg_dice(args, epoch, model, val_gen, optimizer, testF, config, lr_schdl):
    model.eval()
    test_loss = 0
    dice = [0.0, 0.0]
    # no gradient computation
    batch_count = 0
    num_batch = config['num_batch_per_epoch']  # 5 cross_val
    with torch.no_grad():
        for b in range(num_batch):
            batch = next(val_gen)
            # print("\nLoad batch {}, shape: {}".format(b, batch['data'].shape))
            if args.cuda:
                data, target = torch.tensor(batch['data'], dtype=torch.float32).cuda(), \
                               torch.tensor(batch['seg'], dtype=torch.long).cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)

            # target = target.view(target.numel())

            # add CrossEntropyLoss
            test_loss += dice_and_ce_loss(output,
                                          target).detach().data.cpu().numpy()

            pred = torch.argmax(output, dim=1)
            # print(output.size(), pred.size(), target.size())
            temp_dice = evaluate_dice(pred, target.squeeze(), cpu=True)
            # print("\nBatch: {}, tk_dice: {}, tu_dice: {}, completed.".format(
            # b, temp_dice[0], temp_dice[1]))
            dice[0] += temp_dice[0]
            dice[1] += temp_dice[1]

    test_loss /= num_batch

    dice = [c / num_batch for c in dice]  # average dice on every sample
    print('\nTest set: Average loss: {:.4f}, Kidney_Dice: {:.6f}, Tumor_Dice: {:.6f}\n'.format(
        test_loss, dice[0], dice[1]))
    testF.write('{},{},{},{}\n'.format(epoch, test_loss, dice[0], dice[1]))
    testF.flush()
    lr_schdl.step(test_loss)
    return dice[0] + dice[1]
