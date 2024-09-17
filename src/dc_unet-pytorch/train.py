# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:54:36 2023

@author: loua2
"""

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import argparse
from datetime import datetime

from torch.optim.lr_scheduler import LambdaLR

from dataloader import get_loader, TestDataset
from utils import AvgMeter, adjust_lr
from tqdm import tqdm
from DC_UNet import DC_Unet
from log import TrainLog


def structure_loss(pred, mask):
    # 弃用

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # 给予边界区域更大的权重
    # 使用avg_pool2d对mask进行池化操作，结果与原始mask相减，
    # 得到了边界区域的差异。然后将差异放大5倍并加1，使得边界区域的权重更大
    # shape:(minibatch,in_channels,iH,iW)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    # 对预测值pred应用sigmoid函数，将其压缩到[0, 1]区间内，从而得到概率值
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test_meanDice(model, path):
    # 计算meanDice系数
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####

    model.eval()
    image_root = '{}/images/test/'.format(data_path)
    gt_root = '{}/annotations/test/'.format(data_path)
    test_loader = TestDataset(image_root, gt_root, 256)
    b = 0.0
    # print('[test_size]',test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5 = model(image)
        res = res5
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        input = res
        target = np.array(gt)
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))

        intersection = (input_flat * target_flat)

        loss = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        a = '{:.4f}'.format(loss)
        a = float(a)
        b = b + a

    return b / test_loader.size


def confused_matrix(pred, target, num_classes):
    # 期望传入的pred,target为[h*w]
    k = (pred >= 0) & (pred < num_classes)
    matrix = torch.bincount(num_classes * target[k].to(torch.int32) + pred[k].to(torch.int32),
                            minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return matrix.numpy()


def test_mIoUs(model, image_root, gt_root, num_classes, epoch, logWriter, isTrainSet=False):
    # 计算mIoUs

    model.eval()
    test_loader = TestDataset(image_root, gt_root, 256)
    confused_matrix_sum = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in tqdm(range(test_loader.size), desc='IoU calculate progress', unit="image", ncols=120):
        # 遍历全部的测试图像
        image, target, name = test_loader.load_data()

        image = image.cuda()
        # 期望输入image.size=[1,c,h,w]
        result = model(image)
        # res = F.upsample(res, size=target.shape, mode='bilinear', align_corners=False)

        result = result.argmax(dim=1).data.cpu()
        target = target.data.cpu()
        confused_matrix_temp = confused_matrix(result.flatten(), target.flatten(), num_classes)
        # print(confused_matrix_temp)
        confused_matrix_sum += confused_matrix_temp
    # print(confused_matrix_sum)
    IoUs = (np.diag(confused_matrix_sum) /
            np.maximum(
                (confused_matrix_sum.sum(axis=1) + confused_matrix_sum.sum(axis=0) - np.diag(confused_matrix_sum)),
                torch.ones(num_classes, dtype=torch.int64)))
    mIoU = IoUs.nanmean()
    if not isTrainSet:
        logWriter.writeIoULog(epoch, IoUs, mIoU)
    print("当前IoU:{},mIoU:{}".format(IoUs, mIoU))
    return IoUs, mIoU


def train(train_loader, model, optimizer, epoch, logWriter):
    model.train()
    total_step = len(train_loader)
    # ---- multi-scale training ----
    loss_record = AvgMeter()
    with tqdm(train_loader, total=total_step, desc=f'Epoch{epoch}/{opt.epoch}', unit="batch", ncols=120) as pbar:
        for i, pack in enumerate(pbar, start=1):
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- forward ----
            lateral_map = model(images)
            # ---- loss function ----

            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.124, 0.975, 0.928, 0.9723]))
            loss = loss_fn(lateral_map, gts)

            # loss = structure_loss(lateral_map, gts)

            # ---- backward ----
            loss.backward()
            # clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            loss_record.update(loss.data, opt.batchsize)

            # ---- train visualization ----
            if i % 10 == 0 or i == total_step:
                pbar.set_postfix({
                    'Loss': f'{loss_record.show():0.4f}',  # 显示当前平均损失
                    'Step': f'{i}/{total_step}'
                })
                logWriter.writeLossLog(loss_record.show(), epoch, optimizer.param_groups[0]['lr'], i, total_step)

    # save_path = 'snapshots/{}/'.format(opt.train_save)
    # os.makedirs(save_path, exist_ok=True)
    # if (epoch + 1) % 1 == 0:
    #     meandice = test_meanDice(model, test_path)
    #
    #     fp = open('log/log.txt', 'a')
    #     fp.write(str(meandice) + '\n')
    #     fp.close()
    #
    #     fp = open('log/best.txt', 'r')
    #     best = fp.read()
    #     fp.close()
    #
    #     if meandice > float(best):
    #         fp = open('log/best.txt', 'w')
    #         fp.write(str(meandice))
    #         fp.close()
    #         # best = meandice
    #         fp = open('log/best.txt', 'r')
    #         best = fp.read()
    #         fp.close()
    #         torch.save(model.state_dict(), save_path + 'dcunet-best.pth')
    #         print('[Saving Snapshot:]', save_path + 'dcunet-best.pth', meandice, '[best:]', best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    parser.add_argument("--adjust_lr", type=bool,
                        default=False, help='choose to use learning rate adjust')

    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=256, help='training dataset size')

    parser.add_argument('--data_path', type=str,
                        default='/root/autodl-tmp/data', help='path to train dataset')
    parser.add_argument('--weight_path', type=str,
                        default='', help='the path of weight')
    parser.add_argument('--classes_number', type=int,
                        default=4, help='number of classes')

    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = DC_Unet()
    if opt.weight_path != '':
        model.load_state_dict(torch.load(opt.weight_path))
    model = model.cuda()

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(model, x)

    params = model.parameters()

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - pow((epoch / opt.epoch), 0.9))

    logWriter = TrainLog('./log/', opt)

    image_root = '{}/images/training/'.format(opt.data_path)
    gt_root = '{}/annotations/training/'.format(opt.data_path)
    test_image_root = '{}/images/test/'.format(opt.data_path)
    test_gt_root = '{}/annotations/test/'.format(opt.data_path)
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):

        # -------------------训练-----------------------
        train(train_loader, model, optimizer, epoch, logWriter)

        # -------------------测试测试集的IoU-----------------------
        test_mIoUs(model, test_image_root, test_gt_root, opt.classes_number, epoch, logWriter, isTrainSet=False)

        # -------------------测试训练集的IoU-----------------------
        if epoch % 10 == 0:
            test_mIoUs(model, image_root, gt_root, opt.classes_number, epoch, logWriter, isTrainSet=True)

        # -------------------保存权重-----------------------
        if logWriter.isBestIoUGet:
            torch.save(model.state_dict(), './state/{}.pth'.format(logWriter.generateTime))
            print("已保存最好参数!")

        # -------------------学习率调整-----------------------
        if opt.adjust_lr:
            scheduler.step()
            print("调整学习率至{}".format(optimizer.param_groups[0]['lr']))
