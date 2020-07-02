###################################################################
# File Name: train.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 10:08:49 PM CST
###################################################################

# Imports are done to have functionalities that are actually only available in future Python releases
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os.path as osp
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from . import model
from .feeder.feeder import Feeder
from .utils import to_numpy
from .utils.logging import Logger
from .utils.meters import AverageMeter
from .utils.serialization import save_checkpoint

from sklearn.metrics import precision_score, recall_score


def train_main(args):
    # Set start time for time measuring and seed for reproduceability
    start_time = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    """
    Enables benchmark mode in cudnn.Benchmark mode good whenever input sizes for network do not vary. 
    This way CUDNN will look for optimal set of algorithms for particular configuration (takes some time)
    Usually leads to faster runtime. But if input sizes changes at each iteration, then cudnn will benchmark 
    every time a new size appears, possibly leading to worse runtime performances
    """
    cudnn.benchmark = True

    # Call feeder that creates train data set using arguments; if instance is requested from trainset, it returns the
    # the features, adjacency matrix, center_idx, one_hop_idcs and edge_labels of the IPS; features contains h-hop
    # neigborhood
    trainset = Feeder(args.features,
                      args.knn_graph,
                      args.labels,
                      args.seed,
                      args.k_at_hop,
                      args.active_connection)
    # trainloader element to load batches of training data (bottleneck since for every batch IPS needs to be computed)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=True, pin_memory=True)
    # if statement introduced to make CPU compatible; creates GCN model

    net = model.gcn(args).to(args.gpu)
    # declare optimizer (stochastic gradient descent) with learning rate, momentum and weight_decay (l2 penalty)
    opt = torch.optim.SGD(net.parameters(), args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # if statement introduced to make CPU compatible; define loss function (CE loss)
    criterion = nn.CrossEntropyLoss().to(args.gpu)

    # save state of network after each epoch as ckpt file (see utils.serialization)
    if args.save_checkpoints:
        save_checkpoint({
            'state_dict': net.state_dict(),
            'epoch': 0, }, False,
            fpath=osp.join(args.logs_dir, 'epoch_{}.ckpt'.format(0)))

    # commence training
    for epoch in range(args.epochs):
        # print elapsed time until now
        print("Elapsed time: " + str(time.time() - start_time))
        # adjust learning rate at each new epoch
        adjust_lr(opt, epoch, args)

        # train model using parameters trainloader, network, criterion, optimizer and epoch number
        train(trainloader, net, criterion, opt, epoch, args)
        # after each epoch save state into file (is_best is set to false here)
        if args.save_checkpoints and epoch + 1 == 4:
            save_checkpoint({
                'state_dict': net.state_dict(),
                'epoch': epoch + 1, }, False,
                fpath=osp.join(args.logs_dir, 'epoch_{}.ckpt'.format(epoch + 1)))
    # print final elapsed time
    print("Final elapsed time: " + str(time.time() - start_time))

    return net.state_dict()


def train(loader, net, crit, opt, epoch, args):
    # average meters used to compute average time for certain sections of application (see utils meters)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    # set module in train mode; Sets the module in training mode.
    # this has any effect only on certain modules.
    net.train()
    end = time.time()
    for i, ((feat, adj, cid, h1id), gtmat) in enumerate(loader):
        data_time.update(time.time() - end)
        # if statement for CPU-compability; iterate over all samples contained in batch to create one s
        feat, adj, cid, h1id, gtmat = map(lambda x: x.to(args.gpu), (feat, adj, cid, h1id, gtmat))

        # call neural net using batch variables
        pred = net(feat, adj, h1id, args)
        # create labels array from gtmat (true edge labels)
        labels = make_labels(gtmat).long()
        # compute loss, prcision, recall and accuracy
        loss = crit(pred, labels)
        p, r, acc = accuracy(pred, labels)

        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer. It’s important to call this
        # before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
        opt.zero_grad()
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
        # These are accumulated into x.grad for every parameter x.
        loss.backward()
        # optimizer.step updates the value of x using the gradient x.grad
        opt.step()

        # update time needed for each compuation (see utils meters)
        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))

        # update batch_time and reset end time for new batch
        batch_time.update(time.time() - end)
        end = time.time()
        # print results
        if i % args.print_freq == 0 or i+1 == len(loader):
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                   epoch+1, i+1, len(loader), batch_time=batch_time,
                   data_time=data_time, losses=losses, accs=accs,
                   precisions=precisions, recalls=recalls))


def make_labels(gtmat):
    """
    Function to create labels array used for comparing predictions and true labels
    """
    return gtmat.view(-1)


def adjust_lr(opt, epoch, args):
    """
    Function to adjust learning rate inbetween. For first 4 epoch decreases the learning rate by factor of 10 and update
    optimizer's parameters accordingly
    """
    scale = 0.1
    print('Current lr {}'.format(args.lr))
    if epoch in [1, 2, 3, 4]:
        args.lr *= 0.1
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale


def accuracy(pred, label):
    """
    Function that computes accuracy, precision and recall
    """
    # argmax returns index of max element so
    # prediciton contains prob that is not edge (0 index) and is edge (1 index)
    # so if applied and 1 index is larger than 0 index it automatically declares edge
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p, r, acc


if __name__ == '__main__':
    timestamp = time.strftime("%labels%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs', timestamp))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=24, type=int)
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    parser.add_argument('--save_checkpoints', default=False, type=bool)

    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=4)

    # Training args
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--feat_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../data/MOT/MOT17/MOT17-02/MOT17-02-DPM/features.pooled.npy'))
    parser.add_argument('--knn_graph_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../data/MOT/MOT17/MOT17-02/MOT17-02-DPM/knn.graph.pooled.brute.npy'))
    parser.add_argument('--label_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../data/MOT/MOT17/MOT17-02/MOT17-02-DPM/labels.zero.0.5.0.3.npy'))
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=[200, 10])
    parser.add_argument('--active_connection', type=int, default=10)

    args = parser.parse_args()

    train_main(args)
