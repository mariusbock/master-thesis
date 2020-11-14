###################################################################
# File Name: train.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Modified by: Marius Bock
# mail: marius.bock@protonmail.com
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from misc import plot_knn_graph, plot_embedding_graph
from . import model
from .feeder.feeder import Feeder
from .utils import to_numpy
from .utils.meters import AverageMeter
from .utils.osutils import mkdir_if_missing
from .utils.serialization import save_checkpoint
from .test import val_main, obtain_512_feature_map

from sklearn.metrics import precision_score, recall_score


def train_main(train_args, test_args, detector_name, run, input_channels, features, knn_graph, labels, frames,
               bbox_size_idx=None, state_dict=None):
    """
    Main method for running training.

    Args:
        train_args -- training arguments (see misc.py)
        test_args -- validation arguments (see misc.py). Needed for visualization.
        detector_name -- name of detector that is to be trained. Needed for logging.
        run -- number of current run. Needed for logging.
        input_channels -- input channels used for training (needed for initial compression layer that converts
                          input features into 512 feature dimension
        features -- features numpy array to be used for training
        knn_graph -- knn graph numpy array to be used for training
        labels  -- labels numpy array to be used for training
        frames -- frame array corresponding to frame number of each detection. Needed for visualization.
        bbox_size_idx -- index of absolute size of bounding box column
        state_dict -- state_dict of already trained GCN. If set, then network is first initialized using said
                      network. Needed if network is wanted to be trained again (sequential setup or fine-tuning).

    Returns:
        Returns state_dict of trained network, (average) losses for each network, (average) validation losses
        for each epoch.
    """
    # Set start time for time measuring and seed for reproduceability
    start_time = time.time()
    if train_args.seed is not None:
        np.random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)
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
    trainset = Feeder(features,
                      knn_graph,
                      labels,
                      train_args.seed,
                      train_args.absolute_differences,
                      train_args.normalise_distances,
                      train_args.element_wise_products_feeder,
                      train_args.element_wise_products_type,
                      train_args.k_at_hop,
                      train_args.active_connection,
                      )
    # trainloader element to load batches of training data (bottleneck since for every batch IPS needs to be computed)
    trainloader = DataLoader(trainset,
                             batch_size=train_args.batch_size,
                             num_workers=train_args.workers,
                             shuffle=True,
                             pin_memory=True
                             )
    # if statement introduced to make CPU compatible; creates GCN model
    net = model.gcn(input_channels)
    if state_dict is not None:
        net.load_state_dict(state_dict)
    net = net.to(train_args.gpu)

    # declare optimizer (stochastic gradient descent) with learning rate, momentum and weight_decay (l2 penalty)
    opt = torch.optim.SGD(net.parameters(), train_args.lr,
                          momentum=train_args.momentum,
                          weight_decay=train_args.weight_decay)
    # if statement introduced to make CPU compatible; define loss function (CE loss)
    criterion = nn.CrossEntropyLoss().to(train_args.gpu)

    # save state of network after each epoch as ckpt file (see utils.serialization)
    if train_args.save_checkpoints:
        save_checkpoint({
            'state_dict': net.state_dict(),
            'epoch': 0, }, False,
            fpath=os.path.join(train_args.logs_dir, 'epoch_{}.ckpt'.format(0)))
    epoch_losses = []
    epoch_avg_losses = []
    epoch_val_losses = []
    epoch_avg_val_losses = []
    # commence training
    for epoch in range(train_args.epochs):
        # print elapsed time until now
        print("Elapsed time: " + str(time.time() - start_time))
        # adjust learning rate at each new epoch
        adjust_lr(opt, epoch, train_args)

        # train model using parameters trainloader, network, criterion, optimizer and epoch number
        epoch_loss, epoch_avg_loss = train(trainloader, net, criterion, opt, epoch, train_args, bbox_size_idx)
        _, _, epoch_val_loss, epoch_avg_val_loss, _, _ = val_main(net.state_dict(), test_args, input_channels, features,
                                                                  knn_graph, labels, bbox_size_idx)

        print('\n CREATING 512 FEATURE MAP')
        feature_map = obtain_512_feature_map(net.state_dict(), test_args, input_channels, features, knn_graph,
                                             labels, bbox_size_idx)

        # create knn graph IPS plots
        plot_knn_graph(index=10,
                       detector_name=None,
                       run=run,
                       detector_type=detector_name + '_train_512_epoch_' + str(epoch),
                       log_directory=test_args.log_directory,
                       labels=labels,
                       features=features,
                       knn=knn_graph,
                       frames=frames,
                       k_at_hop=test_args.k_at_hop,
                       active_connection=test_args.active_connection,
                       seed=test_args.seed,
                       absolute_differences=test_args.absolute_differences,
                       normalise_distances=test_args.normalise_distances,
                       element_wise_products_feeder=test_args.element_wise_products_feeder,
                       element_wise_products_type=test_args.element_wise_products_type
                       )

        plot_knn_graph(index='half',
                       detector_name=None,
                       run=run,
                       detector_type=detector_name + '_train_512_epoch_' + str(epoch),
                       log_directory=test_args.log_directory,
                       labels=labels,
                       features=features,
                       knn=knn_graph,
                       frames=frames,
                       k_at_hop=test_args.k_at_hop,
                       active_connection=test_args.active_connection,
                       seed=test_args.seed,
                       absolute_differences=test_args.absolute_differences,
                       normalise_distances=test_args.normalise_distances,
                       element_wise_products_feeder=test_args.element_wise_products_feeder,
                       element_wise_products_type=test_args.element_wise_products_type
                       )

        # create embedding IPS plots
        plot_embedding_graph(index=10,
                             detector_name=None,
                             run=run,
                             detector_type=detector_name + '_train_512_epoch_' + str(epoch),
                             log_directory=test_args.log_directory,
                             labels=labels,
                             features=features,
                             knn=knn_graph,
                             frames=frames,
                             k_at_hop=test_args.k_at_hop,
                             active_connection=test_args.active_connection,
                             seed=test_args.seed,
                             absolute_differences=test_args.absolute_differences,
                             normalise_distances=test_args.normalise_distances,
                             element_wise_products_feeder=test_args.element_wise_products_feeder,
                             element_wise_products_type=test_args.element_wise_products_type
                             )

        plot_embedding_graph(index='half',
                             detector_name=None,
                             run=run,
                             detector_type=detector_name + '_train_512_epoch_' + str(epoch),
                             log_directory=test_args.log_directory,
                             labels=labels,
                             features=features,
                             knn=knn_graph,
                             frames=frames,
                             k_at_hop=test_args.k_at_hop,
                             active_connection=test_args.active_connection,
                             seed=test_args.seed,
                             absolute_differences=test_args.absolute_differences,
                             normalise_distances=test_args.normalise_distances,
                             element_wise_products_feeder=test_args.element_wise_products_feeder,
                             element_wise_products_type=test_args.element_wise_products_type
                             )

        if test_args.save_feature_map:
            mkdir_if_missing(os.path.join(test_args.log_directory, 'feature_maps'))
            print('\n SAVING 512 FEATURE MAP')
            np.save(os.path.join(test_args.log_directory, 'feature_maps',
                                 'train_512_epoch_' + str(epoch + 1) + '.npy'), feature_map)
        epoch_losses.append(epoch_loss)
        epoch_avg_losses.append(epoch_avg_loss)
        epoch_val_losses.append(epoch_val_loss)
        epoch_avg_val_losses.append(epoch_avg_val_loss)
        # after each epoch save state into file (is_best is set to false here)
        if train_args.save_checkpoints and epoch + 1 == 4:
            save_checkpoint({
                'state_dict': net.state_dict(),
                'epoch': epoch + 1, }, False,
                fpath=os.path.join(train_args.logs_dir, 'epoch_{}.ckpt'.format(epoch + 1)))
    # print final elapsed time
    print("Final elapsed time: " + str(time.time() - start_time))

    return net.state_dict(), epoch_losses, epoch_avg_losses, epoch_val_losses, epoch_avg_val_losses


def train(loader, net, crit, opt, epoch, args, bbox_size_idx):
    """
    Function to train a GCN network.

    Args:
        loader -- DataLoader object
        net -- GCN network object
        crit -- criterion object used as objective for optimization (e.g. Cross Entropy Loss)
        opt -- optimizer used to optimize network (e.g. Cross-Entropy Loss)
        args -- training arguments (see create_test_args)
        bbox_size_idx -- bounding box size index (needed for proper handling in batch)

    Returns:
        (Average) Training losses
    """
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
    training_losses = []
    training_avg_losses = []
    for i, ((feat, adj, cid, h1id), gtmat) in enumerate(loader):
        data_time.update(time.time() - end)
        # if statement for CPU-compability; iterate over all samples contained in batch to create one s
        feat, adj, cid, h1id, gtmat = map(lambda x: x.to(args.gpu), (feat, adj, cid, h1id, gtmat))
        if bbox_size_idx is not None:
            feat[:, bbox_size_idx] = (feat[:, bbox_size_idx] - feat[:, bbox_size_idx].min()) / \
                                     (feat[:, bbox_size_idx].max() - feat[:, bbox_size_idx].min())
        # call neural net using batch variables
        pred = net(feat, adj, h1id, args)
        # create labels array from gtmat (true edge labels)
        labels = make_labels(gtmat).long()
        # compute loss, precision, recall and accuracy
        loss = crit(pred, labels)
        training_losses.append(loss.detach().item())

        p, r, acc = accuracy(pred, labels)

        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer. It’s important to call this
        # before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
        opt.zero_grad()
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
        # These are accumulated into x.grad for every parameter x.
        loss.backward()
        # optimizer.step updates the value of x using the gradient x.grad
        opt.step()

        # update time needed for each computation (see utils meters)
        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))

        training_avg_losses.append(losses.avg)

        # update batch_time and reset end time for new batch
        batch_time.update(time.time() - end)
        end = time.time()
        # print results
        if i % args.print_freq == 0 or i + 1 == len(loader):
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                epoch + 1, i + 1, len(loader), batch_time=batch_time,
                data_time=data_time, losses=losses, accs=accs,
                precisions=precisions, recalls=recalls))
    return training_losses, training_avg_losses


def make_labels(gtmat):
    """
    Function to create labels array used for comparing predictions and true labels
    """
    return gtmat.view(-1)


def adjust_lr(opt, epoch, args):
    """
    Function to adjust learning rate inbetween epochs. For first 4 epoch decreases the learning rate by factor of 10
    and update optimizer's parameters accordingly.

    Args:
        opt -- update parameters of optimizer
        epoch -- epoch number
        args -- train arguments (see misc.py)
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
    Function that computes accuracy, precision and recall between predictions and true labels.

    Args:
        pred -- predictions
        label -- true labels

    Returns:
        Precision, recall and accuracy measures
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
