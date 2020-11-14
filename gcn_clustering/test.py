###################################################################
# File Name: test.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Modified by: Marius Bock
# mail: marius.bock@protonmail.com
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader

from gcn_clustering.utils.logging import Logger
from . import model
from .feeder.feeder import Feeder, TestFeeder
from .utils import to_numpy
from .utils.meters import AverageMeter
from .utils.serialization import load_checkpoint
from .utils.utils import bcubed
from .utils.graph import graph_propagation

from sklearn.metrics import normalized_mutual_info_score, precision_score, recall_score


def val_main(state_dict, args, input_channels, features, knn_graph, labels, bbox_size_idx=None, removed=True):
    """
    Main method for running validation.

    Args:
        state_dict -- state_dict used to initialise network; if None network is trained from scratch
        args -- validation arguments (see misc.py)
        input_channels -- input channels used for validation (needed for initial compression layer that converts
                          input features into 512 feature dimension
        features -- features numpy array to be used for validation
        knn_graph -- knn graph numpy array to be used for validation
        labels  -- labels numpy array to be used for validation
        bbox_size_idx -- index of absolute size of bounding box column
        removed -- whether to remove singleton clusters from final prediction

    Returns:
        If removed = True, returns final prediction, indices for filtering out singletons, validation losses,
        average validation losses and edges and scores arrays
        If removed = False, returns final prediction, validation losses, average validation losses and
        edges and scores arrays
    """
    # same settings as training
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    cudnn.benchmark = True

    if args.use_checkpoint:
        sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))

    # initiate feeder but with training, false meaning that unique_nodes_list are also returned per instance
    valset = Feeder(features,
                    knn_graph,
                    labels,
                    args.seed,
                    args.absolute_differences,
                    args.normalise_distances,
                    args.element_wise_products_feeder,
                    args.element_wise_products_type,
                    args.k_at_hop,
                    args.active_connection,
                    train=False)
    # DataLoader for validation set
    valloader = DataLoader(
        valset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=False, pin_memory=True)

    # changed this to save space on server (state_dict needs to be passed to test_main method)
    if args.use_checkpoint:
        ckpt = load_checkpoint(args.checkpoint)
        net = model.gcn(args)
        net.load_state_dict(ckpt['state_dict'])
        net = net.cuda()
    else:
        net = model.gcn(input_channels)
        net.load_state_dict(state_dict)
    # .cuda() copies CPU data to GPU. You probably don't want to keep the data in GPU all the time.
    # That means, you only store data in GPU when it's really necessary.
    net = net.to(args.gpu)

    # initialise knn-graph
    knn_graph = valset.knn_graph
    # creates a dict where for each node its 200 NN are made a list with [] as content e.g. {1543: [], 6053: []}
    knn_graph_dict = list()
    for neighbors in knn_graph:
        knn_graph_dict.append(dict())
        for n in neighbors[1:]:
            knn_graph_dict[-1][n] = []
    # define criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # obtain edges and corresponding scores for network
    edges, scores, val_losses, val_avg_losses = validate(valloader, net, criterion, args, bbox_size_idx)

    # create clusters using pseudo label propagation
    clusters = graph_propagation(edges, scores, max_sz=900, step=0.6, pool='avg')
    # create final prediction by translating clusters to labels
    final_pred = clusters2labels(clusters, len(valset))
    # obtain labels from validation set
    labels = valset.labels

    print('------------------------------------')
    print('Number of nodes: ', len(labels))
    print('Precision   Recall   F-Sore   NMI')
    p, r, f = bcubed(labels, final_pred)
    nmi = normalized_mutual_info_score(final_pred, labels)
    print(('{:.4f}    ' * 4).format(p, r, f, nmi))
    if removed:
        # remove single clusters
        labels, final_pred_removed, remain_idcs = single_remove(labels, final_pred)
        print('------------------------------------')
        print('After removing singleton culsters, number of nodes: ', len(labels))
        print('Precision   Recall   F-Sore   NMI')
        p, r, f = bcubed(labels, final_pred_removed)
        nmi = normalized_mutual_info_score(final_pred_removed, labels)
        print(('{:.4f}    ' * 4).format(p, r, f, nmi))

        return final_pred, remain_idcs, val_losses, val_avg_losses, edges, scores
    else:
        return final_pred, val_losses, val_avg_losses, edges, scores


def test_main(state_dict, args, input_channels, features, knn_graph, bbox_size_idx=None):
    """
    Main method for running testing.

    Args:
        state_dict -- state_dict used to initialise network; if None network is trained from scratch
        args -- testing arguments (see misc.py)
        input_channels -- input channels used for testing (needed for initial compression layer that converts
                          input features into 512 feature dimension
        features -- features numpy array to be used for testing
        knn_graph -- knn graph numpy array to be used for testing
        bbox_size_idx -- index of absolute size of bounding box column

    Returns:
        Final prediction, edges and scores arrays
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    if args.use_checkpoint:
        sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))

    # initiate test feeder object
    testset = TestFeeder(features,
                         knn_graph,
                         args.seed,
                         args.absolute_differences,
                         args.normalise_distances,
                         args.element_wise_products_feeder,
                         args.element_wise_products_type,
                         args.k_at_hop,
                         args.active_connection
                         )
    # DataLoader for test dataset
    testloader = DataLoader(testset,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            shuffle=False,
                            pin_memory=True)

    # changed this to save space on server (state_dict needs to be passed to test_main method)
    if args.use_checkpoint:
        ckpt = load_checkpoint(args.checkpoint)
        net = model.gcn(args)
        net.load_state_dict(ckpt['state_dict'])
        net = net.cuda()
    else:
        net = model.gcn(input_channels)
        net.load_state_dict(state_dict)
    # .cuda() copies CPU data to GPU. You probably don't want to keep the data in GPU all the time.
    # That means, you only store data in GPU when it's really necessary.
    net = net.to(args.gpu)

    # initialise knn-graph
    knn_graph = testset.knn_graph
    # creates a dict where for each node its 200 NN are made a list with [] as content e.g. {1543: [], 6053: []}
    knn_graph_dict = list()
    for neighbors in knn_graph:
        knn_graph_dict.append(dict())
        for n in neighbors[1:]:
            knn_graph_dict[-1][n] = []

    # obtain edges and corresponding scores for network
    edges, scores = test(testloader, net, args, bbox_size_idx)

    # create clusters using pseudo label propagation
    clusters = graph_propagation(edges, scores, max_sz=900, step=0.6, pool='avg')
    # create final prediction by translating clusters to labels
    final_pred = clusters2labels(clusters, len(testset))

    return final_pred, edges, scores


def validate(loader, net, crit, args, bbox_size_idx):
    """
    Function to validate on dataset using trained network and criterion.

    Args:
        loader -- DataLoader object
        net -- GCN network object
        crit -- criterion object used as objective for optimization (e.g. Cross Entropy Loss)
        args -- validation arguments (see create_test_args)
        bbox_size_idx -- bounding box size index (needed for proper handling in batch)

    Returns:
        Predicted edges and scores as well as validation and average validation losses
    """
    # initialize timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    # set network to eval mode
    net.eval()
    end = time.time()
    # create lists for edges and scores
    edges = list()
    scores = list()
    val_losses = []
    val_avg_losses = []
    # loop over all batches
    for i, ((feat, adj, pivot_ids, h1id, node_list, _), gtmat) in enumerate(loader):
        data_time.update(time.time() - end)
        # create variables for batch
        feat, adj, pivot_ids, h1id, gtmat, _ = map(lambda x: x.to(args.gpu), (feat, adj, pivot_ids, h1id, gtmat, _))

        # rescale bbox size feature by batch if specified
        if bbox_size_idx is not None:
            feat[:, bbox_size_idx] = (feat[:, bbox_size_idx] - feat[:, bbox_size_idx].min()) / \
                                     (feat[:, bbox_size_idx].max() - feat[:, bbox_size_idx].min())

        # use network to obtain predicted link likelihoods
        pred = net(feat, adj, h1id, args)
        # obtain true labels
        labels = make_labels(gtmat).long()
        # compute loss
        loss = crit(pred, labels)
        val_losses.append(loss.detach().item())
        # compute probabilites by applying softmax; somehow not needed I think??
        pred = F.softmax(pred, dim=1)
        # compute accuracy, recall, and precision as in training
        p, r, acc = accuracy(pred, labels)

        # update times and values
        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))

        val_avg_losses.append(losses.avg)

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 or i + 1 == len(loader):
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\n'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.
                  format(i + 1, len(loader), batch_time=batch_time,
                         data_time=data_time, losses=losses, accs=accs,
                         precisions=precisions, recalls=recalls))

        # convert unique nodes list of h-hop neighborhood to numpy array
        node_list = node_list.long().squeeze().numpy()
        # returns batch size i.e. size of feature 0 axis
        batch_size = feat.size(0)
        for b in range(batch_size):
            pivot_id_b = pivot_ids[b].int().item()
            # get neighborhood of node
            if batch_size == 1:
                unique_nodes_b = node_list
            else:
                unique_nodes_b = node_list[b]
            for j, n in enumerate(h1id[b]):
                n = n.item()
                # append all edges of pivot to 1-hop neighbor as [a,b]
                edges.append([unique_nodes_b[pivot_id_b], unique_nodes_b[n]])
                # append all scores of pivot to 1-hop neighbor
                scores.append(pred[b * args.k_at_hop[0] + j, 1].item())
    # convert both edges and scores to arrays
    edges = np.asarray(edges)
    scores = np.asarray(scores)
    return edges, scores, val_losses, val_avg_losses


def test(loader, net, args, bbox_size_idx):
    """
    Function to test on dataset using trained network. Does not return loss as no true labels known.

    Args:
        loader -- DataLoader object
        net -- GCN network object
        args -- validation arguments (see create_test_args)
        bbox_size_idx -- bounding box size index (needed for proper handling in batch)

    Returns:
        Predicted edges and scores.
    """
    # initialize timers
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # set network to eval mode
    net.eval()
    end = time.time()
    # create lists for edges and scores
    edges = list()
    scores = list()
    # loop over all batches
    for i, (feat, adj, pivot_ids, h1id, node_list, _) in enumerate(loader):
        data_time.update(time.time() - end)
        # create variables for batch
        feat, adj, pivot_ids, h1id = map(lambda x: x.to(args.gpu), (feat, adj, pivot_ids, h1id))

        # rescale bbox size feature by batch if specified
        if bbox_size_idx is not None:
            feat[:, bbox_size_idx] = (feat[:, bbox_size_idx] - feat[:, bbox_size_idx].min()) / \
                                     (feat[:, bbox_size_idx].max() - feat[:, bbox_size_idx].min())

        # use network to obtain predicted link likelihoods
        pred = net(feat, adj, h1id, args)

        # compute probabilites by applying softmax; somehow not needed I think??
        pred = F.softmax(pred, dim=1)

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 or i + 1 == len(loader):
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  .format(i + 1, len(loader), batch_time=batch_time, data_time=data_time))

        # convert unique nodes list of h-hop neighborhood to numpy array
        node_list = node_list.long().squeeze().numpy()
        # returns batch size i.e. size of feature 0 axis
        batch_size = feat.size(0)
        for b in range(batch_size):
            pivot_id_b = pivot_ids[b].int().item()
            # get neighborhood of node
            if batch_size == 1:
                unique_nodes_b = node_list
            else:
                unique_nodes_b = node_list[b]
            for j, n in enumerate(h1id[b]):
                n = n.item()
                # append all edges of pivot to 1-hop neighbor as [a,b]
                edges.append([unique_nodes_b[pivot_id_b], unique_nodes_b[n]])
                # append all scores of pivot to 1-hop neighbor
                scores.append(pred[b * args.k_at_hop[0] + j, 1].item())
    # convert both edges and scores to arrays
    edges = np.asarray(edges)
    scores = np.asarray(scores)
    return edges, scores


def handle_intermediate_network(state_dict, args, input_channels, features, knn_graph, labels, bbox_size_idx=None):
    """
    Function used to obtain feature maps after second classifier layer during ensemble GCN setup.

    Args:
        state_dict --state_dict used to initialise network; if None network is trained from scratch
        args -- testing/ validation arguments (see misc.py)
        input_channels -- input channels used for testing (needed for initial compression layer that converts
                          input features into 512 feature dimension
        features -- features numpy array to be used
        knn_graph -- knn graph numpy array to be used
        labels -- labels numpy array to be used
        bbox_size_idx -- index of absolute size of bounding box column

    Returns:
        Feature maps after second classifier layer for each detection in dataset
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    if args.use_checkpoint:
        sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))

    # initiate feeder but with training, false meaning that unique_nodes_list are also returned per instance
    valset = Feeder(features,
                    knn_graph,
                    labels,
                    args.seed,
                    args.absolute_differences,
                    args.normalise_distances,
                    args.element_wise_products_feeder,
                    args.element_wise_products_type,
                    args.k_at_hop,
                    args.active_connection,
                    train=False)
    # DataLoader for validation set
    valloader = DataLoader(
        valset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=False, pin_memory=True)

    # changed this to save space on server (state_dict needs to be passed to test_main method)
    if args.use_checkpoint:
        ckpt = load_checkpoint(args.checkpoint)
        net = model.gcn_intermediate(input_channels)
        net.load_state_dict(ckpt['state_dict'])
        net = net.cuda()
    else:
        net = model.gcn_intermediate(input_channels)
        net.load_state_dict(state_dict)
    # .cuda() copies CPU data to GPU. You probably don't want to keep the data in GPU all the time.
    # That means, you only store data in GPU when it's really necessary.
    net = net.to(args.gpu)

    # initialise knn-graph
    knn_graph = valset.knn_graph
    # creates a dict where for each node its 200 NN are made a list with [] as content e.g. {1543: [], 6053: []}
    knn_graph_dict = list()
    for neighbors in knn_graph:
        knn_graph_dict.append(dict())
        for n in neighbors[1:]:
            knn_graph_dict[-1][n] = []

    # obtain edges and corresponding scores for network
    feature_maps = obtain_feature_map(valloader, net, len(features), 256, args, bbox_size_idx)

    return feature_maps


def obtain_512_feature_map(state_dict, args, input_channels, features, knn_graph, labels, bbox_size_idx=None):
    """
    Function used to obtain 512 feature maps after last graph convolution layer of a GCN network.
    Used for visualization.

    Args:
        state_dict --state_dict used to initialise network; if None network is trained from scratch
        args -- testing/ validation arguments (see misc.py)
        input_channels -- input channels used for testing (needed for initial compression layer that converts
                          input features into 512 feature dimension
        features -- features numpy array to be used
        knn_graph -- knn graph numpy array to be used
        labels -- labels numpy array to be used
        bbox_size_idx -- index of absolute size of bounding box column

    Returns:
        512 feature maps after last graph convolution layer for each detection in dataset
    """

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    if args.use_checkpoint:
        sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))

    # initiate feeder but with training, false meaning that unique_nodes_list are also returned per instance
    valset = Feeder(features,
                    knn_graph,
                    labels,
                    args.seed,
                    args.absolute_differences,
                    args.normalise_distances,
                    args.element_wise_products_feeder,
                    args.element_wise_products_type,
                    args.k_at_hop,
                    args.active_connection,
                    train=False)
    # DataLoader for validation set
    valloader = DataLoader(
        valset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=False, pin_memory=True)

    # changed this to save space on server (state_dict needs to be passed to test_main method)
    if args.use_checkpoint:
        ckpt = load_checkpoint(args.checkpoint)
        net = model.gcn_feature_map(input_channels)
        net.load_state_dict(ckpt['state_dict'])
        net = net.cuda()
    else:
        net = model.gcn_feature_map(input_channels)
        net.load_state_dict(state_dict)
    # .cuda() copies CPU data to GPU. You probably don't want to keep the data in GPU all the time.
    # That means, you only store data in GPU when it's really necessary.
    net = net.to(args.gpu)

    # initialise knn-graph
    knn_graph = valset.knn_graph
    # creates a dict where for each node its 200 NN are made a list with [] as content e.g. {1543: [], 6053: []}
    knn_graph_dict = list()
    for neighbors in knn_graph:
        knn_graph_dict.append(dict())
        for n in neighbors[1:]:
            knn_graph_dict[-1][n] = []

    # obtain edges and corresponding scores for network
    feature_maps = obtain_feature_map(valloader, net, len(features), 512, args, bbox_size_idx)

    return feature_maps


def obtain_feature_map(loader, net, num_samples, out_dim, args, bbox_size_idx):
    """
    Function used to obtain a feature map of a given GCN.

    Args:
        loader -- DataLoader object
        net -- GCN network to obtain feature maps from
        num_samples -- number of samples in dataset
        out_dim -- output dimension of feature maps (num. of output channels)
        args -- testing/ validation arguments (see misc.py)
        bbox_size_idx -- index of absolute size of bounding box column

    Returns:
        Feature map array
    """
    # set network to eval mode
    net.eval()
    output_features = np.empty((num_samples, out_dim))
    # loop over all batches
    for i, ((feat, adj, pivot_ids, h1id, node_list, indeces), gtmat) in enumerate(loader):
        # create variables for batch
        feat, adj, pivot_ids, h1id, gtmat, indeces = map(lambda x: x.to(args.gpu),
                                                         (feat, adj, pivot_ids, h1id, gtmat, indeces))

        # rescale bbox size feature by batch if specified
        if bbox_size_idx is not None:
            feat[:, bbox_size_idx] = (feat[:, bbox_size_idx] - feat[:, bbox_size_idx].min()) / \
                                     (feat[:, bbox_size_idx].max() - feat[:, bbox_size_idx].min())

        # use network to obtain predicted link likelihoods
        feature_maps = net(feat, adj, h1id, args).detach().cpu()
        for j, indeces in enumerate(indeces):
            output_features[indeces, :] = feature_maps[j, :]

    return output_features


def single_remove(Y, pred):
    """
    Function that removes all singleton clusters from prediction output file.

    Args:
        Y -- True labels as array
        pred -- predicted labels as array

    Returns:
        Y and pred without singleton clusters as well as indices of remaining indices
    """
    # creates array of zeros same size as pred
    single_idcs = np.zeros_like(pred)
    # obtain unique labels of prediction
    pred_unique = np.unique(pred)
    for u in pred_unique:
        # iterate over all labels of prediction
        idcs = pred == u
        # if there is only one that was set to true then it is a single idc
        if np.sum(idcs) == 1:
            # np.where(idcs)[0][0] return index of node that is single one
            single_idcs[np.where(idcs)[0][0]] = 1
    # use single idcs array to create array of all remaining idcs (has idcs of all nodes not in single_idcs
    remain_idcs = [i for i in range(len(pred)) if not single_idcs[i]]
    remain_idcs = np.asarray(remain_idcs)
    # return labels and pred only of indexes that are in remain_idcs and return indeces
    return Y[remain_idcs], pred[remain_idcs], remain_idcs


def clusters2labels(clusters, n_nodes):
    """
    Method that takes clusters and number of nodes and creates a labeling.

    Args:
        clusters -- predicted clusters
        n_nodes -- number of nodes in graph

    Returns:
        Labeling for each instance according to clustering.
    """
    # labels array of size n_nodes; -1 for nodes not belonging to clusters
    labels = (-1) * np.ones((n_nodes,))
    # ci = label, c = cluster, xid = node in cluster
    for ci, c in enumerate(clusters):
        for xid in c:
            # set for id of node its cluster label
            labels[xid.name] = ci
    assert np.sum(labels < 0) < 1
    return labels


def make_labels(gtmat):
    """
    Function to create label element
    """
    return gtmat.view(-1)


def accuracy(pred, label):
    """
    Function that calculates accuracy, precision and recall for given predictions and true labels.

    Args:
        pred -- prediction array
        label -- true labels array

    Returns:
        Precision, recall and accuracy measures
    """
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p, r, acc
