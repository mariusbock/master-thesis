###################################################################
# File Name: test.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 10:08:49 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import os.path as osp
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader

from . import model
from .feeder.feeder import Feeder
from .utils import to_numpy
from .utils.meters import AverageMeter
from .utils.serialization import load_checkpoint
from .utils.utils import bcubed
from .utils.graph import graph_propagation, graph_propagation_soft, graph_propagation_naive

from sklearn.metrics import normalized_mutual_info_score, precision_score, recall_score


def single_remove(Y, pred):
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
    # return Y and pred only of indexes that are in remain_idcs
    return Y[remain_idcs], pred[remain_idcs]


def test_main(args):
    # same settings as training
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # initiate feeder but with training, false meaning that unique_nodes_list are also returned per instance
    valset = Feeder(args.features,
                    args.knn_graph,
                    args.labels,
                    args.seed,
                    args.k_at_hop,
                    args.active_connection,
                    train=False)
    # DataLoader for validation set
    valloader = DataLoader(
        valset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=False, pin_memory=True)

    # load checkpoint and parse its settings to gcn -> so that it is the same
    ckpt = load_checkpoint(args.checkpoint)
    net = model.gcn()
    net.load_state_dict(ckpt['state_dict'])
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
    edges, scores = validate(valloader, net, criterion, args)

    # save edges and scores as files
    np.save('edges', edges)
    np.save('scores', scores)
    # edges=np.load('edges.npy')
    # scores = np.load('scores.npy')

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
    # remove single clusters
    labels, final_pred = single_remove(labels, final_pred)
    print('------------------------------------')
    print('After removing singleton culsters, number of nodes: ', len(labels))
    print('Precision   Recall   F-Sore   NMI')
    p, r, f = bcubed(final_pred, labels)
    nmi = normalized_mutual_info_score(final_pred, labels)
    print(('{:.4f}    ' * 4).format(p, r, f, nmi))


def clusters2labels(clusters, n_nodes):
    """
    Method that takes clusters and number of nodes and creates a labeling
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


def validate(loader, net, crit, args):
    """
    Function to validate on dataset using trained network and criterion
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
    # loop over all batches
    for i, ((feat, adj, pivot_ids, h1id, node_list), gtmat) in enumerate(loader):
        data_time.update(time.time() - end)
        # create variables for batch
        feat, adj, pivot_ids, h1id, gtmat = map(lambda x: x.to(args.gpu), (feat, adj, pivot_ids, h1id, gtmat))
        # use network to obtain predicted link likelihoods
        pred = net(feat, adj, h1id, args)
        # obtain true labels
        labels = make_labels(gtmat).long()
        # compute loss
        loss = crit(pred, labels)
        # compute probabilites by applying softmax; somehow not needed I think??
        pred = F.softmax(pred, dim=1)
        # compute accuracy, recall, and precision as in training
        p, r, acc = accuracy(pred, labels)

        # update times and values
        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\n'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time,
                data_time=data_time, losses=losses, accs=accs,
                precisions=precisions, recalls=recalls))

        # convert unique nodes list of h-hop neighborhood to numpy array
        node_list = node_list.long().squeeze().numpy()
        # returns batch size i.e. size of feature 0 axis
        batch_size = feat.size(0)
        for b in range(batch_size):
            pivot_id_b = pivot_ids[b].int().item()
            # get neighborhood of node
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


def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p, r, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--print_freq', default=1, type=int)

    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=[20, 5])
    parser.add_argument('--active_connection', type=int, default=5)

    # Validation args 
    parser.add_argument('--features', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../data/facedata/1845.fea.npy'))
    parser.add_argument('--knn_graph', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../data/facedata/knn.graph.1845.bf.npy'))
    parser.add_argument('--labels', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../data/facedata/1845.labels.npy'))

    # Test args
    parser.add_argument('--checkpoint', type=str, metavar='PATH', default='./logs/epoch_4.ckpt')
    args = parser.parse_args()
    test_main(args)
