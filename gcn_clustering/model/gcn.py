###################################################################
# File Name: gcn.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri 07 Sep 2018 01:16:31 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
They do not provide other aggregators (Mean aggregator is the only one); missing: weighted and attention aggregation
Mean aggregation: mean aggregation performs average pooling among neighbors. Do not understand computation completely -
but result is what is reported in paper.
"""


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        """
        Performs a batch matrix-matrix product of matrices stored in input and mat2.
        input and mat2 must be 3-D tensors each containing the same number of matrices.
        If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor
        """
        x = torch.bmm(A, features)
        return x


"""
GraphConv class (declared with input dimensions and output dimensions and aggregation prediciton_type
"""
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # weight vector twice as big as input dimensions
        self.weight = nn.Parameter(
            torch.FloatTensor(in_dim * 2, out_dim))
        # bias size of output dimensions
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        """
        Fills the input Tensor with values according to the method described in Understanding the difficulty of 
        training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution
        REPLACE HERE IF OTHER INPUT WEIGHTS WANTED!
        """
        init.xavier_uniform_(self.weight)
        """
        initializes bias tensor with all 0's
        """
        init.constant_(self.bias, 0)
        self.agg = agg()

    """
    Defines computation done at every call
    """

    def forward(self, features, A):
        b, n, d = features.shape
        # test if depth of tensor is same as input dimensions
        assert (d == self.in_dim)
        # call aggregation function on features and adjacency matrix
        agg_feats = self.agg(features, A)
        # cat concatenates two matrices along specified dimension
        cat_feats = torch.cat([features, agg_feats], dim=2)
        # This function provides a way of computing multilinear expressions (i.e. sums of products)
        # using the Einstein summation convention. Here: bnd of cat_features, df of weight matrix -> bnf
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out


class gcn(nn.Module):
    def __init__(self):
        super(gcn, self).__init__()
        # declare all elements of GCN
        self.convReduce = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.bn0 = nn.BatchNorm1d(512, affine=False)
        self.conv1 = GraphConv(512, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 512, MeanAggregator)
        self.conv3 = GraphConv(512, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256, MeanAggregator)
        # declare classifier used in end for obtaining edge weights
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(256),
            nn.Linear(256, 2))

    def forward(self, x, A, one_hop_idcs, args, train=True):
        # data normalization l2 -> bn
        # xnorm = x.norm(2,2,keepdim=True) + 1e-8
        # xnorm = xnorm.expand_as(x)
        # x = x.div(xnorm)
        # reshape tensor and apply reduction convolution
        x = x.transpose(1, 2)
        x = self.convReduce(x)
        x = x.transpose(1, 2)

        B, N, D = x.shape
        # reshape tensor for batch normalization
        x = x.reshape(-1, D)
        x = self.bn0(x)
        # reshape tensor for convolution
        x = x.view(B, N, D)
        # apply convolutions on x iteratively
        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        # obtain amount of one hop neighbors
        k1 = one_hop_idcs.size(-1)
        # obtain feature dimension
        dout = x.size(-1)
        # create edge feature matrix of 1-hop neighbors
        edge_feat = torch.zeros(B, k1, dout).to(args.gpu)
        # fill for each 1-hop neighbor its features in matrix
        for b in range(B):
            edge_feat[b, :, :] = x[b, one_hop_idcs[b]]

        edge_feat = edge_feat.view(-1, dout)
        pred = self.classifier(edge_feat)

        # shape: (B*k1)x2
        # (16 * 200)x2
        # 2 dimension cause we want probability that is not edge and is edge
        # 16 cause there are 16 nodes in batch
        return pred
