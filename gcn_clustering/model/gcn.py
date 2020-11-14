###################################################################
# File Name: gcn.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Modified by: Marius Bock
# mail: marius.bock@protonmail.com
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MeanAggregator(nn.Module):
    """
    Mean aggregation: performs average pooling among neighbors.
    Wang et al. did not provide other aggregators mentioned within their paper.
    Missing aggregators: weighted and attention aggregation
    """
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


class GraphConv(nn.Module):
    """
    GraphConv class (declared with input dimensions and output dimensions and aggregation prediciton_type)
    Defines the Graph Convolution layer as described in Wang et al.'s paper.
    """
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
        training deep feedforward neural networks - Glorot, features. & Bengio, labels. (2010), using a uniform 
        distribution.
        """
        init.xavier_uniform_(self.weight)
        # initializes bias tensor with all 0's
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        """
        Defines computation done at every call
        """
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
    """
    Actual GCN with all its layers and compuations. Returns for a graph input as defined by Wang et al. a prediction for
    each one-hop neighbor of each instance whether the two nodes should link or should not link.
    """
    def __init__(self, input_channels):
        super(gcn, self).__init__()
        # declare all elements of GCN
        self.convAdjustInput = nn.Conv1d(in_channels=input_channels, out_channels=512, kernel_size=1)
        torch.nn.init.xavier_uniform_(self.convAdjustInput.weight)
        self.bn0 = nn.BatchNorm1d(512, affine=False)
        # define four graph convolution layers
        self.conv1 = GraphConv(512, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 512, MeanAggregator)
        self.conv3 = GraphConv(512, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256, MeanAggregator)
        # define classifier layers used in end for obtaining edge weights
        self.classifier1 = nn.Linear(256, 256)
        self.classifier2 = nn.PReLU(256)
        self.classifier3 = nn.Linear(256, 2)

    def forward(self, x, A, one_hop_idcs, args, train=True):
        """
        Actual computations performed in GCN.

        Args:
            x -- feature matrix
            A -- adjacency matrix
            one_hop_idcs -- one-hop neighbor indices
            args -- settings to employ (see create_train_args, create_test_args)
            train -- whether used during training

        Returns:
            Predictions for all one-hop neighbors of instance (batch of instances), i.e. probability of link or no link
        """
        # data normalization l2 -> bn
        # xnorm = x.norm(2,2,keepdim=True) + 1e-8
        # xnorm = xnorm.expand_as(x)
        # x = x.div(xnorm)
        # reshape tensor and apply reduction/ upsample convolution (depending on type
        x = x.transpose(1, 2)
        x = self.convAdjustInput(x)
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
        pred = self.classifier1(edge_feat)
        pred = self.classifier2(pred)
        pred = self.classifier3(pred)

        # shape: (B*k1)x2
        # (16 * 200)x2
        # 2 dimension cause we want probability that is not edge and is edge
        # 16 cause there are 16 nodes in batch
        return pred


class gcn_intermediate(nn.Module):
    """
    Modified version of the GCN class which returns the features returned after the second classifier layer instead of
    predicitons. Used for the ensemble GCN setup.
    """
    def __init__(self, in_dim):
        super(gcn_intermediate, self).__init__()
        # declare all elements of GCN
        self.convAdjustInput = nn.Conv1d(in_channels=in_dim, out_channels=512, kernel_size=1)
        torch.nn.init.xavier_uniform_(self.convAdjustInput.weight)
        self.bn0 = nn.BatchNorm1d(512, affine=False)
        # define four graph convolutions
        self.conv1 = GraphConv(512, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 512, MeanAggregator)
        self.conv3 = GraphConv(512, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256, MeanAggregator)
        # declare classifier used in end for obtaining edge weights
        self.classifier1 = nn.Linear(256, 256)
        self.classifier2 = nn.PReLU(256)
        self.classifier3 = nn.Linear(256, 2)

    def forward(self, x, A, one_hop_idcs, args, train=True):
        """
        Actual computations performed in GCN.

        Args:
            x -- feature matrix
            A -- adjacency matrix
            one_hop_idcs -- one-hop neighbor indices
            args -- settings to employ (see create_train_args, create_test_args)
            train -- whether used during training

        Returns:
            Features returned after second classifier layer of GCN for a given instance or batch.
        """
        x = x.transpose(1, 2)
        x = self.convAdjustInput(x)
        x = x.transpose(1, 2)
        B, N, D = x.shape
        x = x.reshape(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)
        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        k1 = one_hop_idcs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B, k1, dout).to(args.gpu)
        for b in range(B):
            edge_feat[b, :, :] = x[b, one_hop_idcs[b]]

        edge_feat = edge_feat.view(-1, dout)
        out_features = self.classifier1(edge_feat)
        out_features = self.classifier2(out_features)

        return out_features


class gcn_feature_map(nn.Module):
    """
    Modified version of the GCN class which returns the 512 feature map instead of predicitons. Used for visualization.
    """
    def __init__(self, in_dim):
        super(gcn_feature_map, self).__init__()
        # declare all elements of GCN
        self.convAdjustInput = nn.Conv1d(in_channels=in_dim, out_channels=512, kernel_size=1)
        torch.nn.init.xavier_uniform_(self.convAdjustInput.weight)
        self.bn0 = nn.BatchNorm1d(512, affine=False)
        self.conv1 = GraphConv(512, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 512, MeanAggregator)
        self.conv3 = GraphConv(512, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256, MeanAggregator)
        # declare classifier used in end for obtaining edge weights
        self.classifier1 = nn.Linear(256, 256)
        self.classifier2 = nn.PReLU(256)
        self.classifier3 = nn.Linear(256, 2)

    def forward(self, x, A, one_hop_idcs, args, train=True):
        """
        Actual computations performed in GCN.

        Args:
            x -- feature matrix
            A -- adjacency matrix
            one_hop_idcs -- one-hop neighbor indices
            args -- settings to employ (see create_train_args, create_test_args)
            train -- whether used during training

        Returns:
            512 feature map of GCN for a given instance or batch.
        """
        x = x.transpose(1, 2)
        x = self.convAdjustInput(x)
        x = x.transpose(1, 2)
        B, N, D = x.shape
        x = x.reshape(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)
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
        return edge_feat
