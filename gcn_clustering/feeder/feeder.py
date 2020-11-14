###################################################################
# File Name: feeder.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Modified by: Marius Bock
# mail: marius.bock@protonmail.com
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import random
import torch
import torch.utils.data as data


class Feeder(data.Dataset):
    """
    Feeder Object that returns for an index the instance pivot subgraph according to the process mentioned
    in Wang et al.
    Added slight modifications namely: absolute difference calculation, pairwise element-wise and frame-pairwise
    element-wise products and normalised distance calculation. For more details see the provided thesis.
    """

    # Once initialised data is loaded and parameters are set
    def __init__(self, features, knn_graph, labels, seed, absolute_differences, normalise_distances,
                 element_wise_products_feeder, element_wise_products_type, k_at_hop, active_connection, train=True):
        """
        To create Feeder element for a dataset it needs to be given the dataset itself and all
        relevant parameters for graph creation; all variables are set (features, knn_graph, labels,
        num_samples (len of features), depth (len of k_at_hop), active connection (variable u in paper),
        train (boolean indicating if training or validation feeder); additionally parameters specifying which
        modifications to use during calculation are also set.

        Args:
            features -- features used to create feeder
            knn_graph -- knn graph used to create feeder
            labels -- labels used to create feeder
            seed -- seed to be used
            absolute_differences -- whether to calculate absolute differences instead of signed differences for
                                    node features
            normalise_distances -- whether to normalise distances between bounding boxes during node feature calculation
            element_wise_products_feeder -- boolean saying whether or not to calculate and append element-wise features
            element_wise_products_type -- type of element-wise features to calculate ('frame_pairwise' or 'pairwise')
            k_at_hop -- list of k_at_hop (parameter k in paper) employed during IPS construction
            active_connection -- active connections (parameter u in paper) employed during IPS construction
            train -- boolean indicating whether feeder is used during training or validation (changes return values)
        """
        np.random.seed(seed)
        random.seed(seed)
        self.features = features
        self.knn_graph = knn_graph[:, :k_at_hop[0] + 1]
        self.labels = labels
        self.num_samples = len(self.features)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.train = train
        self.element_wise_products_feeder = element_wise_products_feeder
        self.element_wise_products_type = element_wise_products_type
        self.absolute_differences = absolute_differences
        self.normalise_distances = normalise_distances
        assert np.mean(k_at_hop) >= active_connection

    def __len__(self):
        """
        Returns length of loader, i.e. number of samples
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        If wants to get any element of dataset it creates the respective instance pivot subgraph of that element using
        the attributes mentioned above. The requested element is then the pivot instance. First, a list of lists of all
        1-hop, 2-hop, ...,  h-hop neighbors (including pivot instance) is created. Then a set of said nodes is created
        (so that there are no duplicates and also removing pivot). Then tensor variables for the pivot_instance id
        (center_idx), one-hop indeces (one_hop_idcs) and features are created. For the features from each neighbor's
        feature vector features of the pivot are substracted. If selected, absolute differences are used here. Feature
        vector is tensor with dimensions [k1*(k2+1)+1]x[feature_length]. Note that not all spaces in vector are filled
        since neighborhood can actually be smaller (missing parts are filled out with zero). This is done so that
        tensors are all of same size. Same is done for adjacency matrix. The adjacency matrix is as mentioned in
        Wang et al.'s paper (only if among top-u neighbors). The adjacency matrix is normalized so that rows sum to 1.
        Then a edge_label matrix is created which states what are the true edges i.e. nodes that are actually the same
        instance within neighborhood. Additionally, the element-wise products are appended to the feature vector
        (if selected) and distances for spatial data are normalised (if selected).

        Args:
            index -- index of node that index pivot graph wants to be calculated

        Returns:
            If train = False, then returns features of neighborhood of center node, adjacency matrix, center node index
            one-hop neighborhood index, list of unique nodes, center node features (pivot_instance), edge labels of
            1-hop neighborhood
            If train = True, then returns same as above but without list of unique nodes and center node index
        """
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        hops = list()
        # node that is inspected (pivot instance)
        pivot_instance = index
        # set() creates set (distinct elements!)
        # appends nearest neighbors for pivot node
        hops.append(set(self.knn_graph[pivot_instance][1:]))
        # Actually we dont need the loop since the depth is fixed here,
        # But we still remain the code for further revision
        """
        Loop that goes 1 to depth
        At each step it appends a new empty set; h[-2] refers to all neighbors of second to last element in hops
        so at first iteration it is center node; updates the last element then with all neighbors belonging of a nn
        So in total it needs to go k1 times and saves for each its k2 neighbors
        """
        for d in range(1, self.depth):
            hops.append(set())
            # [-2] refers to second to last element, [-1] refers to last element
            for nearest_neighbor in hops[-2]:
                hops[-1].update(set(self.knn_graph[nearest_neighbor][1:self.k_at_hop[d] + 1]))
        # Creates a set of each neighborhood (no duplicates)
        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([pivot_instance, ])
        # add pivot_instance as well
        unique_nodes_list = list(hops_set)
        # creates list of node_number: number; probably done for faster look-up times
        unique_nodes_map = {j: i for i, j in enumerate(unique_nodes_list)}
        # create tensors of all needed elements for training
        center_idx = torch.Tensor([unique_nodes_map[pivot_instance], ]).type(torch.long)
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        center_feat = torch.Tensor(self.features[pivot_instance]).type(torch.float)
        feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        feat = feat.detach().numpy()
        center_feat = center_feat.detach().numpy()
        # absolute values to check if makes difference to signed version
        if self.normalise_distances:
            feat[:, 1] = (feat[:, 1] - center_feat[1]) / (feat[:, 3] + center_feat[3])
            feat[:, 2] = (feat[:, 2] - center_feat[2]) / (feat[:, 4] + center_feat[4])
            feat[:, 3] = (feat[:, 3] - center_feat[3]) / (feat[:, 3] + center_feat[3])
            feat[:, 4] = (feat[:, 4] - center_feat[4]) / (feat[:, 4] + center_feat[4])
            feat[:, 5:] = feat[:, 5:] - center_feat[5:]
        else:
            feat = feat - center_feat
        # absolute values to check if makes difference to signed version
        if self.absolute_differences:
            feat = np.abs(feat)
        # correlation part added to introduce multiplicative part to GCN
        if self.element_wise_products_feeder:
            if self.element_wise_products_type == 'pairwise':
                # pairwise products (source: https://stackoverflow.com/questions/46220362/python-fastest-way-of-adding-columns-containing-pairwise-products-of-column-ele)
                n = feat.shape[1] - 1
                r, c = np.triu_indices(n, 1)
                feat_pair = feat[:, r+1] * feat[:, c+1]
                feat = np.concatenate((feat, feat_pair), axis=1)
            elif self.element_wise_products_type == 'frame_pairwise':
                # multiply frame distance
                for i in range(feat.shape[1] - 1):
                    new_column = feat[:, i + 1] * feat[:, 5]
                    new_column = np.reshape(new_column, (-1, 1))
                    feat = np.concatenate((feat, new_column), axis=1)
        feat = torch.Tensor(feat).type(torch.float)
        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(unique_nodes_list)
        A = torch.zeros(num_nodes, num_nodes)

        _, fdim = feat.shape
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)

        # creates adjacency matrix. Only add edge if nearest neighbor (active_connection) is in nearest neighbors
        # of node compared to entire collection
        for node in unique_nodes_list:
            neighbors = self.knn_graph[node, 1:self.active_connection + 1]
            for n in neighbors:
                if n in unique_nodes_list:
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1
        # normalize adjacency so that rows sum to 1 individually
        D = A.sum(1, keepdim=True)
        A = A.div(D)
        # extend adjacency so that dimensions are fixed (add 0 so that it is (max_num_nodes)x(max_num_nodes)
        A_ = torch.zeros(max_num_nodes, max_num_nodes)
        A_[:num_nodes, :num_nodes] = A

        # create matrix that contains true edges of neighborhood (where neighbor in 1-hop has actually same label as
        # pivot (edge_labels matrix as result)
        labels = self.labels[np.asarray(unique_nodes_list)]
        labels = torch.from_numpy(labels).type(torch.long)
        # edge_labels = labels.expand(num_nodes,num_nodes).eq(
        #        labels.expand(num_nodes,num_nodes).t())
        one_hop_labels = labels[one_hop_idcs]
        center_label = labels[center_idx]
        edge_labels = (center_label == one_hop_labels).long()

        if self.train:
            return (feat, A_, center_idx, one_hop_idcs), edge_labels

        # Testing (also return unique_nodes_list)
        unique_nodes_list = torch.Tensor(unique_nodes_list)
        unique_nodes_list = torch.cat(
            [unique_nodes_list, torch.zeros(max_num_nodes - num_nodes)], dim=0)

        return (feat, A_, center_idx, one_hop_idcs, unique_nodes_list, pivot_instance), edge_labels


class TestFeeder(data.Dataset):
    """
    Feeder Object used for testing that returns for an index the instance pivot graph similar to the original Feeder
    object (just with modifications that features which require label information are not created nor returned)
    """

    def __init__(self, features, knn_graph, seed, absolute_differences, normalise_distances,
                 element_wise_products_feeder, element_wise_products_type, k_at_hop, active_connection):
        """
        To create TestFeeder element for a dataset it needs to be given the dataset itself and all
        relevant parameters for graph creation; all variables are set (features, knn_graph,
        num_samples (len of features), depth (len of k_at_hop), active connection (variable u in paper);
        additionally, settings which added modifications to apply are set as well.

        Args:
            features -- features used to create feeder
            knn_graph -- knn graph used to create feeder
            seed -- seed to be used
            absolute_differences -- whether to calculate absolute differences instead of signed differences for
                                    node features
            normalise_distances -- whether to normalise distances between bounding boxes during node feature calculation
            element_wise_products_feeder -- boolean saying whether or not to calculate and append element-wise features
            element_wise_products_type -- type of element-wise features to calculate ('frame_pairwise' or 'pairwise')
            k_at_hop -- list of k_at_hop (parameter k in paper) employed during IPS construction
            active_connection -- active connections (parameter u in paper) employed during IPS construction
        """
        np.random.seed(seed)
        random.seed(seed)
        self.features = features
        self.knn_graph = knn_graph[:, :k_at_hop[0] + 1]
        self.num_samples = len(self.features)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.element_wise_products_feeder = element_wise_products_feeder
        self.element_wise_products_type = element_wise_products_type
        self.absolute_differences = absolute_differences
        self.normalise_distances = normalise_distances
        assert np.mean(k_at_hop) >= active_connection

    def __len__(self):
        """
        Returns length of loader, i.e. number of samples
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        Same as for normal Feeder, but does not return edge labels (since testing and labels are not available)

        Args:
            index -- index to be used as pivot instance

        Returns:
            Returns features of neighborhood of center node, adjacency matrix, center node index one-hop
            neighborhood index, list of unique nodes, center node features (pivot_instance)
        """
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        hops = list()
        # node that is inspected (pivot instance)
        pivot_instance = index
        # set() creates set (distinct elements!)
        # appends nearest neighbors for pivot node
        hops.append(set(self.knn_graph[pivot_instance][1:]))
        # Actually we dont need the loop since the depth is fixed here,
        # But we still remain the code for further revision
        """
        Loop that goes 1 to depth
        At each step it appends a new empty set; h[-2] refers to all neighbors of second to last element in hops
        so at first iteration it is center node; updates the last element then with all neighbors belonging of a nn
        So in total it needs to go k1 times and saves for each its k2 neighbors
        """
        for d in range(1, self.depth):
            hops.append(set())
            # [-2] refers to second to last element, [-1] refers to last element
            for nearest_neighbor in hops[-2]:
                hops[-1].update(set(self.knn_graph[nearest_neighbor][1:self.k_at_hop[d] + 1]))
        # Creates a set of each neighborhood (no duplicates)
        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([pivot_instance, ])
        # add pivot_instance as well
        unique_nodes_list = list(hops_set)
        # creates list of node_number: number; probably done for faster look-up times
        unique_nodes_map = {j: i for i, j in enumerate(unique_nodes_list)}
        # create tensors of all needed elements for training
        center_idx = torch.Tensor([unique_nodes_map[pivot_instance], ]).type(torch.long)
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        center_feat = torch.Tensor(self.features[pivot_instance]).type(torch.float)
        feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        feat = feat.detach().numpy()
        center_feat = center_feat.detach().numpy()
        # absolute values to check if makes difference to signed version
        if self.normalise_distances:
            feat[:, 1] = (feat[:, 1] - center_feat[1]) / (feat[:, 3] + center_feat[3])
            feat[:, 2] = (feat[:, 2] - center_feat[2]) / (feat[:, 4] + center_feat[4])
            feat[:, 3] = (feat[:, 3] - center_feat[3]) / (feat[:, 3] + center_feat[3])
            feat[:, 4] = (feat[:, 4] - center_feat[4]) / (feat[:, 4] + center_feat[4])
            feat[:, 5:] = feat[:, 5:] - center_feat[5:]
        else:
            feat = feat - center_feat
        # absolute values to check if makes difference to signed version
        if self.absolute_differences:
            feat = np.abs(feat)
        # correlation part added to introduce multiplicative part to GCN
        if self.element_wise_products_feeder:
            if self.element_wise_products_type == 'pairwise':
                # pairwise products (source: https://stackoverflow.com/questions/46220362/python-fastest-way-of-adding-columns-containing-pairwise-products-of-column-ele)
                n = feat.shape[1] - 1
                r, c = np.triu_indices(n, 1)
                feat_pair = feat[:, r + 1] * feat[:, c + 1]
                feat = np.concatenate((feat, feat_pair), axis=1)
            elif self.element_wise_products_type == 'frame_pairwise':
                # multiply frame distance
                for i in range(feat.shape[1] - 1):
                    new_column = feat[:, i + 1] * feat[:, 5]
                    new_column = np.reshape(new_column, (-1, 1))
                    feat = np.concatenate((feat, new_column), axis=1)
        feat = torch.Tensor(feat).type(torch.float)
        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(unique_nodes_list)
        A = torch.zeros(num_nodes, num_nodes)

        _, fdim = feat.shape
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)

        # creates adjacency matrix. Only add edge if nearest neighbor (active_connection) is in nearest neighbors
        # of node compared to entire collection
        for node in unique_nodes_list:
            neighbors = self.knn_graph[node, 1:self.active_connection + 1]
            for n in neighbors:
                if n in unique_nodes_list:
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1
        # normalize adjacency so that rows sum to 1 individually
        D = A.sum(1, keepdim=True)
        A = A.div(D)
        # extend adjacency so that dimensions are fixed (add 0 so that it is (max_num_nodes)x(max_num_nodes)
        A_ = torch.zeros(max_num_nodes, max_num_nodes)
        A_[:num_nodes, :num_nodes] = A

        # Testing (also return unique_nodes_list)
        unique_nodes_list = torch.Tensor(unique_nodes_list)
        unique_nodes_list = torch.cat(
            [unique_nodes_list, torch.zeros(max_num_nodes - num_nodes)], dim=0)

        return feat, A_, center_idx, one_hop_idcs, unique_nodes_list, pivot_instance


class VisualizationFeeder(data.Dataset):
    """
    Feeder Object used for visualization that returns for an index the instance pivot graph similar to the original
    Feeder object (just with modifications what is actually returned, needed for visualization)
    """
    def __init__(self, features, knn_graph, labels, seed, absolute_differences, normalise_distances,
                 element_wise_products_feeder, element_wise_products_type, k_at_hop, active_connection, train=True):
        """
        To create VisualizationFeeder element for a dataset it needs to be given the dataset itself and all
        relevant parameters for graph creation; all variables are set (features, knn_graph, labels,
        num_samples (len of features), depth (len of k_at_hop), active connection (variable u in paper),
        train (boolean indicating if training or validation feeder).
        Additionally, settings which added modifications to apply are set as well.

        Args:
            features -- features used to create feeder
            knn_graph -- knn graph used to create feeder
            labels -- labels used to create feeder
            seed -- seed to be used
            absolute_differences -- whether to calculate absolute differences instead of signed differences for
                                    node features
            normalise_distances -- whether to normalise distances between bounding boxes during node feature calculation
            element_wise_products_feeder -- boolean saying whether or not to calculate and append element-wise features
            element_wise_products_type -- type of element-wise features to calculate ('frame_pairwise' or 'pairwise')
            k_at_hop -- list of k_at_hop (parameter k in paper) employed during IPS construction
            active_connection -- active connections (parameter u in paper) employed during IPS construction
            train -- boolean indicating whether feeder is used during training or validation (changes return values)
        """
        np.random.seed(seed)
        random.seed(seed)
        self.features = features
        self.knn_graph = knn_graph[:, :k_at_hop[0] + 1]
        self.labels = labels
        self.num_samples = len(self.features)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.train = train
        self.correlate_feeder = element_wise_products_feeder
        self.correlation_type = element_wise_products_type
        self.absolute_differences = absolute_differences
        self.normalise_distances = normalise_distances
        assert np.mean(k_at_hop) >= active_connection

    def __len__(self):
        """
        Returns length of loader, i.e. number of samples
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        Same as for normal Feeder, but also returns labels of all edges in IPS

        Args:
            index -- index to be used as pivot instance

        Returns:
            If train = False, then returns features of neighborhood of center node, adjacency matrix, center node index
            one-hop neighborhood index, list of unique nodes, center node features (pivot_instance), edge labels of
            1-hop neighborhood and labels of all edges in IPS
            If train = True, then returns same as above but without list of unique nodes indices
        """
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        hops = list()
        center_node = index
        hops.append(set(self.knn_graph[center_node][1:]))

        # Actually we dont need the loop since the depth is fixed here,
        # But we still remain the code for further revision
        for d in range(1, self.depth):
            hops.append(set())
            for h in hops[-2]:
                hops[-1].update(set(self.knn_graph[h][1:self.k_at_hop[d] + 1]))

        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([center_node, ])
        unique_nodes_list = list(hops_set)
        unique_nodes_map = {j: i for i, j in enumerate(unique_nodes_list)}

        center_idx = torch.Tensor([unique_nodes_map[center_node], ]).type(torch.long)
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        center_feat = torch.Tensor(self.features[center_node]).type(torch.float)
        feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        feat = feat.detach().numpy()
        center_feat = center_feat.detach().numpy()
        # absolute values to check if makes difference to signed version
        if self.normalise_distances:
            feat[:, 1] = (feat[:, 1] - center_feat[1]) / (feat[:, 3] + center_feat[3])
            feat[:, 2] = (feat[:, 2] - center_feat[2]) / (feat[:, 4] + center_feat[4])
            feat[:, 3] = (feat[:, 3] - center_feat[3]) / (feat[:, 3] + center_feat[3])
            feat[:, 4] = (feat[:, 4] - center_feat[4]) / (feat[:, 4] + center_feat[4])
            feat[:, 5:] = feat[:, 5:] - center_feat[5:]
        else:
            feat = feat - center_feat
        # absolute values to check if makes difference to signed version
        if self.absolute_differences:
            feat = np.abs(feat)
        # correlation part added to introduce multiplicative part to GCN
        if self.correlate_feeder:
            if self.correlation_type == 'pairwise':
                # pairwise products (source: https://stackoverflow.com/questions/46220362/python-fastest-way-of-adding-columns-containing-pairwise-products-of-column-ele)
                n = feat.shape[1] - 1
                r, c = np.triu_indices(n, 1)
                feat_pair = feat[:, r + 1] * feat[:, c + 1]
                feat = np.concatenate((feat, feat_pair), axis=1)
            elif self.correlation_type == 'frame_pairwise':
                # multiply frame distance
                for i in range(feat.shape[1] - 1):
                    new_column = feat[:, i + 1] * feat[:, 5]
                    new_column = np.reshape(new_column, (-1, 1))
                    feat = np.concatenate((feat, new_column), axis=1)
        feat = torch.Tensor(feat).type(torch.float)

        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(unique_nodes_list)
        A = torch.zeros(num_nodes, num_nodes)

        _, fdim = feat.shape
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)

        for node in unique_nodes_list:
            neighbors = self.knn_graph[node, 1:self.active_connection + 1]
            for n in neighbors:
                if n in unique_nodes_list:
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1

        D = A.sum(1, keepdim=True)
        A = A.div(D)
        A_ = torch.zeros(max_num_nodes, max_num_nodes)
        A_[:num_nodes, :num_nodes] = A

        labels = self.labels[np.asarray(unique_nodes_list)]
        labels = torch.from_numpy(labels).type(torch.long)
        # edge_labels = labels.expand(num_nodes,num_nodes).eq(
        #        labels.expand(num_nodes,num_nodes).t())
        one_hop_labels = labels[one_hop_idcs]
        center_label = labels[center_idx]
        edge_labels = (center_label == one_hop_labels).long()

        if self.train:
            return (feat, A_, center_idx, one_hop_idcs), edge_labels, labels

        # Testing
        unique_nodes_list = torch.Tensor(unique_nodes_list)
        unique_nodes_list = torch.cat(
            [unique_nodes_list, torch.zeros(max_num_nodes - num_nodes)], dim=0)
        return (feat, A_, center_idx, one_hop_idcs, unique_nodes_list), edge_labels, labels
