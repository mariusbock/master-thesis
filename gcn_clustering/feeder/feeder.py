###################################################################
# File Name: feeder.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 01:06:16 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time

import numpy as np
import random
import torch
import torch.utils.data as data

'''
Feeder class:
    Functions:
        - init:     to create Feeder element for a dataset it needs to be given the dataset itself and all 
                    relevant parameters for graph creation; all variables are set (features, knn_graph, labels, 
                    num_samples (len of features), depth (len of k_at_hop), active connection (variable u in paper), 
                    train (boolean indicating if training or testing feeder)
                
        - len:      returns number of samples
        
        - getitem:  if wants to get any element of dataset it creates the respective subgraph of that element using
                    the attributes mentioned above; requested element is then pivot instance; first creates a list of
                    lists of all 1-hop, 2-hop, ...,  h-hop neighbors (including pivot instance) and then creates a set
                    of all nodes (no duplicates without pivot); then creates tensor variables for pivot_instance id 
                    (center_idx), one-hop indeces (one_hop_idcs) and features (each neighbor - features of pivot
                    instance; features is tensor with dimensions [k1*(k2+1)+1]x[feature_length] -> can be actually
                    smaller so missing parts are filled out with zero (if there are are less neighbors than there can 
                    be maximal) -> done so that tensors are all of same size. Same is done for adjacency matrix
                    Adjacency matrix is build according to paper (only if among top-u neighbors); adjacency is 
                    normalized so that rows sum to 1.
                    Then creates edge_label matrix which says what are true edges i.e. nodes that are actually the same 
                    instance within neighborhood.
                    If training then (feat, A_, center_idx, one_hop_idcs), edge_labels are returned.
                    If testing then unique_nodes_list is returned as well.
                    
                    

'''


class Feeder(data.Dataset):
    '''
    Generate a sub-graph from the feature graph centered at some node, 
    and now the sub-graph has a fixed depth, i.e. 2
    '''
    # Once initialised data is loaded and parameters are set
    def __init__(self, feat_path, knn_graph_path, label_path, seed=1, 
                 k_at_hop=[200,5], active_connection=5, train=True):
        np.random.seed(seed)
        random.seed(seed)
        self.features = np.load(feat_path)
        self.knn_graph = np.load(knn_graph_path)[:,:k_at_hop[0]+1]
        self.labels = np.load(label_path)
        self.num_samples = len(self.features)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.train = train
        assert np.mean(k_at_hop)>=active_connection

    def __len__(self):
        """
        Returns number of samples
        """
        return self.num_samples

    def __getitem__(self, index):
        '''
        return the vertex feature and the adjacent matrix A, together 
        with the indices of the center node and its 1-hop nodes
        Note that this is done per BATCH! Thus this method returns more than one idx
        '''
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
        for d in range(1,self.depth): 
            hops.append(set())
            # [-2] refers to second to last element, [-1] refers to last element
            for nearest_neighbor in hops[-2]:
                hops[-1].update(set(self.knn_graph[nearest_neighbor][1:self.k_at_hop[d]+1]))
        # Creates a set of each neighborhood (no duplicates)
        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([pivot_instance,])
        # add pivot_instance as well
        unique_nodes_list = list(hops_set)
        # creates list of node_number: number; probably done for faster look-up times
        unique_nodes_map = {j:i for i,j in enumerate(unique_nodes_list)}
        # create tensors of all needed elements for training
        center_idx = torch.Tensor([unique_nodes_map[pivot_instance],]).type(torch.long)
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        center_feat = torch.Tensor(self.features[pivot_instance]).type(torch.float)
        feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        feat = feat - center_feat
        
        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(unique_nodes_list)
        A = torch.zeros(num_nodes, num_nodes)

        _, fdim = feat.shape
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)

        # creates adjacency matrix. Only add edge if nearest neighbor (active_connection) is in nearest neighbors
        # of node compared to entire collection
        for node in unique_nodes_list:
            neighbors = self.knn_graph[node, 1:self.active_connection+1]
            for n in neighbors:
                if n in unique_nodes_list: 
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1
        # normalize adjacency so that rows sum to 1 individually
        D = A.sum(1, keepdim=True)
        A = A.div(D)
        # extend adjacency so that dimensions are fixed (add 0 so that it is (max_num_nodes)x(max_num_nodes)
        A_ = torch.zeros(max_num_nodes,max_num_nodes)
        A_[:num_nodes,:num_nodes] = A

        # create matrix that contains true edges of neighborhood (where neighbor in 1-hop has actually same label as
        # pivot (edge_labels matrix as result)
        labels = self.labels[np.asarray(unique_nodes_list)]
        labels = torch.from_numpy(labels).type(torch.long)
        #edge_labels = labels.expand(num_nodes,num_nodes).eq(
        #        labels.expand(num_nodes,num_nodes).t())
        one_hop_labels = labels[one_hop_idcs]
        center_label = labels[center_idx]
        edge_labels = (center_label == one_hop_labels).long()

        if self.train:
            return (feat, A_, center_idx, one_hop_idcs), edge_labels

        # Testing (also return unique_nodes_list)
        unique_nodes_list = torch.Tensor(unique_nodes_list)
        unique_nodes_list = torch.cat(
                [unique_nodes_list, torch.zeros(max_num_nodes-num_nodes)], dim=0)

        return(feat, A_, center_idx, one_hop_idcs, unique_nodes_list), edge_labels



