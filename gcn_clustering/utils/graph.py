from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import time


class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other, score):
        self.__links.add(other)
        other.__links.add(self)


def connected_components(nodes, score_dict, th):
    '''
    conventional/ naive connected components searching; goes for each node through its connections and keeps edge if
    for above certain threshold
    th = threshold
    '''
    result = []
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        while queue:
            n = queue.pop(0)
            if th is not None:
                neighbors = {l for l in n.links if score_dict[tuple(sorted([n.name, l.name]))] >= th}
            else:
                neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
        result.append(group)
    return result


def connected_components_constraint(nodes, max_sz, score_dict=None, th=None):
    '''
    only use edges whose scores are above `th`
    if a component is larger than `max_sz`, all the nodes in this component are added into `remain` and
    returned for next iteration.
    '''
    result = []
    remain = set()
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        valid = True
        while queue:
            n = queue.pop(0)
            if th is not None:
                neighbors = {l for l in n.links if score_dict[tuple(sorted([n.name, l.name]))] >= th}
            else:
                neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
            if len(group) > max_sz or len(remain.intersection(neighbors)) > 0:
                # if this group is larger than `max_sz`, add the nodes into `remain`
                valid = False
                remain.update(group)
                break
        if valid:  # if this group is smaller than or equal to `max_sz`, finalize it.
            result.append(group)
    return result, remain


def graph_propagation_naive(edges, score, th):
    """
    returns clustering of edges given scores and threshold in naive way
    """
    edges = np.sort(edges, axis=1)

    # construct dictionary for easy lookup of costs of edge
    score_dict = {}  # score lookup table
    for i, e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]

    # sort nodes and remove duplicates
    nodes = np.sort(np.unique(edges.flatten()))
    # creates a numpy array of the size of amount of nodes
    mapping = -1 * np.ones((nodes.max() + 1), dtype=np.int)
    # creates array of size of amount of nodes (going from 0 to num_nodes)
    mapping[nodes] = np.arange(nodes.shape[0])
    # creates array that contains ids of linked nodes [node_1, node_2]
    link_idx = mapping[edges]
    # creates vertex data structure for all nodes and adds all link information
    vertex = [Data(n) for n in nodes]
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)

    # creates connected components
    # first iteration
    comps = connected_components(vertex, score_dict, th)

    return comps


def graph_propagation(edges, score, max_sz, step=0.1, beg_th=0.9, pool=None):
    edges = np.sort(edges, axis=1)
    th = score.min()
    # th = beg_th
    # construct graph
    score_dict = {}  # score lookup table
    if pool is None:
        # if pool = none create lookup table normal
        for i, e in enumerate(edges):
            score_dict[e[0], e[1]] = score[i]
    elif pool == 'avg':
        for i, e in enumerate(edges):
            # checks if edge already exists if so then compute average else just assign value
            if (e[0], e[1]) in score_dict:
                score_dict[e[0], e[1]] = 0.5 * (score_dict[e[0], e[1]] + score[i])
            else:
                score_dict[e[0], e[1]] = score[i]
    elif pool == 'max':
        # checks if edge already exists if so keep only max else just assign value
        for i, e in enumerate(edges):
            if (e[0], e[1]) in score_dict:
                score_dict[e[0], e[1]] = max(score_dict[e[0], e[1]], score[i])
            else:
                score_dict[e[0], e[1]] = score[i]
    else:
        raise ValueError('Pooling operation not supported')
    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max() + 1), dtype=np.int)
    # creates array of size of amount of nodes (going from 0 to num_nodes)
    mapping[nodes] = np.arange(nodes.shape[0])
    # creates array that contains ids of linked nodes [node_1, node_2]
    link_idx = mapping[edges]
    # creates vertex data structure for all nodes and adds all link information
    vertex = [Data(n) for n in nodes]
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)

    """
    until no remaining nodes left use connected_components_constraint function to come up with components
    if clusters are too large then threshold is increased according to step value
    """
    # first iteration
    comps, remain = connected_components_constraint(vertex, max_sz)

    # iteration
    components = comps[:]
    while remain:
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
    return components


def graph_propagation_soft(edges, score, max_sz, step=0.1, **kwargs):
    """
    Graph propagation using BFS to propagate labels
    """
    edges = np.sort(edges, axis=1)
    # set threshold to minimum score (so that all are assigned a cluster??)
    th = score.min()

    # construct graph
    score_dict = {}  # score lookup table
    for i, e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max() + 1), dtype=np.int)
    # creates array of size of amount of nodes (going from 0 to num_nodes)
    mapping[nodes] = np.arange(nodes.shape[0])
    # creates array that contains ids of linked nodes [node_1, node_2]
    link_idx = mapping[edges]
    # creates vertex data structure for all nodes and adds all link information
    vertex = [Data(n) for n in nodes]
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)

    # first iteration (run connected_components with constraint on max size)
    comps, remain = connected_components_constraint(vertex, max_sz)
    # creates two lists; one empty and one with all nodes
    first_vertex_idx = np.array([mapping[n.name] for c in comps for n in c])
    fusion_vertex_idx = np.setdiff1d(np.arange(nodes.shape[0]), first_vertex_idx, assume_unique=True)
    # iteration
    # assign all nodes part of comps to new variable
    components = comps[:]
    while remain:
        # increase threshold and add new comps to components variable until remaining nodes are all assigned
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
    # create label dictionary
    label_dict = {}
    for i, c in enumerate(components):
        for n in c:
            label_dict[n.name] = i
    print('Propagation ...')
    # create vertex list for propagation
    prop_vertex = [vertex[idx] for idx in fusion_vertex_idx]
    # start diffusion
    label, label_fusion = diffusion(prop_vertex, label_dict, score_dict, **kwargs)
    return label, label_fusion


def diffusion(vertex, label, score_dict, max_depth=5, weight_decay=0.6, normalize=True):
    class BFSNode():
        """
        Class BFS node with node itself, depth and value
        """
        def __init__(self, node, depth, value):
            self.node = node
            self.depth = depth
            self.value = value

    # creates list of node name and label that it is assigned
    label_fusion = {}
    for name in label.keys():
        label_fusion[name] = {label[name]: 1.0}
    # variables to account for progress
    prog = 0
    prog_step = len(vertex) // 20
    start = time.time()
    for root in vertex:
        if prog % prog_step == 0:
            print("progress: {} / {}, elapsed time: {}".format(prog, len(vertex), time.time() - start))
        prog += 1
        # queue = {[root, 0, 1.0]}
        queue = {BFSNode(root, 0, 1.0)}
        visited = [root.name]
        root_label = label[root.name]
        while queue:
            curr = queue.pop()
            if curr.depth >= max_depth:  # pruning
                continue
            neighbors = curr.node.links
            tmp_value = []
            tmp_neighbor = []
            for n in neighbors:
                if n.name not in visited:
                    sub_value = score_dict[tuple(sorted([curr.node.name, n.name]))] * weight_decay * curr.value
                    tmp_value.append(sub_value)
                    tmp_neighbor.append(n)
                    if root_label not in label_fusion[n.name].keys():
                        label_fusion[n.name][root_label] = sub_value
                    else:
                        label_fusion[n.name][root_label] += sub_value
                    visited.append(n.name)
                    # queue.add([n, curr.depth+1, sub_value])
            sortidx = np.argsort(tmp_value)[::-1]
            for si in sortidx:
                queue.add(BFSNode(tmp_neighbor[si], curr.depth + 1, tmp_value[si]))
    if normalize:
        for name in label_fusion.keys():
            summ = sum(label_fusion[name].values())
            for k in label_fusion[name].keys():
                label_fusion[name][k] /= summ
    return label, label_fusion
