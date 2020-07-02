import os
import pandas as pd
import numpy as np

import torch
from torch.backends import cudnn

from dataset_creation import create_modified_detection_file, create_knn_graph_dataset, adjust_labeling
from gcn_clustering.feeder.feeder_visualization import Feeder
from gcn_clustering.utils import to_numpy

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_knn(features, labels, index, one_hop_indeces, adjacency_matrix):
    # x_min, x_max = np.min(features,0), np.max(features,0)
    # features = (features-x_min) / (x_max  - x_min)
    plt.figure(figsize=(20, 20))
    for i in range(labels.shape[0]):

        if i == index:
            c = 'g'
            s = 500
        elif labels[i] == labels[index]:
            c = 'r'
            s = 20
        else:
            c = 'b'
            s = 20
        plt.scatter(features[i, 0], features[i, 1], s, color=c)
    for one_hop_index in one_hop_indeces:
        c = 'r' if labels[index] == labels[one_hop_index] else 'gray'
        w = 1 if labels[index] == labels[one_hop_index] else 0.5

        plt.plot([features[index, 0], features[one_hop_index, 0]], [features[index, 1], features[one_hop_index, 1]], linestyle='--', linewidth=w, color=c)

    plt.show()


def plot_embedding(features, labels, cid, one_hop_indeces, adjacency_matrix):
    # x_min, x_max = np.min(features,0), np.max(features,0)
    # features = (features-x_min) / (x_max  - x_min)
    plt.figure(figsize=(20, 20))
    for i in range(labels.shape[0]):

        if i == cid:
            c = 'g'
            s = 500
        elif labels[i] == labels[cid]:
            c = 'r'
            s = 20
        else:
            c = 'b'
            s = 20
        plt.scatter(features[i, 0], features[i, 1], s, color=c)
    # for one_hop_index in one_hop_indeces:
    #    c='r' if labels[index]==labels[one_hop_index] else 'gray'
    #    w=1 if labels[index]==labels[one_hop_index] else 0.5
    #    if labels[index]==labels[one_hop_index]:
    #        plt.plot([features[index,0], features[one_hop_index,0]],[features[index,1], features[one_hop_index,1]], linestyle='--', linewidth=w, color=c)
    edges = adjacency_matrix.nonzero()
    edges = np.asarray(edges).T
    for e in edges:
        plt.plot([features[e[0], 0], features[e[1], 0]], [features[e[0], 1], features[e[1], 1]], linestyle='--', linewidth=0.5, color='gray')

    plt.show()


if __name__ == '__main__':
    idx = 1112
    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    directory = 'data/MOT/MOT17'
    sequence = 'MOT17-02'
    detector = 'MOT17-02-DPM'
    metadata_path = '20200621-135039_metadata.csv'

    metadata_file = pd.read_csv(os.path.join(directory, sequence, detector, metadata_path))
    detection_file = create_modified_detection_file(
        np.loadtxt(os.path.join(directory, sequence, detector, 'det.txt'), delimiter=","))

    # labels
    full_labels = metadata_file[metadata_file.columns[pd.Series(metadata_file.columns).str.startswith('labels')]].iloc[
                  :, 0].to_numpy()
    train_labels = adjust_labeling(full_labels[metadata_file['fil_train_0.7_0.3']])
    val_labels = adjust_labeling(full_labels[metadata_file['fil_valid_0.7_0.3']])

    # features
    full_features = np.load(os.path.join(directory, sequence, detector, 'feat_app_pool.npy'))
    train_features = full_features[metadata_file['fil_train_0.7_0.3']]
    val_features = full_features[metadata_file['fil_valid_0.7_0.3']]

    # knn graph
    train_knn_graph = create_knn_graph_dataset(train_features, 200, 'brute')
    val_knn_graph = create_knn_graph_dataset(val_features, 200, 'brute')

    trainset = Feeder(train_features,
                      train_knn_graph,
                      train_labels,
                      seed,
                      [200, 5],
                      5)
    valset = Feeder(val_features,
                    val_knn_graph,
                    val_labels,
                    seed,
                    [20, 5],
                    5, )
    # train=False)

    (feat, A, index, one_hop_idcs), edge_labels, labels = valset[idx]
    feat, A, index, one_hop_idcs, edge_labels, labels = map(to_numpy, (feat, A, index, one_hop_idcs, edge_labels, labels))

    unique_labels = np.unique(labels)
    label_map = {j:i for i, j in enumerate(unique_labels)}
    labels = [label_map[l] for l in labels]
    labels = np.asarray(labels)

    feat_tsne = TSNE(n_components=2).fit_transform(feat)

    plot_embedding(feat_tsne, labels, index, one_hop_idcs, A)

    plot_knn(feat_tsne, labels, index, one_hop_idcs, A)