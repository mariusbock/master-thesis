import argparse
import os
import time

import numpy as np
import pandas as pd

from gcn_clustering.train import train_main
from gcn_clustering.test import test_main

from dataset_creation import create_feature_dataset, create_knn_graph_dataset, adjust_labeling


def create_args(detector, date, timestamp, seed, workers, print_freq, cuda, lr, momentum, weight_decay, epochs, batch_size,
                features, knn_graph, labels, k_at_hop, active_connection):
    """
    Function that creates arguments object that is needed for calling the training/ testing method of the GCN clustering
    detector = detector name, date and timestamp = needed for creating the proper folders
    """
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join('../logs', date, timestamp + '-' + detector))
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--workers', default=workers, type=int)
    parser.add_argument('--print_freq', default=print_freq, type=int)
    parser.add_argument('--cuda', default=cuda, type=bool)

    # Optimization args
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--epochs', type=int, default=epochs)

    # Training args
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--features', default=features)
    parser.add_argument('--knn_graph', default=knn_graph)
    parser.add_argument('--labels', default=labels)
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=k_at_hop)
    parser.add_argument('--active_connection', type=int, default=active_connection)

    parser.add_argument('--checkpoint', type=str, metavar='PATH',
                        default=os.path.join('../logs', date, timestamp + '-' + detector, 'epoch_4.ckpt'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    date = time.strftime("%Y%m%d")
    timestamp = time.strftime("%H%M%S")

    sequence = 'MOT/MOT17/MOT17-02'
    detector = 'sdp'
    metadata_path = '20200618-193212_metadata.csv'
    image_folder_path = os.path.join('../data', sequence, 'images')

    batch_size = 1
    gpu_name = 'cuda:0'
    max_pool = True

    # Face Datasets for comparison
    # label_dataset_faces = np.load("../data/facedata/512.labels.npy")
    # feature_dataset_faces = np.load("../data/facedata/512.fea.npy")
    # knn_graph_faces = np.load("../data/facedata/knn.graph.512.bf.npy")

    metadata_file = pd.read_csv(os.path.join('../data', sequence, detector, metadata_path))
    detection_file = np.loadtxt(os.path.join('../data', sequence, detector, 'det.txt'), delimiter=",")

    full_labels = metadata_file[metadata_file.columns[pd.Series(metadata_file.columns).str.startswith('labels')]].to_numpy()
    full_features = create_feature_dataset(detection_file, image_folder_path, batch_size, gpu_name, max_pool)

    train_features = full_features[metadata_file['fil_train_0.7_0.3']]
    val_features = full_features[metadata_file['fil_valid_0.7_0.3']]
    test_features = full_features[metadata_file['fil_test_0.5_0.3']]

    train_labels = adjust_labeling(full_labels[metadata_file['fil_train_0.7_0.3']])
    val_labels = adjust_labeling(full_labels[metadata_file['fil_valid_0.7_0.3']])
    test_labels = adjust_labeling(full_labels[metadata_file['fil_test_0.5_0.3']])

    train_neighbors = len(train_labels)
    val_neighbors = len(train_labels)
    test_neighbors = len(train_labels)

    if len(train_labels) > 200:
        train_neighbors = 200
    if len(val_labels) > 200:
        val_neighbors = 200
    if len(test_labels) > 200:
        test_neighbors = 200

    train_knn_graph = create_knn_graph_dataset(train_features, train_neighbors, 'brute')
    val_knn_graph = create_knn_graph_dataset(val_features, val_neighbors, 'brute')
    test_knn_graph = create_knn_graph_dataset(test_features, test_neighbors, 'brute')

    print("TRAIN SHAPES")
    print(train_labels.shape)
    print(train_features.shape)
    print(train_knn_graph.shape)

    print("VALID SHAPES")
    print(val_labels.shape)
    print(val_features.shape)
    print(val_knn_graph.shape)

    print("TEST SHAPES")
    print(test_labels.shape)
    print(test_features.shape)
    print(test_knn_graph.shape)

    train_args = create_args(detector=detector,
                             date=date,
                             timestamp=timestamp,
                             seed=1,
                             workers=16,
                             print_freq=1,
                             cuda=True,
                             lr=1e-2,
                             momentum=0.9,
                             weight_decay=1e-4,
                             epochs=4,
                             batch_size=16,
                             features=train_features,
                             knn_graph=train_knn_graph,
                             labels=train_labels,
                             k_at_hop=[200, 10],
                             active_connection=10,
                             )

    train_main(train_args)

    val_args = create_args(detector=detector,
                           timestamp=timestamp,
                           date=date,
                           seed=1,
                           workers=16,
                           print_freq=1,
                           cuda=True,
                           lr=1e-5,
                           momentum=0.9,
                           weight_decay=1e-4,
                           epochs=20,
                           batch_size=32,
                           features=val_features,
                           knn_graph=val_knn_graph,
                           labels=val_labels,
                           k_at_hop=[20, 5],
                           active_connection=5
                           )

    test_main(val_args)

    test_args = create_args(detector=detector,
                            timestamp=timestamp,
                            date=date,
                            seed=1,
                            workers=16,
                            print_freq=1,
                            cuda=True,
                            lr=1e-5,
                            momentum=0.9,
                            weight_decay=1e-4,
                            epochs=20,
                            batch_size=32,
                            features=test_features,
                            knn_graph=test_knn_graph,
                            labels=test_labels,
                            k_at_hop=[20, 5],
                            active_connection=5
                            )

    test_main(test_args)
