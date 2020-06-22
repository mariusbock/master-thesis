import argparse
import os
import time

import numpy as np
import pandas as pd

from gcn_clustering.train import train_main
from gcn_clustering.test import test_main

from dataset_creation import create_modified_detection_file, create_feature_dataset, create_knn_graph_dataset, adjust_labeling


def create_args(detector_name, date_string, timestamp_string, seed, workers, print_freq, gpu, lr, momentum, weight_decay, epochs, batch_size,
                features, knn_graph, labels, k_at_hop, active_connection):
    """
    Function that creates arguments object that is needed for calling the training/ testing method of the GCN clustering
    detector = detector name, date and timestamp = needed for creating the proper folders
    """
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join('../logs', date_string, timestamp_string + '-' + detector_name))
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--workers', default=workers, type=int)
    parser.add_argument('--print_freq', default=print_freq, type=int)
    parser.add_argument('--gpu', default=gpu, type=str)

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
                        default=os.path.join('../logs', date_string, timestamp_string + '-' + detector_name, 'epoch_4.ckpt'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # experiment parameters
    date = time.strftime("%Y%m%d")
    timestamp = time.strftime("%H%M%S")
    sequence = 'MOT/MOT17/MOT17-04'
    detector = 'sdp'
    metadata_path = '20200621-180259_metadata.csv'
    image_folder_path = os.path.join('../data', sequence, 'images')

    # feature creation parameters
    batch_size_features = 1
    gpu_name = 'cuda:0'
    max_pool = True

    # Face Datasets for comparison
    # label_dataset_faces = np.load("../data/facedata/512.labels.npy")
    # feature_dataset_faces = np.load("../data/facedata/512.fea.npy")
    # knn_graph_faces = np.load("../data/facedata/knn.graph.512.bf.npy")

    metadata_file = pd.read_csv(os.path.join('../data', sequence, detector, metadata_path))
    detection_file = create_modified_detection_file(np.loadtxt(os.path.join('../data', sequence, detector, 'det.txt'), delimiter=","))

    # obtain labels from metadata file and save to numpy array
    full_labels = metadata_file[metadata_file.columns[pd.Series(metadata_file.columns).str.startswith('labels')]].iloc[:, 0].to_numpy()

    # create splits and adjust labeling so that it goes from 0 to num_classes in split
    train_labels = adjust_labeling(full_labels[metadata_file['fil_train_0.7_0.3']])
    val_labels = adjust_labeling(full_labels[metadata_file['fil_valid_0.7_0.3']])
    test_labels = adjust_labeling(full_labels[metadata_file['fil_test_0.5_0.3']])

    # create features and apply splits
    full_features = create_feature_dataset(detection_file, image_folder_path, batch_size_features, gpu_name, max_pool)

    train_features = full_features[metadata_file['fil_train_0.7_0.3']]
    val_features = full_features[metadata_file['fil_valid_0.7_0.3']]
    test_features = full_features[metadata_file['fil_test_0.5_0.3']]

    # this section checks whether the train/ test/ valid splits are too small for creating a KNN graph with 200
    # neighbors. if there are more than 200 instances then one can create a KNN graph with 200 nearest neighbors
    # if there are less instances the maximum size knn graph is created (i.e. len(split))
    train_neighbors = len(train_labels)
    val_neighbors = len(train_labels)
    test_neighbors = len(train_labels)

    if len(train_labels) > 200:
        train_neighbors = 200
    if len(val_labels) > 200:
        val_neighbors = 200
    if len(test_labels) > 200:
        test_neighbors = 200

    # create knn graphs for each split (ensures labeling is correct)
    train_knn_graph = create_knn_graph_dataset(train_features, train_neighbors, 'brute')
    val_knn_graph = create_knn_graph_dataset(val_features, val_neighbors, 'brute')
    test_knn_graph = create_knn_graph_dataset(test_features, test_neighbors, 'brute')

    # print shapes of files
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

    # training
    train_args = create_args(detector_name=detector,
                             date_string=date,
                             timestamp_string=timestamp,
                             seed=1,
                             workers=16,
                             print_freq=100,
                             gpu=gpu_name,
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

    # validation
    val_args = create_args(detector_name=detector,
                           timestamp_string=timestamp,
                           date_string=date,
                           seed=1,
                           workers=16,
                           print_freq=100,
                           gpu=gpu_name,
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

    # testing
    test_args = create_args(detector_name=detector,
                            timestamp_string=timestamp,
                            date_string=date,
                            seed=1,
                            workers=16,
                            print_freq=100,
                            gpu=gpu_name,
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
