import argparse
import numpy as np
import pandas as pd


def create_train_args(seed, workers, print_freq, gpu, input_channels, save_checkpoints, checkpoint_directory, lr,
                      momentum, weight_decay, epochs, batch_size, features, knn_graph, labels, k_at_hop,
                      active_connection):
    """
    Function that creates arguments object that is needed for calling the training method of the GCN clustering

    Parameters:
    seed -- seed used during training
    workers -- number of workers employed during testing
    print_freq -- batch print frequency of GCN output
    gpu -- gpu used for testing
    input_channels -- dimension of feature dataset that is used for input (needed for proper compression)
    save_checkpoints -- boolean whether to save checkpoints in between epochs
    checkpoint_directory -- directory where to save checkpoints to
    lr -- learning rate during training
    momentum -- momentum during training
    weight_decay -- weight decay during training
    epochs -- number of epochs for testing
    batch_size -- size of batch for testing steps
    features -- feature dataset
    knn_graph -- knn graph dataset
    labels -- label dataset
    k-at-hop -- k-at-hop employed for instance pivot graph creation
    active_connection -- active connection employed for instance pivot graph creation
    """
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--workers', default=workers, type=int)
    parser.add_argument('--print_freq', default=print_freq, type=int)
    parser.add_argument('--gpu', default=gpu, type=str)
    parser.add_argument('--input_channels', type=int, default=input_channels)
    parser.add_argument('--save_checkpoints', type=bool, default=save_checkpoints)
    parser.add_argument('--checkpoint_directory', type=str, metavar='PATH', default=checkpoint_directory)

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

    args = parser.parse_args()
    return args


def create_test_args(seed, workers, print_freq, gpu, input_channels, use_checkpoint, checkpoint_directory, lr, momentum,
                     weight_decay, epochs, batch_size, features, knn_graph, labels, k_at_hop, active_connection):
    """
    Function that creates arguments object that is needed for calling the testing method of the GCN clustering

    Keyword arguments:
    seed -- seed used during training
    workers -- number of workers employed during testing
    print_freq -- batch print frequency of GCN output
    gpu -- gpu used for testing
    input_channels -- dimension of feature dataset that is used for input (needed for proper compression)
    use_checkpoint -- boolean whether to load up a checkpoint file instead of using the passed training dictionary
    checkpoint_directory -- directory where checkpoints are located in
    lr -- learning rate during testing
    momentum -- momentum during testing
    weight_decay -- weight decay during testing
    epochs -- number of epochs for testing
    batch_size -- size of batch for testing steps
    features -- feature dataset
    knn_graph -- knn graph dataset
    labels -- label dataset
    k-at-hop -- k-at-hop employed for instance pivot graph creation
    active_connection -- active connection employed for instance pivot graph creation
    """
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--workers', default=workers, type=int)
    parser.add_argument('--print_freq', default=print_freq, type=int)
    parser.add_argument('--gpu', default=gpu, type=str)
    parser.add_argument('--input_channels', type=int, default=input_channels)
    parser.add_argument('--use_checkpoint', default=use_checkpoint, type=bool)
    parser.add_argument('--checkpoint_directory', type=str, metavar='PATH', default=checkpoint_directory)

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

    args = parser.parse_args()
    return args


def create_prediction_output_file(metadata, prediction):
    output = pd.DataFrame()
    output_pred = np.full(len(metadata['idx']), -1)

    j = 0
    for i in metadata['idx']:
        if metadata.loc[i]['is_included']:
            output_pred[i] = prediction[j]
            j += 1

    output['frame'] = metadata['frame']
    output['pred'] = output_pred
    output['x1'] = metadata['det_x1']
    output['y1'] = metadata['det_y1']
    output['x2'] = metadata['det_x1'] + metadata['det_w']
    output['y2'] = metadata['det_y1'] + metadata['det_h']

    return output.to_numpy()


def adjust_labeling(label_file):
    """
    Function that adjust the labeling of a labels file. Needed to avoid indexing errors during training of the GCN.
    Substates the original labeling with a continuous one from 0 to num_labels in labels file. Does so by ordering the
    labels in increasing order and creating a dictionary with its new labels.
    """
    print("Adjusting Labels...")
    # obtain list of all labels occurring in label file
    unique_labels = np.unique(label_file)
    # create an array from 0 to num_unique_labels (new labels)
    new_labels = np.arange(len(unique_labels))
    # create a dict from the two arrays and use it to substitute labels
    label_dict = dict(zip(unique_labels, new_labels))
    relabel_dict = dict(zip(new_labels, unique_labels))
    new_labels = np.array([label_dict[x] for x in label_file])

    return new_labels
