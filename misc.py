import argparse
import numpy as np
import pandas as pd
from torchvision import datasets


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


def create_prediction_output_file(metadata, prediction, filter_to_use):
    output = pd.DataFrame()
    output_pred = np.full(len(metadata['idx']), -1)

    j = 0
    for i in metadata['idx']:
        if metadata.loc[i][filter_to_use]:
            output_pred[i] = prediction[j]
            j += 1

    output['frame'] = metadata['frame']
    output['pred'] = output_pred
    output['x1'] = metadata['det_x1']
    output['y1'] = metadata['det_y1']
    output['w'] = metadata['det_w']
    output['h'] = metadata['det_h']

    return output.to_numpy()


def create_modified_ground_truth_file(gt_file, meta_file, filter_to_use):
    filtered_meta = meta_file[meta_file[filter_to_use]]
    unique_labels_used = np.unique(filtered_meta['gt_labels'])
    for i, (gt_line) in enumerate(gt_file):
        if gt_line[1] not in unique_labels_used:
            gt_file[i, 7] = 7
    return gt_file


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
    #relabel_dict = dict(zip(new_labels, unique_labels))
    new_labels = np.array([label_dict[x] for x in label_file])

    return new_labels


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    Source: https://discuss.pytorch.org/t/dataloader-filenames-in-each-batch/4212/2
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union
