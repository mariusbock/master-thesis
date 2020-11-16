###################################################################
# File Name: misc.py
# Author: Marius Bock
# mail: marius.bock@protonmail.com
###################################################################

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets
from gcn_clustering.feeder.feeder import VisualizationFeeder
from gcn_clustering.utils import to_numpy
from gcn_clustering.utils.osutils import mkdir_if_missing


def create_train_args(seed, workers, print_freq, gpu, log_directory, lr, momentum, weight_decay, epochs, batch_size,
                      k_at_hop, active_connection, element_wise_products_feeder, element_wise_products_type,
                      absolute_differences, normalise_distances):
    """
    Function that creates arguments object that contains parameters for the training method of the GCN clustering

    Args:
        seed -- seed used during training
        workers -- number of workers employed during training
        print_freq -- batch print frequency of GCN output
        gpu -- gpu used for training
        log_directory -- directory where to save checkpoints to
        lr -- learning rate during training
        momentum -- momentum during training
        weight_decay -- weight decay during training
        epochs -- number of epochs during training
        batch_size -- size of batch for training steps
        k-at-hop -- k-at-hop employed for instance pivot graph creation
        active_connection -- active connection employed for instance pivot graph creation
        element_wise_products_feeder -- boolean indicating whether or not to calculate element-wise products of features
                                        within the Feeder
        element_wise_products_type -- type of element-wise products of features that are to be calculated
                                      ('pairwise' or  'frame_pairwise')
        absolute_differences -- boolean indicating whether to use absolute differences within the feeder when pivot
                                instance is substracted from each of its neighbor nodes within IPS
        normalise_distances -- boolean indicating whether to normalise the distances between bounding boxes

    Returns:
        Training Args object
    """
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--seed', default=seed)
    parser.add_argument('--workers', default=workers, type=int)
    parser.add_argument('--print_freq', default=print_freq, type=int)
    parser.add_argument('--gpu', default=gpu, type=str)
    parser.add_argument('--log_directory', type=str, metavar='PATH', default=log_directory)

    # Optimization args
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--epochs', type=int, default=epochs)

    # Training args
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=k_at_hop)
    parser.add_argument('--active_connection', type=int, default=active_connection)
    parser.add_argument('--element_wise_products_feeder', type=bool, default=element_wise_products_feeder)
    parser.add_argument('--element_wise_products_type', type=str, default=element_wise_products_type)
    parser.add_argument('--absolute_differences', type=bool, default=absolute_differences)
    parser.add_argument('--normalise_distances', type=bool, default=normalise_distances)

    args = parser.parse_args()
    return args


def create_test_args(seed, workers, print_freq, gpu, log_directory, batch_size, k_at_hop, active_connection,
                     element_wise_products_feeder, element_wise_products_type, absolute_differences,
                     normalise_distances):
    """
    Function that creates arguments object that contains parameters for the testing method of the GCN clustering

    Args:
        seed -- seed used during testing
        workers -- number of workers employed during testing
        print_freq -- batch print frequency of GCN output
        gpu -- gpu used for testing
        log_directory -- directory where to save checkpoints to
        batch_size -- size of batch for testing steps
        k-at-hop -- k-at-hop employed for instance pivot graph creation
        active_connection -- active connection employed for instance pivot graph creation
        element_wise_products_feeder -- boolean indicating whether or not to calculate correlation features within the Feeder
        element_wise_products_type -- type of correlation features that are to be calculated ('pairwise' or  'frame_pairwise')
        absolute_differences -- boolean indicating whether to use absolute differences within the feeder when pivot
                                instance is substracted from each of its neighbor nodes within IPS
        normalise_distances -- boolean indicating whether to normalise the distances between bounding boxes

    Returns:
        Testing (Validation) Args object
    """
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--seed', default=seed)
    parser.add_argument('--workers', default=workers, type=int)
    parser.add_argument('--print_freq', default=print_freq, type=int)
    parser.add_argument('--gpu', default=gpu, type=str)
    parser.add_argument('--log_directory', type=str, metavar='PATH', default=log_directory)

    # Testing args
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=k_at_hop)
    parser.add_argument('--active_connection', type=int, default=active_connection)
    parser.add_argument('--element_wise_products_feeder', type=bool, default=element_wise_products_feeder)
    parser.add_argument('--element_wise_products_type', type=str, default=element_wise_products_type)
    parser.add_argument('--absolute_differences', type=bool, default=absolute_differences)
    parser.add_argument('--normalise_distances', type=bool, default=normalise_distances)

    args = parser.parse_args()
    return args


def create_validation_output_file(metadata, prediction, filter_to_use, removed_filter):
    """
    Function that creates a output file from a GCN prediction that follows the format for the input used for the
    postprocessing_script.m located in the eval folder.

    Args:
        metadata -- metadata file that was used during prediciton
        prediction -- prediction output of the GCN
        filter_to_use -- filter column that was used during prediction
        removed_filter -- boolean mask that filters array to only contain instances that were not excluded during
                          singleton cluster removal
    """
    output = pd.DataFrame()
    output_pred = np.full(len(metadata['idx']), -1)

    if filter_to_use is not None:
        j = 0
        for i in metadata['idx']:
            if metadata.loc[i][filter_to_use]:
                output_pred[i] = prediction[j]
                j += 1
    else:
        output_pred = prediction
    output['idx'] = metadata['idx']
    output['frame'] = metadata['frame']
    output['pred'] = output_pred
    output['x1'] = metadata['det_x1']
    output['y1'] = metadata['det_y1']
    output['w'] = metadata['det_w']
    output['h'] = metadata['det_h']

    output = output.to_numpy()
    if removed_filter is not None:
        rm_filter_mask = np.ones(len(metadata['idx']), np.bool)
        rm_filter_mask[removed_filter - 1] = 0
        removed_instances = output[rm_filter_mask]
        removed_instances[:, 2] = -1
        kept_instances = output[~rm_filter_mask]
        output_removed = np.concatenate((kept_instances, removed_instances))
        output_removed = output_removed[np.argsort(output_removed[:, 0])]
        output_removed = np.delete(output_removed, 0, 1)
        output = np.delete(output, 0, 1)
        return output, output_removed
    else:
        output = np.delete(output, 0, 1)
        return output, None


def create_testing_output_file(detection_file, prediction):
    """
    Function that creates a testing output file from a GCN prediction that follows the format for the input used for the
    postprocessing_script.m located in the eval folder.

    Args:
        detection_file -- detection file that was used during testing
        prediction -- prediction output of the GCN

    Returns:
        Eval input file for test sequence
    """
    output = np.empty((len(detection_file), 6))

    output[:, 0] = detection_file[:, 0]
    output[:, 1] = prediction
    output[:, 2] = detection_file[:, 2]
    output[:, 3] = detection_file[:, 3]
    output[:, 4] = detection_file[:, 4]
    output[:, 5] = detection_file[:, 5]

    return output


def create_heurisitc_output_file(output_dir, det_file, edges, scores, add_edges):
    """
    Function that creates a input file for the graph heurisitic as proposed by Keuper et al. (see thesis for details).

    Args:
        output_dir -- output directory to save file to
        det_file -- detection file which file is based on
        edges --  edges predicted by GCN
        scores -- scores assigned to edges predicted by GCN
        add_edges -- boolean whether to add dummy edges to file
                     (i.e. 0 confidence edges for non-overlapping bounding boxes)
    """
    num_nodes = max(max(edges[:, 0]), max(edges[:, 1])) + 1
    num_edges = len(edges)
    output = np.empty((num_edges, 3))
    output[:, :2] = edges.astype(int)
    output[:, 2] = scores
    np.savetxt(os.path.join(output_dir, 'heuristic_input.txt'), output, delimiter=' ', fmt=['%i', '%i', '%f'])

    def line_prepender(filename, line):
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(line.rstrip('\r\n') + '\n' + content)

    line_prepender(os.path.join(output_dir, 'heuristic_input.txt'), str(num_edges))
    line_prepender(os.path.join(output_dir, 'heuristic_input.txt'), str(num_nodes))
    if add_edges:
        add_dummy_edges(output_dir, det_file)


def add_dummy_edges(output_dir, det_file):
    """
    Function which appends dummy edges to a heuristic_input.txt file in a given output directory. A dummy edge is a
    zero confidence edge between detections in the same frame which do not have any overlap (IoU)

    Args:
        output_dir -- output directory where file is located
        det_file -- detection file which file is based on
    """
    output_det = np.empty((det_file.shape[0], det_file.shape[1] + 1))
    output_det[:, 0] = range(det_file.shape[0])
    output_det[:, 1:] = det_file
    output = []
    for i, (det_line) in enumerate(output_det):
        fil_dets = output_det[output_det[:, 1] == det_line[1]]
        det_bbox = [det_line[3], det_line[4], det_line[3] + det_line[5], det_line[4] + det_line[6]]
        for fil_det in fil_dets:
            fil_bbox = [fil_det[3], fil_det[4], fil_det[3] + fil_det[5], fil_det[4] + fil_det[6]]
            if iou(det_bbox, fil_bbox) == 0:
                output.append((int(det_line[0]), int(fil_det[0]), 0.0))
        # print progression of for loop
        if i % 2500 == 0:
            print("Detection processed: " + str(i) + "/" + str(len(output_det)))

    with open(os.path.join(output_dir, 'heuristic_input.txt'), "a") as heuristic_file:
        for pair in output:
            heuristic_file.write(str(pair[0]) + ' ' + str(pair[1]) + ' ' + str(pair[2]) + '\n')


def create_modified_gt_file(gt_file, meta_file, filter_to_use):
    """
    Function that creates a modified version of a ground truth file where each label in the ground truth that was not
    used in the detection file is substituted with the label of 'static person' (i.e. 7) so that the ground truth
    detections are ignored during evaluation.

    Args:
        gt_file -- ground truth file to be modified
        meta_file -- metadata file that evaluation is based upon
        filter_to_use -- filter column of the metadata file that was employed during prediction

    Returns:
        Modified Ground Truth file
    """
    output_gt = gt_file
    if filter_to_use is not None:
        filtered_meta = meta_file.to_numpy()[meta_file[filter_to_use]]
    else:
        filtered_meta = meta_file.to_numpy()

    # obtain all ground truth labels that were used in metadata file
    unique_labels_used = np.unique(filtered_meta[:, 7])
    # go through each ground truth line
    for i, (gt_line) in enumerate(gt_file):
        # if the label of the ground truth detection is not in the used ones then assign it the static pedestrian label
        if gt_line[1] not in unique_labels_used:
            output_gt[i, 7] = 7
    return output_gt


def adjust_labeling(label_file):
    """
    Function that adjust the labeling of a labels file. Needed to avoid indexing errors during training of the GCN.
    Substates the original labeling with a continuous one from 0 to num_labels in labels file. Does so by ordering the
    labels in increasing order and creating a dictionary with its new labels.

    Args:
        label_file -- label file to be modified

    Returns:
        List of adjusted labels
    """
    print("Adjusting Labels...")
    # obtain list of all labels occurring in label file
    unique_labels = np.unique(label_file)
    # create an array from 0 to num_unique_labels (new labels)
    new_labels = np.arange(len(unique_labels))
    # create a dict from the two arrays and use it to substitute labels
    label_dict = dict(zip(unique_labels, new_labels))
    # relabel_dict = dict(zip(new_labels, unique_labels))
    new_labels = np.array([label_dict[x] for x in label_file])

    return new_labels


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    Source: https://discuss.pytorch.org/t/dataloader-filenames-in-each-batch/4212/2

    Returns:
        Tuple with image file path
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
    Source: https://github.com/bochinski/iou-tracker -> utils.py

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


def split(a, n):
    """
    Function that splits a list a into two equal n (somewhat) equal parts

    Args:
        a -- array to be split
        n -- number of parts the array is to be split in

    Returns:
        n (somewhat) equal length parts of the list
    """
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


def plot_knn_graph(index, run, detector_name, detector_type, log_directory, labels, features, knn, frames, k_at_hop,
                   active_connection, seed, element_wise_products_feeder, element_wise_products_type, absolute_differences,
                   normalise_distances):
    """
    Function that creates a plot of the knn graph for a given instance. Saves plots into log_directory/plots.
    Modified version of function in visualization.ipynb within Wang et al. repository.

    Args:
        index -- detection index which to plot graph for
        run -- number of current run; used for proper saving
        detector_name -- name of detector that is plotted
        detector_type -- type of detector (DPM, FRCNN or SDP)
        log_directory -- directory where log files are saved to
        labels -- labels array to use
        features -- features array to use
        knn -- knn graph array to use
        frames -- frames array of detections (used for frame distance calculation)
        k_at_hop -- k1, k2, ... used during knn graph calculation
        active_connection -- parameter u during knn graph calculation
        seed -- seed to use
        element_wise_products_feeder -- boolean whether to calculate element-wise products in feeder
        element_wise_products_type -- type of element-wise products in feeder calculated
        absolute_differences -- boolean whether to calculate absolute differences in feeder
        normalise_distances -- boolean whether to normalise distances in feeder
    """
    print('\n Creating KNN Graph Plots')

    dataset = VisualizationFeeder(features,
                                  knn,
                                  labels,
                                  seed=seed,
                                  k_at_hop=k_at_hop,
                                  active_connection=active_connection,
                                  element_wise_products_feeder=element_wise_products_feeder,
                                  element_wise_products_type=element_wise_products_type,
                                  absolute_differences=absolute_differences,
                                  normalise_distances=normalise_distances
                                  )

    if index == 'half':
        new_index = int(np.ceil(len(features) / 2))
        print('Index: ')
        print(new_index)
        (feat, A, cid, one_hop_idcs), edge_labels, labels = dataset[new_index]
    else:
        new_index = index
        print('Index: ')
        print(new_index)
        (feat, A, cid, one_hop_idcs), edge_labels, labels = dataset[new_index]

    feat, A, cid, one_hop_idcs, edge_labels, labels = map(to_numpy,
                                                          (feat, A, cid, one_hop_idcs, edge_labels, labels))


    ulabel = np.unique(labels)
    lmap = {j: i for i, j in enumerate(ulabel)}
    labels = [lmap[l] for l in labels]
    labels = np.asarray(labels)

    # ANALYSIS PARAMETERS
    no_instances = len(labels)
    same_label = 0
    different_label = 0
    one_hop_same_label = 0
    one_hop_different_label = 0

    tsne_features = TSNE(n_components=2).fit_transform(feat)
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(30, 10)
    for i in range(labels.shape[0]):
        if i == cid:
            c = 'g'
            s = 500
        elif labels[i] == labels[cid]:
            c = 'r'
            s = 20
            same_label += 1
        else:
            c = 'b'
            s = 20
            different_label += 1
        axs[0].scatter(tsne_features[i, 0], tsne_features[i, 1], s=s, c=c)

    for one_hop_index in one_hop_idcs:
        if labels[cid] == labels[one_hop_index]:
            c = 'r'
            one_hop_same_label += 1
        else:
            c = 'gray'
            one_hop_different_label += 1
        if labels[cid] == labels[one_hop_index]:
            w = 1
        else:
            w = 0.5

        axs[0].plot([tsne_features[cid, 0], tsne_features[one_hop_index, 0]],
                    [tsne_features[cid, 1], tsne_features[one_hop_index, 1]],
                    linestyle='--', linewidth=w, color=c)

    x = []
    y = []
    c = []
    for i in range(labels.shape[0]):
        if i != cid:
            x.append(tsne_features[i, 0])
            y.append(tsne_features[i, 1])
            c.append((frames[i] - frames[cid[0]]))
    c_one_hop = []
    for one_hop_index in one_hop_idcs:
        if one_hop_index != cid:
            c_one_hop.append(frames[one_hop_index] - frames[cid[0]])
    print("FRAME DISTANCES: \n")
    print(c)
    average_frame_distance = np.average(c)
    print("FRAME DISTANCES ONE HOP: \n")
    print(c_one_hop)
    one_hop_average_frame_distance = np.average(c_one_hop)

    im = axs[1].scatter(x, y, cmap='viridis', c=c)
    axs[1].scatter(tsne_features[cid, 0], tsne_features[cid, 1], s=500, c='g')

    fig.colorbar(im, ax=axs[1])

    print('\n ANALYSIS SUMMARY:\n')
    print('NO. NODES: ' + str(no_instances))
    print('NO. NODES SAME LABEL: ' + str(same_label))
    print('NO. NODES DIFFERENT LABEL: ' + str(different_label))
    print('AVERAGE FRAME DISTANCE: ' + str(average_frame_distance))
    print('-----------------------------------------')
    print('NO. NODES SAME LABEL (ONE-HOP): ' + str(one_hop_same_label))
    print('NO. NODES DIFFERENT LABEL (ONE-HOP): ' + str(one_hop_different_label))
    print('AVERAGE FRAME DISTANCE (ONE-HOP): ' + str(one_hop_average_frame_distance))

    mkdir_if_missing(os.path.join(log_directory, 'run_' + str(run), 'knn_graphs', 'idx_' + str(index)))
    if detector_name is None:
        plt.savefig(
            os.path.join(log_directory, 'run_' + str(run), 'knn_graphs', 'idx_' + str(index), detector_type + '.jpg'))
    else:
        plt.savefig(os.path.join(log_directory, 'run_' + str(run), 'knn_graphs', 'idx_' + str(index), detector_name +
                                 '_' + detector_type + '.jpg'))
    plt.close()


def plot_embedding_graph(index, run, detector_name, detector_type, log_directory, labels, features, knn, frames,
                         k_at_hop, active_connection, seed, element_wise_products_feeder, element_wise_products_type,
                         absolute_differences, normalise_distances):
    """
    Function that creates a plot of the embedding graph for a given instance. Saves plots into log_directory/plots.
    Modified version of function in visualization.ipynb within Wang et al. repository.

    Args:
        index -- detection index which to plot graph for
        run -- number of current run; used for proper saving
        detector_name -- name of detector that is plotted
        detector_type -- type of detector (DPM, FRCNN or SDP)
        log_directory -- directory where log files are saved to
        labels -- labels array to use
        features -- features array to use
        knn -- knn graph array to use
        frames -- frames array of detections (used for frame distance calculation)
        k_at_hop -- k1, k2, ... used during knn graph calculation
        active_connection -- parameter u during knn graph calculation
        seed -- seed to use
        element_wise_products_feeder -- boolean whether to calculate element-wise products in feeder
        element_wise_products_type -- type of element-wise products in feeder calculated
        absolute_differences -- boolean whether to calculate absolute differences in feeder
        normalise_distances -- boolean whether to normalise distances in feeder
    """
    print('\n Creating Embedding Plots')

    dataset = VisualizationFeeder(features,
                                  knn,
                                  labels,
                                  seed=seed,
                                  k_at_hop=k_at_hop,
                                  active_connection=active_connection,
                                  element_wise_products_feeder=element_wise_products_feeder,
                                  element_wise_products_type=element_wise_products_type,
                                  absolute_differences=absolute_differences,
                                  normalise_distances=normalise_distances
                                  )
    if index == 'half':
        new_index = int(np.ceil(len(features) / 2))
        print('Index: ')
        print(new_index)
        (feat, A, cid, one_hop_idcs), edge_labels, labels = dataset[new_index]
    else:
        new_index = index
        print('Index: ')
        print(new_index)
        (feat, A, cid, one_hop_idcs), edge_labels, labels = dataset[new_index]

    feat, A, cid, one_hop_idcs, edge_labels, labels = map(to_numpy,
                                                          (feat, A, cid, one_hop_idcs, edge_labels, labels))

    ulabel = np.unique(labels)
    lmap = {j: i for i, j in enumerate(ulabel)}
    labels = [lmap[l] for l in labels]
    labels = np.asarray(labels)

    tsne_features = TSNE(n_components=2).fit_transform(feat)
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(30, 10)

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
        axs[0].scatter(tsne_features[i, 0], tsne_features[i, 1], s, color=c)

    edges = A.nonzero()
    edges = np.asarray(edges).T
    for e in edges:
        axs[0].plot([tsne_features[e[0], 0], tsne_features[e[1], 0]],
                    [tsne_features[e[0], 1], tsne_features[e[1], 1]], linestyle='--', linewidth=0.5,
                    color='gray')

    x = []
    y = []
    c = []
    for i in range(labels.shape[0]):
        if i != cid:
            x.append(tsne_features[i, 0])
            y.append(tsne_features[i, 1])
            c.append((frames[i] - frames[cid[0]]))
    c_one_hop = []
    for one_hop_index in one_hop_idcs:
        if one_hop_index != cid:
            c_one_hop.append(frames[one_hop_index] - frames[cid[0]])

    im = axs[1].scatter(x, y, cmap='viridis', c=c)
    axs[1].scatter(tsne_features[cid, 0], tsne_features[cid, 1], s=500, c='g')

    fig.colorbar(im, ax=axs[1])
    mkdir_if_missing(os.path.join(log_directory, 'run_' + str(run), 'embeddings', 'idx_' + str(index)))
    if detector_name is None:
        plt.savefig(
            os.path.join(log_directory, 'run_' + str(run), 'embeddings', 'idx_' + str(index), detector_type + '.jpg'))
    else:
        plt.savefig(os.path.join(log_directory, 'run_' + str(run), 'embeddings', 'idx_' + str(index), detector_name +
                                 '_' + detector_type + '.jpg'))
    plt.close()
