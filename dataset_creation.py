import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms, ops
from math import *
from decimal import Decimal

from networks.person.network_sn_101 import ACSPNet
from networks.appearance.resnet_modified import resnet50
from networks.reidentification.reidentification_models import ft_net
import misc

pd.options.mode.chained_assignment = None  # default='warn'

"""
This File contains all relevant functions for the creation of the three input files for the GCN as well as data
consistency checks.
"""


def create_meta_dataset(detection_file):
    """
    Function that creates a meta_dataset for a detection_file as a Pandas dataframe
    Initially saves ids, frames and bounding boxes of detections

    Args:
        detection_file -- detection file used to create meta_dataset

    Returns:
         Metadata file as pandas dataframe
    """
    # create dataframe and add index, frame and bounding box infos of detections
    metadata_file = pd.DataFrame()
    metadata_file['idx'] = detection_file[:, 0]
    metadata_file['frame'] = detection_file[:, 1]
    metadata_file['det_x1'] = detection_file[:, 3]
    metadata_file['det_y1'] = detection_file[:, 4]
    metadata_file['det_w'] = detection_file[:, 5]
    metadata_file['det_h'] = detection_file[:, 6]
    return metadata_file


def create_modified_detection_file(detection_file):
    """
    Modifies a detection file by adding an id column as first column. ID is needed to identify detection in
    subsequent steps.

    Args:
        detection_file -- detection file used to modified detection file

    Returns:
         Modified detection file as numpy array with appended index column
    """
    print("Creating Modified Detection File...")
    # create emtpy dataframe with one added column
    output_det = np.empty((detection_file.shape[0], detection_file.shape[1] + 1))
    # fill dataframe with index column and rest of data
    output_det[:, 0] = range(detection_file.shape[0])
    output_det[:, 1:] = detection_file

    return output_det


def create_label_dataset(detection_file, metadata_file, ground_truth_file, iou_ranges, background_handling):
    """
    Function that creates a dataset containing the detection id and its corresponding assigned label. The labels
    are obtained by calculating the iou (intersection over union) between each detection and ground truth box and saving
    the largest iou for each detection. Thresholds are defined via the split_iou (which itself states the range in which
    the iou should be considered a non-match i.e. neither a class nor background). Further, the function saves
    meta information i.e. filter arrays to obtain excluded, background and class detections, as well as the maximum iou
    for each detection and the corresponding ground truth bounding box.

    Args:
        detection_file -- detection file used to create label dataset
        metadata_file -- metadata file where results are saved to
        ground_truth_file -- ground truth file of detection
        iou_ranges -- list of iou ranges to employ for label calculation (list of pairs (upper and lower threshold))
        background_handling --

    Returns:
        Modified metadata file that contains assigned labels as well as corresponding ground truth bounding boxes for
        each detection, and filter arrays to obtain labeling according to different iou thresholds
    """
    print("Creating Label Dataset...")
    # no classes (i.e. max label) in ground truth file
    no_classes = max([item[1] for item in ground_truth_file])

    # information of matched ground truth bounding box (all -1 if matched with no box)
    metadata_file['iou'] = -1.0
    metadata_file['gt_labels'] = -1
    metadata_file['gt_x1'] = -1
    metadata_file['gt_y1'] = -1
    metadata_file['gt_w'] = -1
    metadata_file['gt_h'] = -1
    metadata_file['background_handling'] = background_handling

    # create columns for each split_iou employed
    for iou in iou_ranges:
        metadata_file['labels_' + str(iou[0]) + "_" + str(iou[1])] = -1
        metadata_file['is_class_' + str(iou[0]) + "_" + str(iou[1])] = False
        metadata_file['is_background_' + str(iou[0]) + "_" + str(iou[1])] = False
        metadata_file['is_nothing_' + str(iou[0]) + "_" + str(iou[1])] = True

    # main for loop for iterating through each detection
    for i, (det_line) in enumerate(detection_file):
        # only compare detection to ground truth bounding boxes from same frame which are not occluded (i.e. visible)
        filtered_ground_truth = [line for line in ground_truth_file if line[0] == det_line[1] and line[7] != 0]
        # save detection bounding box into array
        det_bbox = [det_line[3], det_line[4], det_line[3] + det_line[5], det_line[4] + det_line[6]]
        # for loop for iterating through ground truth lines
        for gt_line in filtered_ground_truth:
            # save ground truth bounding box into array
            gt_bbox = [gt_line[2], gt_line[3], gt_line[2] + gt_line[4], gt_line[3] + gt_line[5]]
            # calculate iou between detection and ground truth bbox
            curr_iou = misc.iou(det_bbox, gt_bbox)
            # if iou is larger than any other previous observed iou (iterations.e. compare to one saved in iou_array)
            if curr_iou > metadata_file.loc[i]['iou'] and curr_iou > 0:
                # then make current iou new largest one and update iou_array as well as gt_bbox arrays
                metadata_file.at[i, 'iou'] = curr_iou
                metadata_file.at[i, 'gt_labels'] = gt_line[1]
                metadata_file.at[i, 'gt_x1'] = gt_line[2]
                metadata_file.at[i, 'gt_y1'] = gt_line[3]
                metadata_file.at[i, 'gt_w'] = gt_line[4]
                metadata_file.at[i, 'gt_h'] = gt_line[5]
                # loop over all iou ranges for which detection is to be evaluated
                for iou in iou_ranges:
                    # create new columns for current iou threshold
                    label_column = 'labels_' + str(iou[0]) + "_" + str(iou[1])
                    cl_column = 'is_class_' + str(iou[0]) + "_" + str(iou[1])
                    bg_column = 'is_background_' + str(iou[0]) + "_" + str(iou[1])
                    no_column = 'is_nothing_' + str(iou[0]) + "_" + str(iou[1])
                    # if the iou is larger than the upper bound of the split_iou assign it the class of the gt
                    if curr_iou > iou[0]:
                        metadata_file.at[i, label_column] = gt_line[1]
                        metadata_file.at[i, cl_column] = True
                        metadata_file.at[i, bg_column] = False
                        metadata_file.at[i, no_column] = False
                    # if the iou is within the split_iou exclude the detection by updating the filter arrays
                    elif iou[0] >= curr_iou >= iou[1]:
                        # if not background nor class, detection is assigned -1
                        metadata_file.at[i, label_column] = -1
                        metadata_file.at[i, cl_column] = False
                        metadata_file.at[i, bg_column] = False
                        metadata_file.at[i, no_column] = True
                    # if the iou is smaller than the lower bound of the split_iou it is background
                    # also update filter arrays
                    elif iou[1] > curr_iou > 0:
                        # depending on which method is to be used for background the detection is either assigned
                        # a zero or a new class (by increasing no_instances)
                        if background_handling == "clusters":
                            metadata_file.at[i, label_column] = 0
                        elif background_handling == "singletons":
                            # increase max label value by one and assign it to detection
                            no_classes += 1
                            metadata_file.at[i, label_column] = no_classes
                        metadata_file.at[i, cl_column] = False
                        metadata_file.at[i, bg_column] = True
                        metadata_file.at[i, no_column] = False
            # if the iou is not larger than any other previous one, then skip to next ground truth detection
            else:
                continue
        # print progression of for loop
        if i % 2500 == 0:
            print("Detection processed: " + str(i) + "/" + str(len(detection_file)))
    return metadata_file


def create_three_split(metadata_file, split_ratio, split_iou):
    """
    Function that creates (approx.) evenly distributed train, valid and test datasets for a given labels_file
    It does so by handling classes and background separately to ensure that each dataset is assigned equal amounts
    of background observations. The resulting datasets are translated into filter arrays, which are appended to the
    metadata_file.
    Currently only the splitting by ID is supported.

    Args:
        metadata_file -- metadata file used for splitting
        split_ratio -- split ratio employed for creating train, valid and test set (needs to be array of size 3)
        split_iou -- iou used to identify which column to use for splitting; pair [upper, lower]

    Returns:
        Metadata file with appended boolean columns for the split
    """
    # create filter array columns in metadata_file
    train_column = 'is_train_two_split_' + split_iou[0] + '_' + split_iou[1]
    valid_column = 'is_valid_two_split_' + split_iou[0] + '_' + split_iou[1]
    test_column = 'is_test_two_split_' + split_iou[0] + '_' + split_iou[1]
    metadata_file[train_column] = False
    metadata_file[valid_column] = False
    metadata_file[test_column] = False

    # split metadata file into records that are assigned a class, background or nothing
    background = metadata_file[metadata_file['is_background_' + str(split_iou[0]) + "_" + str(split_iou[1])]]
    classes = metadata_file[metadata_file['is_class_' + str(split_iou[0]) + "_" + str(split_iou[1])]]
    removed = metadata_file[metadata_file['is_nothing_' + str(split_iou[0]) + "_" + str(split_iou[1])]]

    print("TOTAL BACKGROUND DETECTIONS:")
    print(background.shape)
    print("TOTAL CLASSES DETECTIONS:")
    print(classes.shape)
    print("REMOVED DETECTIONS:")
    print(removed.shape)

    # Currently only split by ID is supported. Idea is to split classes and background independently
    # This ensures an even distribution. The code splits the classes and background attribute above into train, valid
    # and test via GroupShuffleSplit, ShuffleSplit and indexing.
    # create idx for classes
    # TODO: add choice to be able to split by frame (add parameter split by)
    split_col = "labels_" + str(split_iou[0]) + "_" + str(split_iou[1])
    train_idx, val_test_idx = next(GroupShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7)
                                   .split(classes, groups=classes[split_col]))

    val_idx, test_idx = next(GroupShuffleSplit(test_size=split_ratio[1] / (split_ratio[1] + split_ratio[2]),
                                               n_splits=1, random_state=7)
                             .split(classes.iloc[val_test_idx, :],
                                    groups=classes.iloc[val_test_idx, :][split_col]))

    # split classes attribute into train, valid and test using indeces obtained above
    train_cl = classes.iloc[train_idx, :]
    train_cl[train_column] = True
    val_cl = classes.iloc[val_test_idx, :].iloc[val_idx, :]
    val_cl[valid_column] = True
    test_cl = classes.iloc[val_test_idx, :].iloc[test_idx, :]
    test_cl[test_column] = True

    # create idx for background
    try:
        train_bg_idx, val_test_bg_idx = next(
            ShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7).split(background))
        val_bg_idx, test_bg_idx = next(
            ShuffleSplit(test_size=split_ratio[1] / (split_ratio[1] + split_ratio[2]), n_splits=1,
                         random_state=7).split(background.iloc[val_test_bg_idx, :]))
        # split background attribute into train, valid and test using indeces obtained above
        train_bg = background.iloc[train_bg_idx, :]
        train_bg[train_column] = True
        val_bg = background.iloc[val_test_bg_idx, :].iloc[val_bg_idx, :]
        val_bg[valid_column] = True
        test_bg = background.iloc[val_test_bg_idx, :].iloc[test_bg_idx, :]
        test_bg[test_column] = True
    # this exception is needed if the background detections are very small (which will cause the ShuffleSplit to
    # not work. If this is the case, the background is just split equally and not using ratios.
    except ValueError:
        train_bg, val_bg, test_bg = np.array_split(background, 3)
        train_bg[train_column] = True
        val_bg[valid_column] = True
        test_bg[test_column] = True

    # create idx for other detections
    try:
        train_ex_idx, val_test_ex_idx = next(
            ShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7).split(removed))
        val_ex_idx, test_ex_idx = next(
            ShuffleSplit(test_size=split_ratio[1] / (split_ratio[1] + split_ratio[2]), n_splits=1,
                         random_state=7).split(removed.iloc[val_test_ex_idx, :]))
        # split removed attribute into train, valid and test using indeces obtained above
        train_ex = removed.iloc[train_ex_idx, :]
        train_ex[train_column] = True
        val_ex = removed.iloc[val_test_ex_idx, :].iloc[val_ex_idx, :]
        val_ex[valid_column] = True
        test_ex = removed.iloc[val_test_ex_idx, :].iloc[test_ex_idx, :]
        test_ex[test_column] = True
    # this exception is needed if the background detections are very small (which will cause the ShuffleSplit to
    # not work. If this is the case, the background is just split equally and not using ratios.
    except ValueError:
        train_ex, val_ex, test_ex = np.array_split(removed, 3)
        train_ex[train_column] = True
        val_ex[valid_column] = True
        test_ex[test_column] = True

    # reconstruct metadata file
    metadata_file = pd.concat(
        [train_cl, val_cl, test_cl, train_bg, val_bg, test_bg, train_ex, val_ex, test_ex]).sort_values('idx')

    return metadata_file


def create_two_split(metadata_file, split_ratio, split_iou):
    """
    Function that creates (approx.) evenly distributed train and valid datasets for a given labels_file
    It does so by handling classes and background separately to ensure that each dataset is assigned equal amounts
    of background observations. The resulting datasets are translated into filter arrays, which are appended to the
    metadata_file.
    Currently only the splitting by ID is supported.

    Args:
        metadata_file -- metadata file used to create splits
        split_ratio -- split ratio employed (needs to be array of size 2)
        split_iou -- iou used to identify which column to use for splitting

    Returns:
        Metadata file with appended boolean columns for the split
    """
    # create filter array columns in metadata_file
    train_column = 'is_train_two_split_' + split_iou[0] + '_' + split_iou[1]
    valid_column = 'is_valid_two_split_' + split_iou[0] + '_' + split_iou[1]
    metadata_file[train_column] = False
    metadata_file[valid_column] = False

    # split metadata file into records that are assigned a class, background or nothing
    background = metadata_file[metadata_file['is_background_' + str(split_iou[0]) + "_" + str(split_iou[1])]]
    classes = metadata_file[metadata_file['is_class_' + str(split_iou[0]) + "_" + str(split_iou[1])]]
    removed = metadata_file[metadata_file['is_nothing_' + str(split_iou[0]) + "_" + str(split_iou[1])]]

    print("TOTAL BACKGROUND DETECTIONS:")
    print(background.shape)
    print("TOTAL CLASSES DETECTIONS:")
    print(classes.shape)
    print("REMOVED DETECTIONS:")
    print(removed.shape)

    # Currently only split by ID is supported. Idea is to split classes and background independently
    # This ensures a correct distribution. The code splits the classes and background attribute above into train, valid
    # and test via GroupShuffleSplit, ShuffleSplit and indexing.
    # create idx for classes
    # TODO: add choice to be able to split by frame (add parameter split by)
    split_column = "labels_" + str(split_iou[0]) + "_" + str(split_iou[1])
    train_idx, valid_idx = next(GroupShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7)
                                .split(classes, groups=classes[split_column]))

    # split classes attribute into train, valid and test using indeces obtained above
    train_cl = classes.iloc[train_idx, :]
    train_cl[train_column] = True
    val_cl = classes.iloc[valid_idx, :]
    val_cl[valid_column] = True

    # create idx for background
    try:
        train_bg_idx, val_bg_idx = next(
            ShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7).split(background))
        # split background attribute into train, valid and test using indeces obtained above
        train_bg = background.iloc[train_bg_idx, :]
        train_bg[train_column] = True
        val_bg = background.iloc[val_bg_idx, :]
        val_bg[valid_column] = True
    # this exception is needed if the background detections are very small (which will cause the ShuffleSplit to
    # not work. If this is the case, the background is just split equally and not using ratios.
    except ValueError:
        train_bg, val_bg = np.array_split(background, 2)
        train_bg[train_column] = True
        val_bg[valid_column] = True

    # create idx for other detections
    try:
        train_ex_idx, val_ex_idx = next(
            ShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7).split(removed))
        # split background attribute into train, valid and test using indeces obtained above
        train_ex = removed.iloc[train_ex_idx, :]
        train_ex[train_column] = True
        val_ex = removed.iloc[val_ex_idx, :]
        val_ex[valid_column] = True
    # this exception is needed if the background detections are very small (which will cause the ShuffleSplit to
    # not work. If this is the case, the background is just split equally and not using ratios.
    except ValueError:
        train_ex, val_ex = np.array_split(removed, 2)
        train_ex[train_column] = True
        val_ex[valid_column] = True

    # reconstruct metadata file
    metadata_file = pd.concat([train_cl, val_cl, train_bg, val_bg, train_ex, val_ex]).sort_values('idx')

    return metadata_file


def create_appearance_feature_dataset(detection_file, image_folder, batch_size, gpu_name, max_pool):
    """
    Function that creates for each detection a feature array using ResNet50 and RoI-Align.
    Each Image is feed into the resnet, which is modified to return the feature map before the first fc-layer as output.
    Once the feature map is obtained all detections from that frame are filtered from the detection file.
    Using torchvision's RoI-Align each detected bounding box is translated into a 2048x4x4 numpy array.
    If max_pool is true, the 2048x4x4 is max pooled to be 2048x1x1. The resulting array of previous steps
    is flattened and added to the output array in its corresponding place (determined by id of detection).

    Args:
        detection_file -- detection file to be used
        image_folder -- folder where images are located (should be located within a another folder)
        batch_size -- batch size used during feature creation
        gpu_name -- name of the gpu used for feature creation
        max_pool -- boolean whether to do max pooling for each cropped 4x4x2048 bbox so that it becomes 1x1x2048

    Returns:
        Appearance dataset as numpy array.
    """
    print("Creating Appearance Feature Dataset...")
    # apply preprocessing needed for resnet (will be further investigated if normalization is needed or should
    # be changed to be according to dataset used for detection)
    preprocess = transforms.Compose([
        # Need to cut that out since it messes with the bbox detections
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # load dataset using custom function that also returns the path of the image
    dataset = misc.ImageFolderWithPaths(root=image_folder, transform=preprocess)
    # define data loader (currently only batch size 1 is supported)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    # define ResNet model using modified file
    # define array that will hold output features
    resnet_model = resnet50(pretrained=True).to(gpu_name)
    if max_pool:
        output_features = np.empty((len(detection_file), 2048))
    else:
        output_features = np.empty((len(detection_file), (2048 * 4 * 4)))
    # iterate through data loader
    for i, (data, target, paths) in enumerate(train_loader):
        # define input batch
        input_batch = data.to(gpu_name)
        # obtain feature map by feeding batch into ResNet and send feature map tensor back to cpu
        feature_map = resnet_model(input_batch).detach().cpu()
        # obtain current frame by looking at path name of file being processed
        frame = float(paths[0].split("/")[-1].split(".")[0])

        # calculate the reduction factor of the resnet by looking at the dimensions of the input and output
        # of the resnet
        x_reduction_factor = input_batch.shape[3] / feature_map.shape[3]
        y_reduction_factor = input_batch.shape[2] / feature_map.shape[2]

        # filter detections to only obtain detections of that frame
        frame_detections = np.array([det for det in detection_file if det[1] == frame])
        if frame_detections.size == 0:
            continue
        # create bounding box array (need to have 0 at first position for roi_align function
        bboxes = np.empty((len(frame_detections), 5), dtype=np.float32)
        bboxes[:, 0] = np.zeros(len(frame_detections))
        bboxes[:, 1] = frame_detections[:, 3] / x_reduction_factor
        bboxes[:, 2] = frame_detections[:, 4] / y_reduction_factor
        # define x2 by adding 40% of the height of the bounding box to the x1 coordinate
        bboxes[:, 3] = (bboxes[:, 1] + frame_detections[:, 6] * 0.4) / x_reduction_factor
        bboxes[:, 4] = bboxes[:, 2] + frame_detections[:, 6] / y_reduction_factor
        # send bboxes to cpu as Torch Tensor
        bboxes = torch.from_numpy(bboxes).to('cpu')
        # apply roi_align algorithm on top of bboxes to obtain the 4x4x2048 cropped bboxes
        cropped_bboxes = ops.roi_align(feature_map, bboxes, (4, 4))
        # if pooling is wanted (iterations.e. only take highest value of each 4x4 array) then Pooling layer is applied
        if max_pool:
            pool = torch.nn.MaxPool3d((1, 4, 4))
            cropped_bboxes = pool(cropped_bboxes)
        # flatten each cropped bbox and assign it to its position in the output features array
        for j, cropped_bbox in enumerate(cropped_bboxes):
            output_features[int(frame_detections[j, 0]), :] = cropped_bbox.flatten()
        if i % 100 == 0:
            print('Batch [{0}/{1}]'.format(i, len(train_loader)))
    return output_features


def create_person_feature_dataset(detection_file, image_folder, batch_size, image_dimensions, gpu_name, max_pool):
    """
    Same as appearance feature creation just that we are employing a detector which is specifically suited for
    identifying persons instead of a ResNet50.
    Source: https://github.com/WangWenhao0716/Adapted-Center-and-Scale-Prediction

    Args:
        detection_file -- detection file to be used
        image_folder -- folder where images are located (should be located within a another folder)
        batch_size -- batch size used during feature creation
        image_dimensions -- initial image dimensions of the input images
        gpu_name -- name of the gpu used for feature creation
        max_pool -- boolean whether to do max pooling for each cropped 4x4x2048 bbox so that it becomes 1x1x2048

    Returns:
        Person feature dataset as numpy array.
    """

    print("Creating Person Feature Dataset...")
    # apply preprocessing needed for resnet (will be further investigated if normalization is needed or should
    # be changed to be according to dataset used for detection)
    padding = (image_dimensions[1] * 2 - image_dimensions[0]) / 2
    preprocess = transforms.Compose([
        # resize image (see paper why)
        transforms.Pad((int(padding), 0)),
        transforms.Resize((640, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # load dataset using custom function that also returns the path of the image
    dataset = misc.ImageFolderWithPaths(root=image_folder, transform=preprocess)
    # define data loader (currently only batch size 1 is supported)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    # model = ft_net(751)
    model = ACSPNet()
    model.eval()
    # model.load_state_dict(torch.load('pretrained_models/ft_net.pth'))
    model.load_state_dict(torch.load('./networks/pretrained_models/ACSP(Smooth L1).pth.tea'))
    model.to(gpu_name)

    if max_pool:
        output_features = np.empty((len(detection_file), 256))
    else:
        output_features = np.empty((len(detection_file), (256 * 4 * 4)))

    # iterate through data loader
    for i, (data, target, paths) in enumerate(train_loader):
        # define input batch
        input_batch = data.to(gpu_name)
        # obtain feature map by feeding batch into ResNet and send feature map tensor back to cpu
        feature_map = model(input_batch).detach().cpu()
        # obtain current frame by looking at path name of file being processed
        frame = float(paths[0].split("/")[-1].split(".")[0])

        # calculate the reduction factor of the resnet by looking at the dimensions of the input and output
        # of the resnet
        x_reduction_factor = input_batch.shape[3] / feature_map.shape[3]
        y_reduction_factor = input_batch.shape[2] / feature_map.shape[2]

        # filter detections to only obtain detections of that frame
        frame_detections = np.array([det for det in detection_file if det[1] == frame])
        if frame_detections.size == 0:
            continue
        # create bounding box array (need to have 0 at first position for roi_align function
        bboxes = np.empty((len(frame_detections), 5), dtype=np.float32)
        bboxes[:, 0] = np.zeros(len(frame_detections))
        bboxes[:, 1] = frame_detections[:, 3] / x_reduction_factor
        bboxes[:, 2] = frame_detections[:, 4] / y_reduction_factor
        # define x2 by adding 40% of the height of the bounding box to the x1 coordinate
        bboxes[:, 3] = (bboxes[:, 1] + frame_detections[:, 6] * 0.4) / x_reduction_factor
        bboxes[:, 4] = bboxes[:, 2] + frame_detections[:, 6] / y_reduction_factor
        # send bboxes to cpu as Torch Tensor
        bboxes = torch.from_numpy(bboxes).to('cpu')
        # apply roi_align algorithm on top of bboxes to obtain the 4x4x2048 cropped bboxes
        cropped_bboxes = ops.roi_align(feature_map, bboxes, (4, 4))
        # if pooling is wanted (iterations.e. only take highest value of each 4x4 array) then Pooling layer is applied
        if max_pool:
            pool = torch.nn.MaxPool3d((1, 4, 4))
            cropped_bboxes = pool(cropped_bboxes)
        # flatten each cropped bbox and assign it to its position in the output features array
        for j, cropped_bbox in enumerate(cropped_bboxes):
            output_features[int(frame_detections[j, 0]), :] = cropped_bbox.flatten()
        # print progression
        if i % 100 == 0:
            print('Batch [{0}/{1}]'.format(i, len(train_loader)))
    return output_features


def create_reid_feature_dataset(detection_file, image_folder, batch_size, gpu_name, max_pool):
    """
    Same as appearance function just that we are employing a detector which is specifically suited for identifying
    persons instead of a ResNet50. Currently set to be the normal ResNet-50.
    Source: https://github.com/layumi/Person_reID_baseline_pytorch

    Args:
        detection_file -- detection file to be used
        image_folder -- folder where images are located (should be located within a another folder)
        batch_size -- batch size used during feature creation
        gpu_name -- name of the gpu used for feature creation
        max_pool -- boolean whether to do max pooling for each cropped 4x4x2048 bbox so that it becomes 1x1x2048

    Returns:
        Reidentification feature dataset as numpy array.
    """

    print("Creating ReID Feature Dataset...")
    # apply preprocessing needed for resnet (will be further investigated if normalization is needed or should
    # be changed to be according to dataset used for detection)
    preprocess = transforms.Compose([
        # Need to cut that out since it messes with the bbox detections
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # load dataset using custom function that also returns the path of the image
    dataset = misc.ImageFolderWithPaths(root=image_folder, transform=preprocess)
    # define data loader (currently only batch size 1 is supported)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    # initialise model and load weights
    model = ft_net(751)
    model.load_state_dict(torch.load('networks/pretrained_models/ft_net.pth'))
    model.to(gpu_name)

    if max_pool:
        output_features = np.empty((len(detection_file), 2048))
    else:
        output_features = np.empty((len(detection_file), (2048 * 4 * 4)))
    # iterate through data loader
    for i, (data, target, paths) in enumerate(train_loader):
        # define input batch
        input_batch = data.to(gpu_name)
        # obtain feature map by feeding batch into ResNet and send feature map tensor back to cpu
        feature_map = model(input_batch).detach().cpu()
        # obtain current frame by looking at path name of file being processed
        frame = float(paths[0].split("/")[-1].split(".")[0])

        # calculate the reduction factor of the resnet by looking at the dimensions of the input and output
        # of the resnet
        x_reduction_factor = input_batch.shape[3] / feature_map.shape[3]
        y_reduction_factor = input_batch.shape[2] / feature_map.shape[2]

        # filter detections to only obtain detections of that frame
        frame_detections = np.array([det for det in detection_file if det[1] == frame])
        if frame_detections.size == 0:
            continue
        # create bounding box array (need to have 0 at first position for roi_align function
        bboxes = np.empty((len(frame_detections), 5), dtype=np.float32)
        bboxes[:, 0] = np.zeros(len(frame_detections))
        bboxes[:, 1] = frame_detections[:, 3] / x_reduction_factor
        bboxes[:, 2] = frame_detections[:, 4] / y_reduction_factor
        # define x2 by adding 40% of the height of the bounding box to the x1 coordinate
        bboxes[:, 3] = (bboxes[:, 1] + frame_detections[:, 6] * 0.4) / x_reduction_factor
        bboxes[:, 4] = bboxes[:, 2] + frame_detections[:, 6] / y_reduction_factor
        # send bboxes to cpu as Torch Tensor
        bboxes = torch.from_numpy(bboxes).to('cpu')
        # apply roi_align algorithm on top of bboxes to obtain the 4x4x2048 cropped bboxes
        cropped_bboxes = ops.roi_align(feature_map, bboxes, (4, 4))
        # if pooling is wanted (iterations.e. only take highest value of each 4x4 array) then Pooling layer is applied
        if max_pool:
            pool = torch.nn.MaxPool3d((1, 4, 4))
            cropped_bboxes = pool(cropped_bboxes)
        # flatten each cropped bbox and assign it to its position in the output features array
        for j, cropped_bbox in enumerate(cropped_bboxes):
            output_features[int(frame_detections[j, 0]), :] = cropped_bbox.flatten()
        # print progression
        if i % 100 == 0:
            print('Batch [{0}/{1}]'.format(i, len(train_loader)))
    return output_features


def create_spatial_feature_dataset(detection_file):
    """
    Function that creates a spatial feature dataset for a given detection file.

    Parameters:
        detection_file -- detection file to be used to create spatial dataset
    """
    print("Creating Spatial Feature Dataset...")
    # create empty output array
    output_features = np.empty((len(detection_file), 5))
    # bbox coordinates
    output_features[:, 0] = detection_file[:, 3]
    output_features[:, 1] = detection_file[:, 4]
    output_features[:, 2] = detection_file[:, 5]
    output_features[:, 3] = detection_file[:, 6]
    # frame number
    output_features[:, 4] = detection_file[:, 1]

    return output_features


def create_extra_feature_dataset(detection_file):
    """
    Function that creates a extra feature dataset for a given detection file.

    Parameters:
        detection_file -- detection file to be used to create extra dataset
    """
    print("Creating Extra Feature Dataset...")
    # create empty output array
    output_features = np.empty((len(detection_file), 1))
    # calculate absolute size of bounding boxes and append it to said detection
    absolute_size = np.empty((len(detection_file), 1))
    for det in detection_file:
        absolute_size[int(det[0]), :] = (det[5] * det[6])
    # detector confidence
    output_features[:, 0] = detection_file[:, -4]

    return np.concatenate((output_features, absolute_size), axis=1)


def create_autocorrelation_dataset(dataset):
    """
    Function that appends to a given dataset its autocorrelation features.

    Args:
        dataset -- dataset to be used

    Returns:
        Dataset with appended autocorrelation features
    """

    # autocorrelation function to be applied to all rows in dataset
    def autocorrelate(row):
        row_a = (row - np.mean(row)) / (np.std(row) * len(row))
        row_b = (row - np.mean(row)) / (np.std(row))
        autocorrelation = np.correlate(row_a, row_b, mode='same')
        return autocorrelation

    # apply function to all rows
    autocorrelate_df = np.apply_along_axis(autocorrelate, 0, dataset)

    # return concatenated dataframe
    return np.concatenate((dataset, autocorrelate_df), axis=1)


def create_knn_graph_dataset(knn_type, features_file, neighbors, knn_calculation, knn_metric='minkowski',
                             frame_dist_forward=0, frame_dist_backward=0, filter_dataset=None):
    """
    Function that creates a knn-graph file according to the input format of the GCN paper. knn_types that are currently
    supported are a 'normal', 'frame_distance' or 'pre_filtered_frame_distance' knn graph. knn_calculation can be
    anything from sklearn.neighbors.NearestNeighbors algorithm choices. knn_metric can be either 'euclidean' or
    'minkowski' distance.

    Args:
        knn_type -- type of knn to calculate ('normal', 'frame_distance' or 'pre_filtered_frame_distance')
        features_file -- feature file to use for knn construction
        neighbors -- no. nearest neighbors to calculate for each detection
        knn_calculation -- algorithm to use for knn graph calculation (see sklearn.neighbors.NearestNeighbors)
        knn_metric -- distance metric to use for knn calculation ('euclidean' or 'minkowski'; default: minkowski)
        frame_dist_forward -- backward frame distance to start with (default: 0)
        frame_dist_backward -- backward frame distance to start with (default: 0)
        filter_dataset -- dataset to use for pre-filtering of knn features (default: None)

    Returns:
        KNN graph dataset following input format i.e. per row index of detection and index of all neighbors
    """
    if knn_type == 'frame_distance':
        print("Creating Frame Distance KNN Graph Dataset...")
        # create empty output array
        output_knn_graph = np.empty([features_file.shape[0], neighbors + 1])
        # append index to feature array for reconstruction
        features_index = np.empty((features_file.shape[0], features_file.shape[1] + 1), dtype=int)
        features_index[:, 0] = np.arange(features_file.shape[0])
        features_index[:, 1:] = features_file
        # go through features of each detection
        for i in range(features_file.shape[0]):
            # get index of currently observed detection
            feat_idx = features_index[features_index[:, 0] == i][0]
            # get frame of current observed detection
            frame_idx = feat_idx[5]
            # reset frame distance values to original one
            frame_dist_fw = frame_dist_forward
            frame_dist_bw = frame_dist_backward
            # filter feature file according to frame distance values, i.e. detections only within the distance
            knn_feat = features_index[features_index[:, 5] <= frame_idx + frame_dist_fw]
            knn_feat = knn_feat[knn_feat[:, 5] >= frame_idx - frame_dist_bw]
            # if there are not enough detections meeting that condition iteratively increase frame distance until
            # there are enough detections, i.e. neighbors + 1 detections
            while len(knn_feat) < neighbors + 1:
                knn_feat = features_index[features_index[:, 5] <= frame_idx + frame_dist_fw]
                knn_feat = knn_feat[knn_feat[:, 5] >= frame_idx - frame_dist_bw]
                frame_dist_fw += 1
                frame_dist_bw += 1
            # obtain sorted knn of filtered data and save ordering to output knn file
            neighbor_idx = get_neighbors(knn_feat, feat_idx[1:], neighbors + 1, knn_metric)
            output_knn_graph[feat_idx[0], :] = neighbor_idx
            # print progress
            if i % 1000 == 0 or i + 1 == features_file.shape[0]:
                print('Processed: ' + str(feat_idx[0]) + '/' + str(features_file.shape[0]))
    elif knn_type == 'pre_filtered_frame_distance':
        print("Creating Pre-Filtered Frame Distance KNN Graph Dataset...")
        # create empty output array
        output_knn_graph = np.empty([features_file.shape[0], neighbors + 1])
        # append index to feature array for reconstruction
        features_index = np.empty((features_file.shape[0], features_file.shape[1] + 1), dtype=int)
        features_index[:, 0] = np.arange(features_file.shape[0])
        features_index[:, 1:] = features_file
        # obtain 500 knn of each detection according to filter dataset
        filter_knn = create_knn_graph_dataset('normal', filter_dataset, 500, knn_calculation, knn_metric)
        # go through features of each detection
        for i in range(features_file.shape[0]):
            # obtain 500 nearest neighbors' features of current detection
            fil_knn_feat = features_index[filter_knn[i, :]]
            # get index of currently observed detection
            feat_idx = fil_knn_feat[fil_knn_feat[:, 0] == i][0]
            # get frame number of currently observed detection
            frame_idx = feat_idx[5]
            # reset frame distance values to original one
            frame_dist_fw = frame_dist_forward
            frame_dist_bw = frame_dist_backward
            # filter 500 nearest neighbors according to frame distance criterion
            knn_feat = fil_knn_feat[fil_knn_feat[:, 5] <= frame_idx + frame_dist_fw]
            knn_feat = knn_feat[knn_feat[:, 5] >= frame_idx - frame_dist_bw]
            # if there are not enough detections left after filtering, i.e. less than neighbors + 1, iteratively
            # increase frame distance criterion until there are enough detections
            while len(knn_feat) < neighbors + 1:
                knn_feat = fil_knn_feat[fil_knn_feat[:, 5] <= frame_idx + frame_dist_fw]
                knn_feat = knn_feat[knn_feat[:, 5] >= frame_idx - frame_dist_bw]
                frame_dist_fw += 1
                frame_dist_bw += 1
            # obtain sorted knn of filtered data and save ordering to output knn file
            neighbor_idx = get_neighbors(knn_feat, feat_idx[1:], neighbors + 1, knn_metric)
            output_knn_graph[feat_idx[0], :] = neighbor_idx
            # print progression
            if i % 1000 == 0 or i + 1 == features_file.shape[0]:
                print('Processed: ' + str(feat_idx[0]) + '/' + str(features_file.shape[0]))
    else:
        print("Creating KNN Graph Dataset...")
        # create output array which is of size (no. detections, no. neighbors + 1)
        output_knn_graph = np.empty((features_file.shape[0], neighbors + 1), dtype=int)
        # create knn_graph object
        knn_graph = NearestNeighbors(n_neighbors=neighbors, n_jobs=20, algorithm=knn_calculation)
        # fit the features data
        # maybe also base this here on spatial distance (currently appearance)
        knn_graph.fit(features_file)
        # obtain the id's of the 200 nearest neighbors of each detection
        neigh_idx = knn_graph.kneighbors(return_distance=False)
        # append the id of the observed node
        output_knn_graph[:, 0] = np.arange(features_file.shape[0])
        # append the id's of the 200 neighbors of the node
        output_knn_graph[:, 1:] = neigh_idx

    return output_knn_graph.astype(int)


def get_neighbors(train, test_row, num_neighbors, metric):
    """
    Function to calculate nearest neighbors according to a metric.

    Args:
        train -- features to use for training (i.e. to calculate distance inbetween)
        test_row -- instance that neighbors are to be obtained of
        num_neighbors -- no. neighbors to calculate for test_row
        metric -- distance metric to use to decide which neighbors are nearest

    Returns:
        Index list of nearest neighbors
    """

    def euclidean_distance(vector1, vector2):
        """
        Function to calculate euclidean distance between two vectors.
        Source: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
        Args:
            row1 -- first input vector
            row2 -- second input vector

        Returns:
            euclidean distance measure between row1 and row2
        """
        distance = 0.0
        for i in range(len(vector1) - 1):
            distance += (vector1[i] - vector2[i]) ** 2
        return sqrt(distance)

    def minkowski_distance(x, y, p_value):
        """
        Function to calculate minkowski distance between two vectors
        Source: https://www.geeksforgeeks.org/minkowski-distance-python/

        Args:
            x -- first input vector
            y -- second input vector
            p_value -- p_value to employ

        Returns:
            minkowski distance between x and y
        """

        def p_root(value, root):
            """
            Function to calculate root p value
            Args:
                value -- p value to use
                root -- root to use

            Returns:
                Root p value
            """
            root_value = 1 / float(root)
            return round(Decimal(value) **
                         Decimal(root_value), 3)

        # pass the p_root function to calculate
        # all the value of vector parallely
        return (p_root(sum(pow(abs(a - b), p_value)
                           for a, b in zip(x, y)), p_value))

    # Locate the most similar neighbors
    distances = list()
    for train_row in train:
        index = train_row[0]
        # either calculate minkoswki or euclidean distance between test_row and each train_row
        if metric == 'euclidean':
            dist = euclidean_distance(test_row, train_row[1:])
        elif metric == 'minkowski':
            dist = minkowski_distance(test_row, train_row[1:], 2)
        # save distance to distance array as triple index, row and distance
        distances.append((index, train_row, dist))
    # sort distances array
    distances.sort(key=lambda tup: tup[2])
    # save index of neighbors to list
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


"""
CONSISTENCY CHECKS
"""


def check_mod_det_file(det_file, mod_det_file):
    """
    Function that checks whether the modified detection file is consistent with the original detection file.
    Goes through each row and compared by equality (except added index column)

    Args:
        det_file -- original detection file
        mod_det_file -- modified detection file that is to be checked
    """
    print('Checking Modified Detection File...')
    for i in range(len(mod_det_file)):
        # If rows are not the same then user is notified
        if (mod_det_file[i, 1:] != det_file[i, :]).all():
            print('Rows' + str(i) + 'not the same')


def get_all_metadata_files(filepath):
    """
    Function that returns the relative paths to all metadata files within a detector directory.
    Return prediciton_type is a list of strings

    Args:
        filepath -- returns all metadata files within a directory

    Returns:
        Paths to metadata files as list
    """
    output_files = []
    for filename in os.listdir(filepath):
        # if filename in directory contains "metadata" its relative path is appended to the output list
        if "metadata" in filename:
            output_files.append(os.path.join(filepath, filename))
    return output_files


def check_metadata_file_length(det_file, metadata_file):
    """
    Function that checks whether a detection file and metadata file are of same length i.e. contain the same amount
    of detections. If it is not the case the user is notified

    Args:
        det_file -- original detection file
        metadata_file -- metadata file whose length is to be checked
    """
    print('Checking Metadata File Length...')
    if len(det_file) != len(metadata_file):
        print("Metadata and detection file are not the same length!")


def check_detections(det_file, metadata_file):
    """
    Function that checks whether the idx, frame and detection coordinates in metadata file match the ones from the
    original detection file. Iterates over each row and checks with corresponding row in detection

    Args:
        det_file -- original detection file
        metadata_file -- metadata file whose detections are to be checked
    """
    print('Checking Detections of Metadata File...')
    # round due to troubles between pandas and numpy
    rounded_metadata = metadata_file.round(3)
    num_lines_det = len(rounded_metadata)
    # iterate over each row of detection file and check with same row in metadata file; if records mismatch user is
    # notified
    for i in range(len(det_file)):
        if det_file[i, 0] != rounded_metadata.iloc[i]['idx']:
            print('Index differ in row ' + str(i))
        if det_file[i, 1] != rounded_metadata.iloc[i]['frame']:
            print('Frames differ in row ' + str(i))
        if det_file[i, 3] != rounded_metadata.iloc[i]['det_x1']:
            print('X1 differ in row ' + str(i))
        if det_file[i, 4] != rounded_metadata.iloc[i]['det_y1']:
            print('Y1 differ in row ' + str(i))
        if det_file[i, 5] != rounded_metadata.iloc[i]['det_w']:
            print('W differ in row ' + str(i))
        if det_file[i, 6] != rounded_metadata.iloc[i]['det_h']:
            print('H differ in row ' + str(i))
        if i % 2500 == 0:
            print("Detection processed: " + str(i) + "/" + str(num_lines_det))


def check_labeling(gt_file, metadata_file):
    """
    Function that checks whether the labeling of the metadata file with the ground truth file. Checks for each row in
    the metadata file whether:
        - Ground truth bounding box in metadata row is contained in ground truth file
        - If the corresponding bbox in the ground truth file is not excluded
        - If the label assigned to the bbox is consistent with the one from the ground truth file
        - If the IoU calculated between the detection and ground truth file is the same

    Args:
        gt_file -- original ground truth file
        metadata_file -- metadata file whose assigned ground truth bboxes are to be checked
    """
    print('Checking Labeling of Metadata File...')
    # need to round metadata file (otherwise format conflict between numpy and pandas)
    rounded_metadata = metadata_file.round(3)
    # number of detection iterations.e. length of metadata file
    num_lines_det = len(rounded_metadata)
    # ground truth bounding box coordinates array and label array
    gt_bboxes = gt_file[:, 2:6].tolist()
    gt_labels = gt_file[:, 1].tolist()
    for i in range(len(metadata_file)):
        # gt bbox tuple [x1, y1, w, h]
        gt_bbox_cord = [rounded_metadata.iloc[i]['gt_x1'], rounded_metadata.iloc[i]['gt_y1'],
                        rounded_metadata.iloc[i]['gt_w'], rounded_metadata.iloc[i]['gt_h']]
        # detection bbox tuple [x1, y1, x2, y2]
        det_bbox = [rounded_metadata.iloc[i]['det_x1'], rounded_metadata.iloc[i]['det_y1'],
                    rounded_metadata.iloc[i]['det_x1'] + rounded_metadata.iloc[i]['det_w'],
                    rounded_metadata.iloc[i]['det_y1'] + rounded_metadata.iloc[i]['det_h']]
        # gt bbox tuple [x1, y1, x2, y2]
        gt_bbox = [rounded_metadata.iloc[i]['gt_x1'], rounded_metadata.iloc[i]['gt_y1'],
                   rounded_metadata.iloc[i]['gt_x1'] + rounded_metadata.iloc[i]['gt_w'],
                   rounded_metadata.iloc[i]['gt_y1'] + rounded_metadata.iloc[i]['gt_h']]
        # check for unassigned bboxes iterations.e. coordinates all -1, iou equal to 0 and label equal to -1
        if gt_bbox_cord == [-1, -1, -1, -1] or rounded_metadata.iloc[i]['iou'] == 0 or rounded_metadata.iloc[i]['gt_labels'] == -1:
            # check if all three conditions are met; if not notify user about which one violated
            if gt_bbox_cord != [-1, -1, -1, -1]:
                print("Non assigned detection in row " + str(i) + " is falsely assigned coordinates")
            if rounded_metadata.iloc[i]['iou'] != 0:
                print("Non assigned detection in row " + str(i) + " is falsely assigned iou above 0")
            if rounded_metadata.iloc[i]['gt_labels'] != -1:
                print("Non assigned detection in row " + str(i) + " is falsely assigned a label")
        else:
            # check if gt bbox in metadata file is in gt file, if not catch ValueError and notify user
            try:
                gt_idx = gt_bboxes.index(gt_bbox_cord)
                # check if corresponding bbox in gt file is not excluded; if not notify user
                if gt_file[gt_idx, 7] == 0:
                    print("GT BBox in row " + str(i) + " is actually excluded")
                # check if corresponding bbox in gt file has same label; if not notify user
                if gt_labels[gt_idx] != rounded_metadata.loc[i]['gt_labels']:
                    print("GT BBox label in row " + str(i) + " differs")
                if rounded_metadata.loc[i]['is_background']:
                    rounded_metadata.loc[i]['gt_labels']
                # check if iou between det bbox and gt bbox can be reproduced; if not notify user
                if np.round(iou(det_bbox, gt_bbox), 3) != rounded_metadata.iloc[i]['iou']:
                    print('IoU differ in row ' + str(i))
            except ValueError:
                print("GT BBox in row " + str(i) + " not in GT file")
        if i % 2500 == 0:
            print("Detection processed: " + str(i) + "/" + str(num_lines_det))


def check_splits(metadata_file, ground_truth_file):
    """
    Function that checks whether the splitting is consistent. That is it is checked whether:
        - Labeling is consistent with IoU and background labeling is consistent with chosen method
        - A detection is never part of more than one split (class/ background/ excluded)
        - A detection is never part of more than one split (train/ valid or train/ valid/ test)
    """
    print('Checking Splits...')
    num_lines_det = len(metadata_file)
    num_classes = max([item[1] for item in ground_truth_file])
    label_cols = [col for col in metadata_file.columns if 'labels_' in col]
    two_splits = [col for col in metadata_file.columns if 'two_split' in col]
    three_splits = [col for col in metadata_file.columns if 'three_split' in col]

    for i in range(len(metadata_file)):
        for label_col in label_cols:
            iou_upper = label_col.split("_")[-2]
            iou_lower = label_col.split("_")[-1]
            bg_handling = metadata_file.iloc[i]['background_handling']

            # check class detections
            if metadata_file.iloc[i]['is_class']:
                if metadata_file.iloc[i]['iou'] <= iou_upper:
                    print("IoU does not satisfy is_class threshold in column row " + str(i))

            # check background detections
            if metadata_file.iloc[i]['is_background']:
                # check cluster background labeling
                if bg_handling == "clusters":
                    # if label is not 0 notify user
                    if metadata_file.iloc[i][label_col] != 0:
                        print("Labeling is inconsistent with background handling in column row " + str(i))
                # check singleton background labeling
                elif bg_handling == "singletons":
                    if metadata_file.iloc[i][label_col] <= num_classes or \
                            np.count_nonzero(metadata_file[label_col] == metadata_file.iloc[i][label_col]):
                        print("Labeling is inconsistent with background handling in column row " + str(i))
                if metadata_file.iloc[i]['iou'] >= iou_lower:
                    print("IoU does not satisfy is_background threshold in column row " + str(i))

            # check excluded detections
            if metadata_file.iloc[i]['is_excluded']:
                if metadata_file.iloc[i]['iou'] > iou_upper or metadata_file.iloc[i]['iou'] < iou_lower:
                    print("IoU does not satisfy is_excluded threshold in column row " + str(i))

            # check consistency of class/ background/ excluded splits
            # if more than one of the is_class, is_background and is_excluded column is true, then the user is notified
            if int(metadata_file.iloc[i]['is_class']) + int(metadata_file.iloc[i]['is_background']) + int(
                    metadata_file.iloc[i]['is_excluded']) > 1:
                print("Row " + str(i) + "is included in more than one split (class/ background/ excluded)!")
            # if none of the is_class, is_background and is_excluded column is true, then the user is notified
            if int(metadata_file.iloc[i]['is_class']) + int(metadata_file.iloc[i]['is_background']) + int(
                    metadata_file.iloc[i]['is_excluded']) == 0:
                print("Row " + str(i) + "is included in no split (class/ background/ excluded)!")

        # check two-way and three-way splits
        for two_split in two_splits:
            iou_upper = two_split.split("_")[-2]
            iou_lower = two_split.split("_")[-1]
            # if more than one of the split filter arrays is true, then the user is notified
            if int(metadata_file.iloc[i]['is_train_two_split_' + str(iou_upper) + '_' + str(iou_lower)]) + \
                    int(metadata_file.iloc[i]['is_valid_two_split_' + str(iou_upper) + '_' + str(iou_lower)]) > 1:
                print("Row " + str(i) + "is included in more than one split (" + str(iou_upper) + "_" + str(iou_lower) +
                      ") (train/ valid)!")
            # if none of the split filter arrays is true, then the user is notified
            if int(metadata_file.iloc[i]['is_train_two_split_' + str(iou_upper) + '_' + str(iou_lower)]) + \
                    int(metadata_file.iloc[i]['is_valid_two_split_' + str(iou_upper) + '_' + str(iou_lower)]) == 0:
                print("Row " + str(i) + "is included in no split (" + str(iou_upper) + "_" + str(iou_lower) +
                      ") (train/ valid)!")
        for three_split in three_splits:
            iou_upper = three_split.split("_")[-2]
            iou_lower = three_split.split("_")[-1]
            # if more than one of the split filter arrays is true, then the user is notified
            if int(metadata_file.iloc[i]['is_train_three_split_' + str(iou_upper) + '_' + str(iou_lower)]) + \
                    int(metadata_file.iloc[i]['is_valid_three_split_' + str(iou_upper) + '_' + str(iou_lower)]) + \
                    int(metadata_file.iloc[i]['is_test_three_split_' + str(iou_upper) + '_' + str(iou_lower)]) > 1:
                print("Row " + str(i) + "is included in more than one split (" + str(iou_upper) + "_" + str(iou_lower) +
                      ") (train/ valid/ test)!")
            # if none of the split filter arrays is true, then the user is notified
            if int(metadata_file.iloc[i]['is_train_two_split_' + str(iou_upper) + '_' + str(iou_lower)]) + \
                    int(metadata_file.iloc[i]['is_valid_two_split_' + str(iou_upper) + '_' + str(iou_lower)]) + \
                    int(metadata_file.iloc[i]['is_valid_three_split_' + str(iou_upper) + '_' + str(iou_lower)]) == 0:
                print("Row " + str(i) + "is included in no split (" + str(iou_upper) + "_" + str(iou_lower) +
                      ") (train/ valid/ test)!")

        # print progress
        if i % 2500 == 0:
            print("Detection processed: " + str(i) + "/" + str(num_lines_det))


"""
COMBINED FUNCTIONS
"""


def check_metadata_file(det_file, mod_det_file, gt_file, metadata_file):
    """
    Function that put together all individual functions and applies each of them on a metadata file. First creates
    modified detection file and checks its consistency; then loops over each metadata file.

    Args:
        det_file -- original detection file
        gt_file -- original ground truth file
        metadata_file -- created metadata file that is to be checked
    """
    # modify detection file (add index as first column)
    check_mod_det_file(det_file, mod_det_file)

    # per metadata file apply all functions that check consistency
    print("PERFORMING DATA CONSISTENCY CHECKS ON CONSTRUCTED METADATA FILE")
    check_metadata_file_length(mod_det_file, metadata_file)
    check_detections(mod_det_file, metadata_file)
    check_labeling(metadata_file, gt_file)
    check_splits(metadata_file, gt_file)
    print("DONE CHECKING METADATA FILE")


def create_label_files(data_directory, sequence, detection_file_path, ground_truth_path, background_handling,
                       iou_ranges_list, iou_split_list, split_ratio_three_way, split_ratio_two_way, ):
    """
    Function that creates the labeling for a sequence a as well as a train-valid and train-valid-test split
    according to split criteria. Saves resulting metadata (unique name with timestamp) in sequence folder.

    Args:
        data_directory -- directory where sequence folder is located in
        sequence -- name of sequence
        detection_file_path -- path to detection file within sequence folder
        ground_truth_path -- path to ground truth file within sequence folder
        background_handling -- method used during label creation to deal with background and other detections
                        ("clusters" or "singletons")
        iou_ranges_list -- list of iou ranges to create labeling for
        iou_split_list -- iou range used for splits
        split_ratio_three_way -- train/ valid/ test split ratio (needs to be list of three percentage values; add to 1)
        split_ratio_two_way -- train/ valid split ratio (needs to be list of three percentage values; add to 1)
    """
    print('Creating Labeling for ' + sequence + '...')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # load detection and ground truth file
    det_file_mot = np.loadtxt(os.path.join(data_directory, sequence, detection_file_path), delimiter=",")
    gt_file_mot = np.loadtxt(os.path.join(data_directory, sequence, ground_truth_path), delimiter=",")

    # folder where output is saved to
    output_folder = os.path.join(data_directory, sequence)

    # create modified detection file and initialise metadata file
    mod_det_file_mot = create_modified_detection_file(detection_file=det_file_mot)
    meta_data_mot = create_meta_dataset(detection_file=mod_det_file_mot)

    # create label dataset and metadata file
    meta_data_mot = create_label_dataset(detection_file=mod_det_file_mot,
                                         ground_truth_file=gt_file_mot,
                                         metadata_file=meta_data_mot,
                                         iou_ranges=iou_ranges_list,
                                         background_handling=background_handling
                                         )
    # create splits for each iou split pair
    for iou_split in iou_split_list:
        # create train and valid and extend metadata file with filter arrays
        meta_data_mot = create_two_split(metadata_file=meta_data_mot,
                                         split_ratio=split_ratio_two_way,
                                         split_iou=iou_split,
                                         )
        # create train valid and test split and extend metadata file with filter arrays
        meta_data_mot = create_three_split(metadata_file=meta_data_mot,
                                           split_ratio=split_ratio_three_way,
                                           split_iou=iou_split
                                           )
    # save metadata file with log_timestamp
    meta_data_mot.to_csv(os.path.join(output_folder, timestamp + "_metadata.csv"), index=False,
                         float_format='%g')
    # perform data consistency checks
    check_metadata_file(det_file_mot, mod_det_file_mot, gt_file_mot, meta_data_mot)


def create_feature_files(data_directory, sequence, detection_file_path, image_path, img_dim, batch_size,
                         gpu_name, max_pool_feat):
    """
    Function that creates feature file for specific sequence and saves feature datasets in sequence folder.

    Args:
        data_directory -- directory where sequence folder is located in
        sequence -- name of sequence
        detection_file_path -- path to detection file within sequence folder
        image_path -- path to image folder within sequence folder
        img_dim -- dimensions of input images
        batch_size -- batch_size employed during feature creation
        gpu_name -- gpu used during feature creation
        max_pool_feat -- boolean whether to max_pool or not after bbox extraction
    """
    print('Creating Feature Files for ' + sequence + '...')
    # load detection and ground truth file
    det_file = np.loadtxt(os.path.join(data_directory, sequence, detection_file_path), delimiter=",")
    # Folder variables
    output_folder = os.path.join(data_directory, sequence)
    img_folder = os.path.join(data_directory, sequence, image_path)

    # create modified detection file and initialise metadata file
    mod_det_file = create_modified_detection_file(detection_file=det_file)

    # create spatial, reid and person feature datasets
    spatial_feat = create_spatial_feature_dataset(mod_det_file)
    reid_feat = create_reid_feature_dataset(mod_det_file, img_folder, batch_size, gpu_name, max_pool_feat)
    person_feat = create_person_feature_dataset(mod_det_file, img_folder, batch_size, img_dim, gpu_name, max_pool_feat)
    appearance_feat = create_appearance_feature_dataset(mod_det_file, img_folder, batch_size, gpu_name, max_pool_feat)
    extra_feat = create_extra_feature_dataset(mod_det_file)

    # save feature datasets
    np.save(os.path.join(output_folder, "feat_spa.npy"), spatial_feat)
    np.save(os.path.join(output_folder, "feat_reid.npy"), reid_feat)
    np.save(os.path.join(output_folder, "feat_person.npy"), person_feat)
    np.save(os.path.join(output_folder, "feat_app.npy"), appearance_feat)
    np.save(os.path.join(output_folder, "feat_extra.npy"), extra_feat)



if __name__ == '__main__':
    data_dir = 'data/MOT/MOT17'
    det_path = 'det/det.txt'
    gt_path = 'gt/gt.txt'
    img_path = 'img'
    gpu = 'cuda:2'
    max_pool = True
    iou_ranges = [[0.7, 0.3], [0.5, 0.3]]
    iou_splits = [[0.7, 0.3], [0.5, 0.3]],
    split_ratio_2w = [0.8, 0.2],
    split_ratio_3w = [0.8, 0.1, 0.1]
    bg_handling = 'clusters'

    train_dets = [
        ['MOT17-02-DPM', (1920, 1080)], ['MOT17-02-FRCNN', (1920, 1080)], ['MOT17-02-SDP', (1920, 1080)],
        ['MOT17-04-DPM', (1920, 1080)], ['MOT17-04-FRCNN', (1920, 1080)], ['MOT17-04-SDP', (1920, 1080)],
        ['MOT17-05-DPM', (640, 480)], ['MOT17-05-FRCNN', (640, 480)], ['MOT17-05-SDP', (640, 480)],
        ['MOT17-09-DPM', (1920, 1080)], ['MOT17-09-FRCNN', (1920, 1080)], ['MOT17-09-SDP', (1920, 1080)],
        ['MOT17-10-DPM', (1920, 1080)], ['MOT17-10-FRCNN', (1920, 1080)], ['MOT17-10-SDP', (1920, 1080)],
        ['MOT17-11-DPM', (1920, 1080)], ['MOT17-11-FRCNN', (1920, 1080)], ['MOT17-11-SDP', (1920, 1080)],
        ['MOT17-13-DPM', (1920, 1080)], ['MOT17-13-FRCNN', (1920, 1080)], ['MOT17-13-SDP', (1920, 1080)]
    ]

    test_dets = [
        ['MOT17-01-DPM', (1920, 1080)], ['MOT17-01-FRCNN', (1920, 1080)], ['MOT17-01-SDP', (1920, 1080)],
        ['MOT17-03-DPM', (1920, 1080)], ['MOT17-03-FRCNN', (1920, 1080)], ['MOT17-03-SDP', (1920, 1080)],
        ['MOT17-06-DPM', (640, 480)], ['MOT17-06-FRCNN', (640, 480)], ['MOT17-06-SDP', (640, 480)],
        ['MOT17-07-DPM', (1920, 1080)], ['MOT17-07-FRCNN', (1920, 1080)], ['MOT17-07-SDP', (1920, 1080)],
        ['MOT17-08-DPM', (1920, 1080)], ['MOT17-08-FRCNN', (1920, 1080)], ['MOT17-08-SDP', (1920, 1080)],
        ['MOT17-12-DPM', (1920, 1080)], ['MOT17-12-FRCNN', (1920, 1080)], ['MOT17-12-SDP', (1920, 1080)],
        ['MOT17-14-DPM', (1920, 1080)], ['MOT17-14-FRCNN', (1920, 1080)], ['MOT17-14-SDP', (1920, 1080)]
    ]

    # save start time of dataset creation
    start = time.time()

    # loop through all training sequences
    for train_det in train_dets:
        # create labeling for train sequence
        create_label_files(data_directory=os.path.join(data_dir, 'train'),
                           sequence=train_det[0],
                           detection_file_path=det_path,
                           ground_truth_path=gt_path,
                           background_handling=bg_handling,
                           iou_ranges_list=iou_ranges,
                           iou_split_list=iou_splits,
                           split_ratio_two_way=split_ratio_2w,
                           split_ratio_three_way=split_ratio_3w
                           )
        # create feature files for train sequence
        create_feature_files(data_directory=os.path.join(data_dir, 'train'),
                             sequence=train_det[0],
                             detection_file_path=det_path,
                             image_path=img_path,
                             img_dim=train_det[1],
                             batch_size=1,
                             gpu_name=gpu,
                             max_pool_feat=max_pool
                             )

    # loop through all testing sequences
    for test_det in test_dets:
        # create feature datasets for test sequence
        create_feature_files(data_directory=os.path.join(data_dir, 'test'),
                             sequence=test_det[0],
                             detection_file_path=det_path,
                             image_path=img_path,
                             img_dim=test_det[1],
                             batch_size=1,
                             gpu_name=gpu,
                             max_pool_feat=max_pool
                             )

    # calculate time data creation took
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Final time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
