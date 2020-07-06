import itertools
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms, ops

from resnet_modified import resnet50
import misc

pd.options.mode.chained_assignment = None  # default='warn'


def create_meta_dataset(detection_file):
    """
    Function that creates a meta_dataset for a detection_file as a Pandas dataframe
    Initially saves ids, frames and bounding boxes of detections
    """
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
    subsequent steps
    Returns the modified detection file
    """
    print("Creating Modified Detection File...")
    output_det = np.empty((detection_file.shape[0], detection_file.shape[1] + 1))
    output_det[:, 0] = range(detection_file.shape[0])
    output_det[:, 1:] = detection_file

    return output_det


def create_label_dataset(detection_file, metadata_file, ground_truth_file, iou_range, background_handling):
    """
    Function that creates a dataset containing the detection id and its corresponding assigned label. The labels
    are obtained by calculating the iou between each detection and ground truth box and saving the largest iou
    for each detection. Thresholds are defined via the iou_range (which itself states the range in which the iou
    should be considered a non-match i.e. neither a class or background). Further, the function saves meta information
    i.e. filter arrays to obtain excluded, background and class detections, as well as the maximum iou for
    each detection and the corresponding ground truth bounding box.
    Returns label dataset and modified meta dataset.
    """
    print("Creating Label Dataset...")
    num_lines_det = len(detection_file)
    # no classes in ground truth file (needed for assigning new classes to singleton background clusters)
    no_instances = max([item[1] for item in ground_truth_file])

    # dataset that will contain both ids and labels
    labels = np.full(num_lines_det, -1)
    gt_labels = np.full(num_lines_det, -1)

    # filter arrays to obtain all detections that were assigned a class, background or both
    class_array = np.zeros(num_lines_det, dtype=bool)
    excluded_array = np.ones(num_lines_det, dtype=bool)
    included_array = np.zeros(num_lines_det, dtype=bool)
    background_array = np.zeros(num_lines_det, dtype=bool)

    # array that will contain bounding box coordinates of the matched ground truth box
    gt_x1 = np.full(num_lines_det, -1)
    gt_y1 = np.full(num_lines_det, -1)
    gt_w = np.full(num_lines_det, -1)
    gt_h = np.full(num_lines_det, -1)
    # array that will contain the corresponding maximum iou from all ground truth boxes for said detection
    iou_array = np.zeros(num_lines_det)
    test = 1
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
            # if iou is larger than any other previous observed iou (i.e. compare to one saved in iou_array)
            if curr_iou > iou_array[int(det_line[0])]:
                # then make current iou new largest one and update iou_array as well as gt_bbox arrays
                iou_array[int(det_line[0])] = curr_iou
                gt_labels[int(det_line[0])] = gt_line[1]
                gt_x1[int(det_line[0])] = gt_line[2]
                gt_y1[int(det_line[0])] = gt_line[3]
                gt_w[int(det_line[0])] = gt_line[4]
                gt_h[int(det_line[0])] = gt_line[5]
                # if the iou is larger than the upper bound of the iou_range assign it the class of the gt
                # also update filter arrays
                if curr_iou > iou_range[0]:
                    labels[int(det_line[0])] = gt_line[1]
                    class_array[int(det_line[0])] = True
                    included_array[int(det_line[0])] = True
                    background_array[int(det_line[0])] = False
                    excluded_array[int(det_line[0])] = False
                # if the iou is within the iou_range exclude the detection by updating the filter arrays
                elif iou_range[0] >= curr_iou >= iou_range[1]:
                    labels[int(det_line[0])] = gt_line[1]
                    class_array[int(det_line[0])] = False
                    background_array[int(det_line[0])] = False
                    included_array[int(det_line[0])] = False
                    excluded_array[int(det_line[0])] = True
                # if the iou is smaller than the lower bound of the iou_range it is background
                # also update filter arrays
                elif iou_range[1] > curr_iou:
                    class_array[int(det_line[0])] = False
                    background_array[int(det_line[0])] = True
                    included_array[int(det_line[0])] = True
                    excluded_array[int(det_line[0])] = False
                    # depending on which method is to be used for background the detection is either assigned
                    # a zero or a new class (by increasing no_instances)
                    if background_handling == "zero":
                        labels[int(det_line[0])] = 0
                    elif background_handling == "singleton":
                        no_instances += 1
                        labels[int(det_line[0])] = no_instances
            # if the iou is not larger than any other previous one, then skip to next ground truth detection
            else:
                continue
        if i % 2500 == 0:
            print("Detection processed: " + str(i) + "/" + str(num_lines_det))
    # save all information to metadata_file
    metadata_file['iou'] = iou_array
    metadata_file['gt_labels'] = gt_labels
    metadata_file['gt_x1'] = gt_x1
    metadata_file['gt_y1'] = gt_y1
    metadata_file['gt_w'] = gt_w
    metadata_file['gt_h'] = gt_h
    metadata_file['labels_' + str(iou_range[0]) + "_" + str(iou_range[1]) + "_" + background_handling] = labels
    metadata_file['is_class'] = class_array
    metadata_file['is_background'] = background_array
    metadata_file['is_included'] = included_array
    metadata_file['is_excluded'] = excluded_array

    return metadata_file


def create_train_valid_test(metadata_file, split_ratio):
    """
    Function that creates (approx.) evenly distributed train, valid and test datasets for a given labels_file
    It does so by handling classes and background separately to ensure that each dataset is assigned equal amounts
    of background observations. The resulting datasets are translated into filter arrays, which are appended to the
    metadata_file.
    Currently only the splitting by ID is supported
    """
    # create filter array columns in metadata_file
    metadata_file['is_train'] = False
    metadata_file['is_valid'] = False
    metadata_file['is_test'] = False

    # split metadata file into records that are assigned a class, background or nothing
    background = metadata_file[metadata_file['is_background']]
    classes = metadata_file[metadata_file['is_class']]
    removed = metadata_file[metadata_file['is_excluded']]

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
    label_col = [col for col in metadata_file.columns if 'labels_' in col]
    train_idx, valid_test_idx = next(GroupShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7)
                                     .split(classes, groups=classes[label_col[0]]))

    valid_idx, test_idx = next(GroupShuffleSplit(test_size=split_ratio[1] / (split_ratio[1] + split_ratio[2]),
                                                 n_splits=1, random_state=7)
                               .split(classes.iloc[valid_test_idx, :],
                                      groups=classes.iloc[valid_test_idx, :][label_col[0]]))
    # create idx for background
    try:
        train_bg_idx, valid_test_bg_idx = next(
            ShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7).split(background))
        valid_bg_idx, test_bg_idx = next(
            ShuffleSplit(test_size=split_ratio[1] / (split_ratio[1] + split_ratio[2]), n_splits=1,
                         random_state=7).split(background.iloc[valid_test_bg_idx, :]))
        # split background attribute into train, valid and test using indeces obtained above
        train_bg = background.iloc[train_bg_idx, :]
        train_bg['is_train'] = True
        valid_bg = background.iloc[valid_test_bg_idx, :].iloc[valid_bg_idx, :]
        valid_bg['is_valid'] = True
        test_bg = background.iloc[valid_test_bg_idx, :].iloc[test_bg_idx, :]
        test_bg['is_test'] = True
    # this exception is needed if the background detections are very small (which will cause the ShuffleSplit to
    # not work. If this is the case, the background is just split equally and not using ratios.
    except ValueError:
        train_bg, valid_bg, test_bg = np.array_split(background, 3)
        train_bg['is_train'] = True
        valid_bg['is_valid'] = True
        test_bg['is_test'] = True

    # split classes attribute into train, valid and test using indeces obtained above
    train_classes = classes.iloc[train_idx, :]
    train_classes['is_train'] = True
    valid_classes = classes.iloc[valid_test_idx, :].iloc[valid_idx, :]
    valid_classes['is_valid'] = True
    test_classes = classes.iloc[valid_test_idx, :].iloc[test_idx, :]
    test_classes['is_test'] = True
    # reconstruct metadata file
    metadata_file = pd.concat(
        [train_classes, valid_classes, test_classes, train_bg, valid_bg, test_bg, removed]).sort_values('idx')

    return metadata_file


def create_train_valid(metadata_file, split_ratio):
    """
    Function that creates (approx.) evenly distributed train, valid and test datasets for a given labels_file
    It does so by handling classes and background separately to ensure that each dataset is assigned equal amounts
    of background observations. The resulting datasets are translated into filter arrays, which are appended to the
    metadata_file.
    Currently only the splitting by ID is supported
    """
    # TODO: add choice to be able to split by frame (add parameter split by)
    # create filter array columns in metadata_file
    metadata_file['is_train'] = False
    metadata_file['is_valid'] = False

    # split metadata file into records that are assigned a class, background or nothing
    background = metadata_file[metadata_file['is_background']]
    classes = metadata_file[metadata_file['is_class']]
    removed = metadata_file[metadata_file['is_excluded']]

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
    label_col = [col for col in metadata_file.columns if 'labels_' in col]
    train_idx, valid_idx = next(GroupShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7)
                                .split(classes, groups=classes[label_col[0]]))
    # create idx for background
    try:
        train_bg_idx, valid_bg_idx = next(
            ShuffleSplit(train_size=split_ratio[0], n_splits=1, random_state=7).split(background))
        # split background attribute into train, valid and test using indeces obtained above
        train_bg = background.iloc[train_bg_idx, :]
        train_bg['is_train'] = True
        valid_bg = background.iloc[valid_bg_idx, :]
        valid_bg['is_valid'] = True
    # this exception is needed if the background detections are very small (which will cause the ShuffleSplit to
    # not work. If this is the case, the background is just split equally and not using ratios.
    except ValueError:
        train_bg, valid_bg = np.array_split(background, 2)
        train_bg['is_train'] = True
        valid_bg['is_valid'] = True

    # split classes attribute into train, valid and test using indeces obtained above
    train_classes = classes.iloc[train_idx, :]
    train_classes['is_train'] = True
    valid_classes = classes.iloc[valid_idx, :]
    valid_classes['is_valid'] = True

    # reconstruct metadata file
    metadata_file = pd.concat([train_classes, valid_classes, train_bg, valid_bg, removed]).sort_values('idx')

    return metadata_file


def create_appearance_feature_dataset(detection_file, image_folder, batch_size, gpu_name, max_pool):
    """
    Function that creates for each detection a feature array using ResNet50 and RoI-Align.
    Each Image is feed into the resnet, which is modified to return the feature map before the first fc-layer as output.
    Once the feature map is obtained all detections from that frame are filtered from the detection file.
    Using torchvision's RoI-Align each detected bounding box is translated into a 2048x4x4 numpy array.
    Said array is flattened and added to the output array in its corresponding place (determined by id of detection)
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
        # if pooling is wanted (i.e. only take highest value of each 4x4 array) then Pooling layer is applied
        if max_pool:
            pool = torch.nn.MaxPool3d((1, 4, 4))
            cropped_bboxes = pool(cropped_bboxes)
        # flatten each cropped bbox and assign it to its position in the output features array
        for j, cropped_bbox in enumerate(cropped_bboxes):
            output_features[int(frame_detections[j, 0]), :] = cropped_bbox.flatten()
        if i % 100 == 0:
            print('Batch [{0}/{1}]'.format(i, len(train_loader)))
    return output_features


def create_spatial_feature_dataset(detection_file):
    print("Creating Spatial Feature Dataset...")
    output_features = np.empty((len(detection_file), 5))
    output_features[:, 0] = detection_file[:, 3]
    output_features[:, 1] = detection_file[:, 4]
    output_features[:, 2] = detection_file[:, 5]
    output_features[:, 3] = detection_file[:, 6]
    output_features[:, 4] = detection_file[:, 1]

    return output_features


def create_knn_graph_dataset(features_file, neighbors, method):
    """
    Function that creates a knn-graph file according to the input format of the GCN paper.
    Method can be anything from sklearn.neighbors.NearestNeighbors.algorithm
    """
    print("Creating KNN Graph Dataset...")
    # create output array which is of size (no. detections, no. neighbors + 1)
    output_knn_graph = np.empty((features_file.shape[0], neighbors + 1), dtype=int)
    # create knn_graph object
    knn_graph = NearestNeighbors(n_neighbors=neighbors, n_jobs=16, algorithm=method)
    # fit the features data
    # maybe also base this here on spatial distance (currently appearance)
    knn_graph.fit(features_file)
    # obtain the id's of the 200 nearest neighbors of each detection
    neigh_idx = knn_graph.kneighbors(return_distance=False)
    # append the id of the observed node
    output_knn_graph[:, 0] = np.arange(features_file.shape[0])
    # append the id's of the 200 neighbors of the node
    output_knn_graph[:, 1:] = neigh_idx

    return output_knn_graph


def filter_split_column_for_iou(metadata_file, iou_type, iou_upper, iou_lower):
    """
    Function used to filter a metadata_file to create different versions of the train, valid and test filter array.
    For each value of iou_train, iou_valid and iou_test it creates a new column that contains a filter array, that
    will return you a dataset that assumes a different iou upper bound threshold.
    """
    for iou in itertools.product(iou_upper, iou_lower):
        # create a filter array column for the new train dataset with tr_iou as new upper bound
        metadata_file['fil_' + iou_type + '_' + str(iou[0]) + "_" + str(iou[1])] = False
        for i, det in metadata_file.iterrows():
            # for each detection, check whether it was train, valid or test
            # if so then check for new upper and original lower bound to obtain a new filter array
            try:
                if det['is_' + iou_type]:
                    if det['iou'] > iou[0]:
                        metadata_file.at[i, 'fil_' + iou_type + '_' + str(iou[0]) + "_" + str(iou[1])] = True
                    elif det['iou'] < iou[1]:
                        metadata_file.at[i, 'fil_' + iou_type + '_' + str(iou[0]) + "_" + str(iou[1])] = True
            except KeyError:
                if det['iou'] > iou[0]:
                    metadata_file.at[i, 'fil_' + iou_type + '_' + str(iou[0]) + "_" + str(iou[1])] = True
                elif det['iou'] < iou[1]:
                    metadata_file.at[i, 'fil_' + iou_type + '_' + str(iou[0]) + "_" + str(iou[1])] = True
    return metadata_file


def create_train_valid_test_files(data_directory, sequence, det_path, gt_path, img_path, bg_handling, split_ratio,
                                  train_iou_upper, train_iou_lower, valid_iou_upper, valid_iou_lower, test_iou_upper,
                                  test_iou_lower, batch_size_features, gpu_name_features, max_pool_features):
    """
    Function that creates from a sequence a train-valid split according to split criteria and creates feature dataset
    for said sequence.
    Parameters:
        Parameters:
        data_directory -- directory where sequence folder is located in
        sequence -- name of sequence
        det_path -- path to detection file within sequence folder
        gt_path -- path to ground truth file within sequence folder
        img_path -- path to image folder within sequence folder
        bg_handling -- background handling method during label creation (currently only "zero" supported)
        split_ratio -- train/ valid/ test split ratio (needs to be list of three percentage values)
        train_iou_upper -- employed upper bound iou's in train set
        train_iou_lower -- employed lower bound iou's in train set
        valid_iou_upper -- employed upper bound iou's in valid set
        valid_iou_lower -- employed lower bound iou's in valid set
        test_iou_upper -- employed upper bound iou's in test set
        test_iou_lower -- employed lower bound iou's in test set
        batch_size_features -- batch_size employed during feature creation
        gpu_name_features -- gpu used during feature creation
        max_pool_features -- boolean whether to max_pool or not after bbox extraction
    """
    print('Creating Train/ Valid Files for ' + sequence + '...')
    start = time.time()
    for detector in os.listdir(os.path.join(data_directory)):
        if sequence in detector:
            print('Processing Detector ' + detector + '...')
            # load detection and ground truth file
            det_file_mot = np.loadtxt(os.path.join(data_directory, detector, det_path), delimiter=",")
            gt_file_mot = np.loadtxt(os.path.join(data_directory, detector, gt_path), delimiter=",")

            # Folder variables
            output_folder = os.path.join(data_directory, detector)
            image_folder_path = os.path.join(data_directory, detector, img_path)

            # create modified detection file and initialise metadata file
            mod_det_file_mot = create_modified_detection_file(detection_file=det_file_mot)

            meta_data_mot = create_meta_dataset(detection_file=mod_det_file_mot)

            # create label dataset and metadata file
            meta_data_mot = create_label_dataset(detection_file=mod_det_file_mot,
                                                 ground_truth_file=gt_file_mot,
                                                 metadata_file=meta_data_mot,
                                                 iou_range=[min(min(train_iou_upper), min(valid_iou_upper),
                                                                min(test_iou_upper)),
                                                            max(max(train_iou_lower), max(valid_iou_lower),
                                                                max(test_iou_lower))],
                                                 background_handling=bg_handling
                                                 )

            # create train valid and test split and extend metadata file with filter arrays
            meta_data_mot = create_train_valid_test(metadata_file=meta_data_mot,
                                                    split_ratio=split_ratio,
                                                    )

            # use filter arrays to create different filter arrays that follow different iou thresholds
            meta_data_mot = filter_split_column_for_iou(metadata_file=meta_data_mot,
                                                        iou_type='train',
                                                        iou_upper=train_iou_upper,
                                                        iou_lower=train_iou_lower
                                                        )
            meta_data_mot = filter_split_column_for_iou(metadata_file=meta_data_mot,
                                                        iou_type='valid',
                                                        iou_upper=valid_iou_upper,
                                                        iou_lower=valid_iou_lower
                                                        )
            meta_data_mot = filter_split_column_for_iou(metadata_file=meta_data_mot,
                                                        iou_type='test',
                                                        iou_upper=test_iou_upper,
                                                        iou_lower=test_iou_lower
                                                        )

            spatial_features = create_spatial_feature_dataset(mod_det_file_mot)
            appearance_features = create_appearance_feature_dataset(mod_det_file_mot, image_folder_path,
                                                                    batch_size_features, gpu_name_features,
                                                                    max_pool_features)

            np.save(os.path.join(output_folder, "feat_spa.npy"), spatial_features)
            if max_pool_features:
                np.save(os.path.join(output_folder, "feat_app_pool.npy"), appearance_features)
            else:
                np.save(os.path.join(output_folder, "feat_app.npy"), appearance_features)

            # save metadata file with current timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            meta_data_mot.to_csv(os.path.join(output_folder, timestamp + "_train_valid_test_metadata.csv"), index=False,
                                 float_format='%g')

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Final time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def create_train_valid_files(data_directory, sequence, det_path, gt_path, img_path, bg_handling, split_ratio,
                             train_iou_upper, train_iou_lower, valid_iou_upper, valid_iou_lower, batch_size_features,
                             gpu_name_features, max_pool_features):
    """
    Function that creates from a sequence a train-valid-test split according to split criteria and creates feature
    dataset for said sequence.
    Parameters:
        Parameters:
        data_directory -- directory where sequence folder is located in
        sequence -- name of sequence
        det_path -- path to detection file within sequence folder
        gt_path -- path to ground truth file within sequence folder
        img_path -- path to image folder within sequence folder
        bg_handling -- background handling method during label creation (currently only "zero" supported)
        split_ratio -- train/ valid split ratio (needs to be list of two percentage values)
        train_iou_upper -- employed upper bound iou's in train set
        train_iou_lower -- employed lower bound iou's in train set
        valid_iou_upper -- employed upper bound iou's in valid set
        valid_iou_lower -- employed lower bound iou's in valid set
        batch_size_features -- batch_size employed during feature creation
        gpu_name_features -- gpu used during feature creation
        max_pool_features -- boolean whether to max_pool or not after bbox extraction
    """
    print('Creating Train/ Valid Files for ' + sequence + '...')
    start = time.time()
    for detector in os.listdir(os.path.join(data_directory)):
        if sequence in detector:
            print('Processing Detector ' + detector + '...')
            # load detection and ground truth file
            det_file_mot = np.loadtxt(os.path.join(data_directory, detector, det_path), delimiter=",")
            gt_file_mot = np.loadtxt(os.path.join(data_directory, detector, gt_path), delimiter=",")

            # Folder variables
            output_folder = os.path.join(data_directory, detector)
            image_folder_path = os.path.join(data_directory, detector, img_path)

            # create modified detection file and initialise metadata file
            mod_det_file_mot = create_modified_detection_file(detection_file=det_file_mot)

            meta_data_mot = create_meta_dataset(detection_file=mod_det_file_mot)

            # create label dataset and metadata file
            meta_data_mot = create_label_dataset(detection_file=mod_det_file_mot,
                                                 ground_truth_file=gt_file_mot,
                                                 metadata_file=meta_data_mot,
                                                 iou_range=[min(min(train_iou_upper), min(valid_iou_upper)),
                                                            max(max(train_iou_lower), max(valid_iou_lower))],
                                                 background_handling=bg_handling
                                                 )

            # create train valid and test split and extend metadata file with filter arrays
            meta_data_mot = create_train_valid(metadata_file=meta_data_mot,
                                               split_ratio=split_ratio,
                                               )

            # use filter arrays to create different filter arrays that follow different iou thresholds
            meta_data_mot = filter_split_column_for_iou(metadata_file=meta_data_mot,
                                                        iou_type='train',
                                                        iou_upper=train_iou_upper,
                                                        iou_lower=train_iou_lower
                                                        )
            meta_data_mot = filter_split_column_for_iou(metadata_file=meta_data_mot,
                                                        iou_type='valid',
                                                        iou_upper=valid_iou_upper,
                                                        iou_lower=valid_iou_lower
                                                        )

            spatial_features = create_spatial_feature_dataset(mod_det_file_mot)
            appearance_features = create_appearance_feature_dataset(mod_det_file_mot, image_folder_path,
                                                                    batch_size_features, gpu_name_features,
                                                                    max_pool_features)

            np.save(os.path.join(output_folder, "feat_spa.npy"), spatial_features)
            if max_pool_features:
                np.save(os.path.join(output_folder, "feat_app_pool.npy"), appearance_features)
            else:
                np.save(os.path.join(output_folder, "feat_app.npy"), appearance_features)

            # save metadata file with current timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            meta_data_mot.to_csv(os.path.join(output_folder, timestamp + "_train_valid" + "_metadata.csv"), index=False,
                                 float_format='%g')

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Final time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def create_files(file_type, data_directory, sequence, det_path, gt_path, img_path, iou_upper, iou_lower, bg_handling,
                 batch_size_features, gpu_name_features, max_pool_features):
    """
    Function that creates from a sequence a test set.
    Parameters:
        data_directory -- directory where sequence folder is located in
        sequence -- name of sequence
        det_path -- path to detection file within sequence folder
        gt_path -- path to ground truth file within sequence folder
        img_path -- path to image folder within sequence folder
        iou_range_labeling -- iou upper and lower bound for label creation (note upper bound needs to be lowest value of
                              train_iou and valid_iou
        bg_handling -- background handling method during label creation (currently only "zero" supported)
        batch_size_features -- batch_size employed during feature creation
        gpu_name_features -- gpu used during feature creation
        max_pool_features -- boolean whether to max_pool or not after bbox extraction
    """
    print('Creating ' + file_type + ' Files for ' + sequence + '...')
    start = time.time()
    for detector in os.listdir(os.path.join(data_directory)):
        if sequence in detector:
            print('Processing Detector ' + detector + '...')
            # load detection and ground truth file
            det_file_mot = np.loadtxt(os.path.join(data_directory, detector, det_path),
                                      delimiter=",")
            gt_file_mot = np.loadtxt(os.path.join(data_directory, detector, gt_path), delimiter=",")

            # Folder variables
            output_folder = os.path.join(data_directory, detector)
            image_folder_path = os.path.join(data_directory, detector, img_path)

            # create modified detection file and initialise metadata file
            mod_det_file_mot = create_modified_detection_file(detection_file=det_file_mot)

            meta_data_mot = create_meta_dataset(detection_file=mod_det_file_mot)

            # create label dataset and metadata file
            meta_data_mot = create_label_dataset(detection_file=mod_det_file_mot,
                                                 ground_truth_file=gt_file_mot,
                                                 metadata_file=meta_data_mot,
                                                 iou_range=[min(iou_upper), max(iou_lower)],
                                                 background_handling=bg_handling
                                                 )

            meta_data_mot = filter_split_column_for_iou(metadata_file=meta_data_mot,
                                                        iou_type=file_type,
                                                        iou_upper=iou_upper,
                                                        iou_lower=iou_lower
                                                        )

            # save metadata file with current timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            meta_data_mot.to_csv(os.path.join(output_folder, timestamp + "_" + file_type + "_metadata.csv"),
                                 index=False,
                                 float_format='%g')

            # create spatial and feature dataset
            spatial_features = create_spatial_feature_dataset(mod_det_file_mot)
            appearance_features = create_appearance_feature_dataset(mod_det_file_mot, image_folder_path,
                                                                    batch_size_features,
                                                                    gpu_name_features, max_pool_features)
            # save feature dataset
            np.save(os.path.join(output_folder, "feat_spa.npy"), spatial_features)
            if max_pool_features:
                np.save(os.path.join(output_folder, "feat_app_pool.npy"), appearance_features)
            else:
                np.save(os.path.join(output_folder, "feat_app.npy"), appearance_features)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Final time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    create_train_valid_test_files(data_directory='data/MOT/MOT17',
                                  sequence='MOT17-09',
                                  det_path='det/det.txt',
                                  gt_path='gt/gt.txt',
                                  img_path='img',
                                  bg_handling='zero',
                                  split_ratio=[0.8, 0.1, 0.1],
                                  train_iou_upper=[0.7],
                                  train_iou_lower=[0.3],
                                  valid_iou_upper=[0.7, 0.5],
                                  valid_iou_lower=[0.3],
                                  test_iou_upper=[0.5],
                                  test_iou_lower=[0.3],
                                  batch_size_features=1,
                                  gpu_name_features='cuda:0',
                                  max_pool_features=True
                                  )

    create_train_valid_files(data_directory='data/MOT/MOT17',
                             sequence='MOT17-09',
                             det_path='det/det.txt',
                             gt_path='gt/gt.txt',
                             img_path='img',
                             bg_handling='zero',
                             split_ratio=[0.8, 0.2],
                             train_iou_upper=[0.7],
                             train_iou_lower=[0.3],
                             valid_iou_upper=[0.5, 0.7],
                             valid_iou_lower=[0.3],
                             batch_size_features=1,
                             gpu_name_features='cuda:0',
                             max_pool_features=True
                             )

    create_files(file_type='train',
                 data_directory='data/MOT/MOT17',
                 sequence='MOT17-09',
                 det_path='det/det.txt',
                 gt_path='gt/gt.txt',
                 img_path='img',
                 bg_handling='zero',
                 iou_upper=[0.7],
                 iou_lower=[0.3],
                 batch_size_features=1,
                 gpu_name_features='cuda:0',
                 max_pool_features=True
                 )

    create_files(file_type='valid',
                 data_directory='data/MOT/MOT17',
                 sequence='MOT17-09',
                 det_path='det/det.txt',
                 gt_path='gt/gt.txt',
                 img_path='img',
                 bg_handling='zero',
                 iou_upper=[0.7, 0.5],
                 iou_lower=[0.3],
                 batch_size_features=1,
                 gpu_name_features='cuda:0',
                 max_pool_features=True
                 )

    create_files(file_type='test',
                 data_directory='data/MOT/MOT17',
                 sequence='MOT17-09',
                 det_path='det/det.txt',
                 gt_path='gt/gt.txt',
                 img_path='img',
                 iou_upper=[0.5],
                 iou_lower=[0.3],
                 bg_handling='zero',
                 batch_size_features=1,
                 gpu_name_features='cuda:0',
                 max_pool_features=True
                 )

