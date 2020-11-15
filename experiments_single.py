###################################################################
# File Name: experiments_single.py
# Author: Marius Bock
# mail: marius.bock@protonmail.com
###################################################################

import os
import os.path as osp
import sys
import time
import numpy as np
import pandas as pd
from gcn_clustering.train import train_main
from gcn_clustering.test import val_main, obtain_512_feature_map, test_main
from dataset_creation import create_knn_graph_dataset, create_autocorrelation_dataset
from gcn_clustering.utils.logging import Logger
from gcn_clustering.utils.osutils import mkdir_if_missing
from misc import adjust_labeling, create_train_args, create_test_args, plot_knn_graph, \
                 plot_embedding_graph, create_validation_output_file, create_testing_output_file, create_heurisitc_output_file

""" 
INPUT SETTINGS:
    1st: features used for training (list of spa, reid, person, clf, extra)
    2nd: features to use for knn construction (list of spa, reid, person, clf, extra)
    3rd: train label column (see corresponding metadata file)
    4th: test label column (see corresponding metadata file)
"""

inputs = [['spa', 'reid'], ['reid'], 'labels_0.7_0.3', 'labels_0.5_0.3']

"""
TRAIN/ EVAL SETTINGS
    training_type -- type of training style employed ('combined' or 'sequential')
    iterations -- number of iterations of script (i.e. training and validation runs)
    gpu_name -- name of gpu to be used (e.g. 'cuda:0)
    workers -- number of workers employed during GCN training/ validation
    batch_size -- batch size employed during training/ validation
    print_freq -- print frequency of intermediate batch results
    knn_method -- method used to construct knn graphs ('auto', 'ball_tree', 'kd_tree' or 'brute') 
    skip_validation -- boolean whether to skip validation or not 
    skip_testing -- boolean whether to skip testing or not 
    modified_detections -- boolean whether to use modified detections
    removed -- boolean whether to calculate singleton removal results
    graph_heuristic -- boolean whether to save graph heuristic input
    add_dummy_edges -- boolean whether to add dummy edges to graph heuristic input
"""

training_type = 'combined'
iterations = 1
gpu_name = 'cuda:3'
workers = 10
batch_size = 16
print_freq = 100
knn_method = 'brute'
skip_validation = False
skip_testing = True
modified_detections = True
removed = True
graph_heuristic = False
add_dummy_edges = False

"""
EXTRA FEATURES SETTINGS
    handle_absolute_size -- how to handle absolute size of bbox in extra feature dataset 
                            ('drop', 'scale_by_dataset' or 'scale_by_batch')
    handle_detector_confidence -- how to handle absolute size of bbox in extra feature dataset ('drop')
"""

handle_absolute_size = 'drop'
handle_detector_confidence = 'drop'

"""
FEEDER OPTIONS
    absolute_differences -- boolean whether to calculate absolute differences in feeder
    normalise_distances -- boolean whether to calculate normalised distances in feeder
    auto_correlate_feat -- boolean whether to calculate autocorrelation for input features
    auto_correlate_knn -- boolean whether to calculate autocorrelation for knn features
    auto_correlate_filter -- boolean whether to calculate autocorrelation for filter dataset features
    element_wise_product_feeder -- boolean whether to calculate element-wise products in feeder
    element_wise_product_type -- type of element-wise products calculated in feeder
"""

absolute_differences = False
normalise_distances = False
auto_correlate_feat = False
auto_correlate_knn = False
auto_correlate_filter = False
element_wise_product_feeder = False
element_wise_product_type = 'frame_pairwise'

"""
TRAINING-/VALIDATION-HYPERPARAMETERS
    k_at_hop -- k_at_hop employed during pivot graph construction
    active_connections -- active_connections employed during pivot graph construction
    knn_type -- type of KNN graph constructed ('normal', 'frame_distance' or 'pre_filtered_frame_distance')
    knn_frame_dist_fw -- starting forward frame distance employed for frame_distance knn graph construction
    knn_frame_dist_bw -- starting backward frame distance employed for frame_distance knn graph construction
    knn_filter_dataset -- list containing dataset ids ('reid', 'spa', etc.) to use to construct filter dataset employed 
                          during pre_filtered_frame_distance knn graph construction
    epochs -- epochs employed during training/ validation
    weight_decay -- weight decay employed during training/ validation
    momentum -- momentum employed during training/ validation
    learning_rate -- starting learning rate employed during training/ validation
"""

# training
k_at_hop_train = [200, 10]
active_connections_train = 10
knn_type_train = 'pre_filtered_frame_distance'
knn_frame_dist_fw_train = 2
knn_frame_dist_bw_train = 2
knn_filter_dataset_train = ['reid']
epochs_train = 4
weight_decay_train = 1e-4
momentum_train = 0.9
learning_rate_train = 1e-2

# validation + testing
k_at_hop_val = [20, 5]
active_connections_val = 5
knn_type_val = 'pre_filtered_frame_distance'
knn_frame_dist_fw_val = 2
knn_frame_dist_bw_val = 2
knn_filter_dataset_val = ['reid']
learning_rate_val = 1e-5
momentum_val = 0.9
weight_decay_val = 1e-4
epochs_val = 20

"""
SEED ARRAY & DETECTOR TYPES
"""

# Seed array used for same seed setup
seed_array = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Seed array used for differing seed setup
# seed_array = [1, 264810426, 437765375, 747038712, 964518008, 233555374, 480341133, 952126841, 580520770, 518450454]

detector_types = ['DPM', 'FRCNN', 'SDP']

"""
TRAIN & TEST SEQUENCES + DIRECTORIES
    gt_directory -- directory where ground truth files are located in
    train_directory -- directory where train files are located in
    test_directory -- directory where test files are located in
    train_seqs -- list of train sequences (always pair of detector name and metadata filename)
    val_seqs -- list of validation sequences (always pair of detector name and metadata filename)
    test_seq -- list of test sequences (only detector names)
"""
gt_directory = 'data/MOT/MOT17/train'

if modified_detections:
    train_directory = 'data/MOT/MOT17_mod/train'
    test_directory = 'data/MOT/MOT17_mod/test'

    train_seqs = [
        [['MOT17-04-DPM', '20201111-185225_metadata.csv']],
        [['MOT17-04-FRCNN', '20201111-191410_metadata.csv']],
        [['MOT17-04-SDP', '20201111-193046_metadata.csv']],
    ]

    val_seqs = [
        [['MOT17-02-DPM', '20201111-183609_metadata.csv'],
         ['MOT17-04-DPM', '20201111-185225_metadata.csv'],
         ['MOT17-05-DPM', '20201111-195043_metadata.csv'],
         ['MOT17-09-DPM', '20201111-200106_metadata.csv'],
         ['MOT17-10-DPM', '20201111-201305_metadata.csv'],
         ['MOT17-11-DPM', '20201111-202938_metadata.csv'],
         ['MOT17-13-DPM', '20201111-205144_metadata.csv']],
        [['MOT17-02-FRCNN', '20201111-184127_metadata.csv'],
         ['MOT17-04-FRCNN', '20201111-191410_metadata.csv'],
         ['MOT17-05-FRCNN', '20201111-195402_metadata.csv'],
         ['MOT17-09-FRCNN', '20201111-200511_metadata.csv'],
         ['MOT17-10-FRCNN', '20201111-201837_metadata.csv'],
         ['MOT17-11-FRCNN', '20201111-203714_metadata.csv'],
         ['MOT17-13-FRCNN', '20201111-205700_metadata.csv']],
        [['MOT17-02-SDP', '20201111-184637_metadata.csv'],
         ['MOT17-04-SDP', '20201111-193046_metadata.csv'],
         ['MOT17-05-SDP', '20201111-195724_metadata.csv'],
         ['MOT17-09-SDP', '20201111-200904_metadata.csv'],
         ['MOT17-10-SDP', '20201111-202402_metadata.csv'],
         ['MOT17-11-SDP', '20201111-204424_metadata.csv'],
         ['MOT17-13-SDP', '20201111-210242_metadata.csv']]
    ]
else:
    train_directory = 'data/MOT/MOT17/train'
    test_directory = 'data/MOT/MOT17/test'
    train_seqs = [
        [['MOT17-04-DPM', '20200714-143328_metadata.csv']],
        [['MOT17-04-FRCNN', '20200714-143328_metadata.csv']],
        [['MOT17-04-SDP', '20200714-143328_metadata.csv']],
    ]

    val_seqs = [
        [['MOT17-02-DPM', '20200714-142410_metadata.csv'],
         ['MOT17-04-DPM', '20200714-143328_metadata.csv'],
         ['MOT17-05-DPM', '20200730-172325_metadata.csv'],
         ['MOT17-09-DPM', '20200714-163347_metadata.csv'],
         ['MOT17-10-DPM', '20200730-180412_metadata.csv'],
         ['MOT17-11-DPM', '20200730-183224_metadata.csv'],
         ['MOT17-13-DPM', '20200730-204120_metadata.csv']],
        [['MOT17-02-FRCNN', '20200714-142410_metadata.csv'],
         ['MOT17-04-FRCNN', '20200714-143328_metadata.csv'],
         ['MOT17-05-FRCNN', '20200730-172325_metadata.csv'],
         ['MOT17-09-FRCNN', '20200714-163347_metadata.csv'],
         ['MOT17-10-FRCNN', '20200730-180412_metadata.csv'],
         ['MOT17-11-FRCNN', '20200730-183224_metadata.csv'],
         ['MOT17-13-FRCNN', '20200730-204120_metadata.csv']],
        [['MOT17-02-SDP', '20200714-142410_metadata.csv'],
         ['MOT17-04-SDP', '20200714-143328_metadata.csv'],
         ['MOT17-05-SDP', '20200730-172325_metadata.csv'],
         ['MOT17-09-SDP', '20200714-163347_metadata.csv'],
         ['MOT17-10-SDP', '20200730-180412_metadata.csv'],
         ['MOT17-11-SDP', '20200730-183224_metadata.csv'],
         ['MOT17-13-SDP', '20200730-204120_metadata.csv']]
    ]

test_seqs = [
    [['MOT17-01-DPM'],
     ['MOT17-03-DPM'],
     ['MOT17-06-DPM'],
     ['MOT17-07-DPM'],
     ['MOT17-08-DPM'],
     ['MOT17-12-DPM'],
     ['MOT17-14-DPM']],
    [['MOT17-01-FRCNN'],
     ['MOT17-03-FRCNN'],
     ['MOT17-06-FRCNN'],
     ['MOT17-07-FRCNN'],
     ['MOT17-08-FRCNN'],
     ['MOT17-12-FRCNN'],
     ['MOT17-14-FRCNN']],
    [['MOT17-01-SDP'],
     ['MOT17-03-SDP'],
     ['MOT17-06-SDP'],
     ['MOT17-07-SDP'],
     ['MOT17-08-SDP'],
     ['MOT17-12-SDP'],
     ['MOT17-14-SDP']]
]

# parameters used to calculate runtime
start = time.time()
log_date = time.strftime('%Y%m%d')
log_timestamp = time.strftime('%H%M%S')

# saves logs to a file (standard output redirected)
sys.stdout = Logger(os.path.join('logs', log_date, log_timestamp, 'log.txt'))

# initialise eval_data array that stores all evaluation relevant data
columns = ['run']
if training_type == 'combined':
    for detector_type in detector_types:
        columns += ['train_' + detector_type + '_loss']
        columns += ['train_' + detector_type + '_avg_loss']
        columns += ['train_' + detector_type + '_val_loss']
        columns += ['train_' + detector_type + '_avg_val_loss']
elif training_type == 'sequential':
    for detectors in train_seqs:
        for detector in detectors:
            for detector_type in detector_types:
                columns += ['train_' + detector_type + '_' + detector[0] + '_val_loss']
                columns += ['train_' + detector_type + '_' + detector[0] + '_avg_val_loss']
for detectors in val_seqs:
    for detector in detectors:
        columns += [detector[0] + '_loss']
        columns += [detector[0] + '_avg_loss']
eval_data = pd.DataFrame(columns=columns)

# append run index to evaluation dataframe
run_array = []
for run in range(iterations):
    run_array.append('run_' + str(run + 1))
eval_data['run'] = run_array

# use k_at_hop and active_connections to calculate no. of neighbors during knn construction
knn_train = max(max(k_at_hop_train), active_connections_train)
knn_val = max(max(k_at_hop_val), active_connections_val)

# print settings for logging
print('\nSEED ARRAY: ')
print(seed_array)
print('\nINPUT SETTINGS: ')
print(inputs)
print('\nTraining using:')
print(train_seqs)
print('\nValidating using:')
print(val_seqs)
print('\nGENERAL SETTINGS: ')
print('GPU: ' + gpu_name +
      '; WORKERS: ' + str(workers) +
      '; BATCH_SIZE: ' + str(batch_size) +
      '; PRINT_FREQ: ' + str(print_freq) +
      '; KNN_METHOD: ' + str(knn_method) +
      '; KNN_TYPE_TRAIN: ' + knn_type_train +
      '; KNN_TRAIN: ' + str(knn_train) +
      '; KNN_TYPE_VAL: ' + knn_type_val +
      '; KNN_VAL: ' + str(knn_val) +
      '; KNN_FRAME_DIST_FW_TRAIN: ' + str(knn_frame_dist_fw_train) +
      '; KNN_FRAME_DIST_BW_TRAIN: ' + str(knn_frame_dist_bw_train) +
      '; KNN_FRAME_DIST_FW_VAL: ' + str(knn_frame_dist_fw_val) +
      '; KNN_FRAME_DIST_BW_VAL: ' + str(knn_frame_dist_bw_val) +
      '; HANDLE_ABSOLUTE_SIZE: ' + str(handle_absolute_size) +
      '; HANDLE_ABSOLUTE_SIZE: ' + str(handle_detector_confidence) +
      '; ABSOLUTE_DIFFERENCES: ' + str(absolute_differences) +
      '; NORMALISE_DISTANCES: ' + str(normalise_distances) +
      '; FILTER_DATASET_TRAIN: ' + str(knn_filter_dataset_train) +
      '; FILTER_DATASET_VAL: ' + str(knn_filter_dataset_val) +
      '; AUTOCORRELATE_FEAT: ' + str(auto_correlate_feat) +
      '; AUTOCORRELATE_KNN: ' + str(auto_correlate_knn) +
      '; AUTOCORRELATE_FILTER: ' + str(auto_correlate_filter) +
      '; ELEMENT_WISE_PRODUCT_FEEDER: ' + str(element_wise_product_feeder) +
      '; ELEMENT_WISE_PRODUCT_TYPE: ' + element_wise_product_type
      )

for iteration in range(iterations):
    print('\nRUN: ' + str(iteration + 1))
    # create training args object
    train_args = create_train_args(seed=seed_array[iteration],
                                   workers=workers,
                                   print_freq=print_freq,
                                   gpu=gpu_name,
                                   lr=learning_rate_train,
                                   momentum=momentum_train,
                                   weight_decay=weight_decay_train,
                                   epochs=epochs_train,
                                   batch_size=batch_size,
                                   k_at_hop=k_at_hop_train,
                                   active_connection=active_connections_train,
                                   log_directory=os.path.join('logs', log_date, log_timestamp),
                                   element_wise_products_feeder=element_wise_product_feeder,
                                   element_wise_products_type=element_wise_product_type,
                                   absolute_differences=absolute_differences,
                                   normalise_distances=normalise_distances
                                   )

    # create validation args object
    val_args = create_test_args(seed=seed_array[iteration],
                                workers=workers,
                                print_freq=print_freq,
                                gpu=gpu_name,
                                batch_size=batch_size,
                                k_at_hop=k_at_hop_val,
                                active_connection=active_connections_val,
                                log_directory=os.path.join('logs', log_date, log_timestamp),
                                element_wise_products_feeder=element_wise_product_feeder,
                                element_wise_products_type=element_wise_product_type,
                                absolute_differences=absolute_differences,
                                normalise_distances=normalise_distances
                                )

    # print args objects of current run for logging
    print('\nTRAIN SETTINGS: ')
    print(train_args)
    print('\nVALIDATION SETTINGS: ')
    print(val_args)

    # iterator needed for proper saving to eval_data file
    det_num = 0

    # TRAINING
    for (train_detector, val_detector, test_detector) in zip(train_seqs, val_seqs, test_seqs):
        print('\nTRAINING ' + str.upper(detector_types[det_num]) + ' DETECTORS')
        train_state_dict = None
        if training_type == 'combined':
            print('\nTRAINING COMBINED GCN')
            comb_train_labels = np.array([])
            comb_train_feat = np.array([])
            comb_train_knn_graph = np.array([])
            curr_max_label = 0
            for i, sequence in enumerate(train_detector):
                print('Training using ' + sequence[0] + '...')
                # load metadata
                meta_train = pd.read_csv(osp.join(train_directory, sequence[0], sequence[1]))

                # create combined dataset by concatenating feature datasets
                train_feat = np.array([])
                bbox_size_idx = None
                for input_feat in inputs[0]:
                    if train_feat.size == 0:
                        train_feat = np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        train_feat = np.concatenate(
                            (train_feat, np.load(
                                os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_feat:
                            train_feat = create_autocorrelation_dataset(train_feat)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            # drop absolute size column
                            train_feat = np.delete(train_feat, -1, axis=1)
                        elif handle_absolute_size == 'scale_by_dataset':
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            train_feat[:, -1] = (train_feat[:, -1] - train_feat[:, -1].min()) / \
                                                (train_feat[:, -1].max() - train_feat[:, -1].min())
                        elif handle_absolute_size == 'scale_by_batch':
                            # remember idx of absolute size column to later alter it during training time
                            bbox_size_idx = train_feat.shape[1] - 1
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            train_feat = np.delete(train_feat, -2, axis=1)

                # load knn features
                knn_feat = np.array([])
                for input_feat in inputs[1]:
                    if knn_feat.size == 0:
                        knn_feat = np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        knn_feat = np.concatenate((knn_feat, np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_knn:
                            knn_feat = create_autocorrelation_dataset(knn_feat)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            # drop absolute size column
                            knn_feat = np.delete(knn_feat, -1, axis=1)
                        # currently, if the absolute size shall be rescaled it is always rescaled according to the
                        # whole dataset; this is due to the fact that replicating the by_batch setting is difficult
                        # due to the randomization of the DataLoader
                        elif handle_absolute_size == 'scale_by_dataset' or handle_absolute_size == 'scale_by_batch':
                            # TODO: Come up with way to replicate rescaling by batch
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            knn_feat[:, -1] = (knn_feat[:, -1] - knn_feat[:, -1].min()) / \
                                              (knn_feat[:, -1].max() - knn_feat[:, -1].min())
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            knn_feat = np.delete(knn_feat, -2, axis=1)

                # create splits and adjust labeling so that it goes from 0 to num_classes in split
                train_labels = adjust_labeling(meta_train[inputs[2]].to_numpy())

                # load filter dataset (concat feature datasets)
                filter_dataset_train = np.array([])
                for input_feat in knn_filter_dataset_train:
                    if filter_dataset_train.size == 0:
                        filter_dataset_train = np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        filter_dataset_train = np.concatenate(
                            (filter_dataset_train, np.load(
                                os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_filter:
                            filter_dataset_train = create_autocorrelation_dataset(filter_dataset_train)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            # drop absolute size column
                            filter_dataset_train = np.delete(filter_dataset_train, -1, axis=1)
                        elif handle_absolute_size == 'scale_by_dataset' or handle_absolute_size == 'scale_by_batch':
                            # TODO: see above
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            filter_dataset_train[:, -1] = (filter_dataset_train[:, -1] - filter_dataset_train[:, -1]
                                                           .min()) / (filter_dataset_train[:, -1].max()
                                                                      - filter_dataset_train[:, -1].min())
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            filter_dataset_train = np.delete(filter_dataset_train, -2, axis=1)

                # create knn graph
                train_knn_graph = create_knn_graph_dataset(knn_type=knn_type_train,
                                                           features_file=knn_feat,
                                                           neighbors=knn_train,
                                                           knn_calculation=knn_method,
                                                           frames=meta_train['frame'].tolist(),
                                                           frame_dist_forward=knn_frame_dist_fw_train,
                                                           frame_dist_backward=knn_frame_dist_bw_train,
                                                           filter_dataset=filter_dataset_train
                                                           )

                # since we are in combined setting, we need to add the max label of the last iteration to ensure
                # consistent labeling across sequences
                train_labels += curr_max_label
                # set max label to new max label within labeling file
                curr_max_label = max(train_labels)

                # sequence array; added to the feature array as indication which record belongs to which sequence
                seq_array = np.full((len(train_feat), 1), i)
                train_feat = np.concatenate((seq_array, train_feat), axis=1)

                # combine dataset of sequences
                if comb_train_labels.size == 0:
                    comb_train_labels = train_labels
                    comb_train_feat = train_feat
                    comb_train_knn_graph = train_knn_graph
                else:
                    comb_train_labels = np.concatenate((comb_train_labels, train_labels), axis=0)
                    comb_train_feat = np.concatenate((comb_train_feat, train_feat), axis=0)
                    comb_train_knn_graph = np.concatenate((comb_train_knn_graph, train_knn_graph), axis=0)

            # print shapes of files
            print('TRAIN SHAPES COMBINED')
            print(comb_train_labels.shape)
            print(comb_train_feat.shape)
            print(comb_train_knn_graph.shape)

            # create knn graph plots
            plot_knn_graph(index=10,
                           run=iteration + 1,
                           detector_name=None,
                           detector_type=detector_types[det_num] + '_train',
                           log_directory=train_args.log_directory,
                           labels=comb_train_labels,
                           features=comb_train_feat,
                           knn=comb_train_knn_graph,
                           frames=meta_train['frame'].tolist(),
                           k_at_hop=k_at_hop_train,
                           active_connection=active_connections_train,
                           seed=train_args.seed,
                           element_wise_products_feeder=element_wise_product_feeder,
                           element_wise_products_type=element_wise_product_type,
                           absolute_differences=absolute_differences,
                           normalise_distances=normalise_distances
                           )

            plot_knn_graph(index='half',
                           run=iteration + 1,
                           detector_name=None,
                           detector_type=detector_types[det_num] + '_train',
                           log_directory=train_args.log_directory,
                           labels=comb_train_labels,
                           features=comb_train_feat,
                           knn=comb_train_knn_graph,
                           frames=meta_train['frame'].tolist(),
                           k_at_hop=k_at_hop_train,
                           active_connection=active_connections_train,
                           seed=train_args.seed,
                           element_wise_products_feeder=element_wise_product_feeder,
                           element_wise_products_type=element_wise_product_type,
                           absolute_differences=absolute_differences,
                           normalise_distances=normalise_distances
                           )

            # create embedding plots
            plot_embedding_graph(index=10,
                                 run=iteration + 1,
                                 detector_name=None,
                                 detector_type=detector_types[det_num] + '_train',
                                 log_directory=train_args.log_directory,
                                 labels=comb_train_labels,
                                 features=comb_train_feat,
                                 knn=comb_train_knn_graph,
                                 frames=meta_train['frame'].tolist(),
                                 k_at_hop=k_at_hop_train,
                                 active_connection=active_connections_train,
                                 seed=val_args.seed,
                                 element_wise_products_feeder=element_wise_product_feeder,
                                 element_wise_products_type=element_wise_product_type,
                                 absolute_differences=absolute_differences,
                                 normalise_distances=normalise_distances
                                 )

            plot_embedding_graph(index='half',
                                 run=iteration + 1,
                                 detector_name=None,
                                 detector_type=detector_types[det_num] + '_train',
                                 log_directory=train_args.log_directory,
                                 labels=comb_train_labels,
                                 features=comb_train_feat,
                                 knn=comb_train_knn_graph,
                                 frames=meta_train['frame'].tolist(),
                                 k_at_hop=k_at_hop_train,
                                 active_connection=active_connections_train,
                                 seed=val_args.seed,
                                 element_wise_products_feeder=element_wise_product_feeder,
                                 element_wise_products_type=element_wise_product_type,
                                 absolute_differences=absolute_differences,
                                 normalise_distances=normalise_distances
                                 )
            if element_wise_product_feeder:
                if element_wise_product_type == 'frame_pairwise':
                    input_channels = ((comb_train_feat.shape[1] - 1) * 2) + 1
                elif element_wise_product_type == 'pairwise':
                    input_channels = ((comb_train_feat.shape[1] - 1) * 3) + 1
            else:
                input_channels = comb_train_feat.shape[1]

            # Training
            train_args.lr = learning_rate_train
            train_state_dict, train_losses, train_avg_losses, val_losses, val_avg_losses = \
                train_main(train_args=train_args,
                           test_args=val_args,
                           detector_name=detector_types[det_num],
                           run=iteration + 1,
                           input_channels=input_channels,
                           features=comb_train_feat,
                           knn_graph=comb_train_knn_graph,
                           labels=comb_train_labels,
                           frames=meta_train['frame'].tolist(),
                           bbox_size_idx=bbox_size_idx
                           )

            eval_data.at[iteration, 'train_' + detector_types[det_num] + '_loss'] = train_losses
            eval_data.at[iteration, 'train_' + detector_types[det_num] + '_avg_loss'] = train_avg_losses
            eval_data.at[iteration, 'train_' + detector_types[det_num] + '_val_loss'] = val_losses
            eval_data.at[iteration, 'train_' + detector_types[det_num] + '_avg_val_loss'] = val_avg_losses
        elif training_type == 'sequential':
            for sequence in train_detector:
                print('Training ' + sequence[0] + '...')
                # load ground truth files
                gt_train = np.genfromtxt(os.path.join(gt_directory, sequence[0], 'gt/gt.txt'), dtype=float,
                                         delimiter=',')

                # load metadata
                meta_train = pd.read_csv(osp.join(train_directory, sequence[0], sequence[1]))

                # load features
                train_feat = np.array([])
                bbox_size_idx = None
                for input_feat in inputs[0]:
                    if train_feat.size == 0:
                        train_feat = np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        train_feat = np.concatenate(
                            (train_feat, np.load(
                                os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_feat:
                            train_feat = create_autocorrelation_dataset(train_feat)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            # drop absolute size column
                            train_feat = np.delete(train_feat, -1, axis=1)
                        elif handle_absolute_size == 'scale_by_dataset':
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            train_feat[:, -1] = (train_feat[:, -1] - train_feat[:, -1].min()) / \
                                                (train_feat[:, -1].max() - train_feat[:, -1].min())
                        elif handle_absolute_size == 'scale_by_batch':
                            bbox_size_idx = train_feat.shape[1] - 1
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            train_feat = np.delete(train_feat, -2, axis=1)

                # load knn features
                knn_feat = np.array([])
                for input_feat in inputs[1]:
                    if knn_feat.size == 0:
                        knn_feat = np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        knn_feat = np.concatenate((knn_feat, np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_knn:
                            knn_feat = create_autocorrelation_dataset(knn_feat)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            # drop absolute size column
                            knn_feat = np.delete(knn_feat, -1, axis=1)
                        # currently, if the absolute size shall be rescaled it is always rescaled according to the
                        # whole dataset; this is due to the fact that replicating the by_batch setting is difficult
                        # due to the randomization of the DataLoader
                        elif handle_absolute_size == 'scale_by_dataset' or handle_absolute_size == 'scale_by_batch':
                            # TODO: see above
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            knn_feat[:, -1] = (knn_feat[:, -1] - knn_feat[:, -1].min()) / \
                                              (knn_feat[:, -1].max() - knn_feat[:, -1].min())
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            knn_feat = np.delete(knn_feat, -2, axis=1)

                # create splits and adjust labeling so that it goes from 0 to num_classes in split
                train_labels = adjust_labeling(meta_train[inputs[2]].to_numpy())

                # load filter dataset
                filter_dataset_train = np.array([])
                for input_feat in knn_filter_dataset_train:
                    if filter_dataset_train.size == 0:
                        filter_dataset_train = np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        filter_dataset_train = np.concatenate(
                            (filter_dataset_train, np.load(
                                os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_filter:
                            filter_dataset_train = create_autocorrelation_dataset(filter_dataset_train)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            # drop absolute size column
                            filter_dataset_train = np.delete(filter_dataset_train, -1, axis=1)
                        # currently, if the absolute size shall be rescaled it is always rescaled according to the
                        # whole dataset; this is due to the fact that replicating the by_batch setting is difficult
                        # due to the randomization of the DataLoader
                        elif handle_absolute_size == 'scale_by_dataset' or handle_absolute_size == 'scale_by_batch':
                            # TODO: see above
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            filter_dataset_train[:, -1] = (filter_dataset_train[:, -1] - filter_dataset_train[:, -1]
                                                           .min()) / (filter_dataset_train[:, -1].max() -
                                                                      filter_dataset_train[:, -1].min())
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            filter_dataset_train = np.delete(filter_dataset_train, -2, axis=1)

                # create knn graph
                train_knn_graph = create_knn_graph_dataset(knn_type=knn_type_train,
                                                           features_file=knn_feat,
                                                           neighbors=knn_train,
                                                           knn_calculation=knn_method,
                                                           frames=meta_train['frame'].tolist(),
                                                           frame_dist_forward=knn_frame_dist_fw_train,
                                                           frame_dist_backward=knn_frame_dist_bw_train,
                                                           filter_dataset=knn_filter_dataset_train
                                                           )

                # print shapes of files
                print('TRAIN SHAPES ' + sequence[0])
                print(train_labels.shape)
                print(train_feat.shape)
                print(train_knn_graph.shape)
                if element_wise_product_feeder:
                    if element_wise_product_type == 'frame_pairwise':
                        input_channels = train_feat.shape[1] * 2
                    elif element_wise_product_type == 'pairwise':
                        input_channels = train_feat.shape[1] * 3
                else:
                    input_channels = train_feat.shape[1]

                # check whether train_state_dict is not none (i.e. one training of sequential process was completed)
                # if it is not none, then use to initialise new network
                if train_state_dict is not None:
                    # reinitialise learning rate
                    train_args.lr = learning_rate_train
                    train_state_dict, train_losses, train_avg_losses, val_losses, val_avg_losses = \
                        train_main(train_args=train_args,
                                   test_args=val_args,
                                   detector_name=detector_types[det_num],
                                   run=iteration + 1,
                                   input_channels=input_channels,
                                   features=train_feat,
                                   knn_graph=train_knn_graph,
                                   labels=train_labels,
                                   frames=meta_train['frame'].tolist(),
                                   bbox_size_idx=bbox_size_idx,
                                   state_dict=train_state_dict
                                   )
                else:
                    # reinitialise learning rate
                    train_args.lr = learning_rate_train
                    train_state_dict, train_losses, train_avg_losses, val_losses, val_avg_losses = \
                        train_main(train_args=train_args,
                                   test_args=val_args,
                                   detector_name=detector_types[det_num],
                                   run=iteration + 1,
                                   input_channels=input_channels,
                                   features=train_feat,
                                   knn_graph=train_knn_graph,
                                   labels=train_labels,
                                   frames=meta_train['frame'].tolist(),
                                   bbox_size_idx=bbox_size_idx,
                                   )

                # save losses to evaluation dataframe
                eval_data.at[iteration, 'train_' + detector_types[det_num] + '_' + sequence[0] + '_loss'] = \
                    train_losses
                eval_data.at[iteration, 'train_' + detector_types[det_num] + '_' + sequence[0] + '_avg_loss'] = \
                    train_avg_losses
                eval_data.at[iteration, 'train_' + detector_types[det_num] + '_' + sequence[0] + '_val_loss'] = \
                    val_losses
                eval_data.at[
                    iteration, 'train_' + detector_types[det_num] + '_' + sequence[0] + '_avg_val_loss'] = \
                    val_avg_losses
        det_num += 1
        # VALIDATION
        if not skip_validation:
            for sequence in val_detector:
                print('Validating ' + sequence[0] + '...')
                # load ground truth files
                gt_val = np.genfromtxt(os.path.join(gt_directory, sequence[0], 'gt/gt.txt'), dtype=float,
                                       delimiter=',')

                # load metadata and detection file
                meta_val = pd.read_csv(osp.join(train_directory, sequence[0], sequence[1]))
                val_det_file = np.loadtxt(os.path.join(train_directory, sequence[0], 'det/det.txt'), delimiter=",")

                # load features
                val_feat = np.array([])
                bbox_size_idx = None
                for input_feat in inputs[0]:
                    if val_feat.size == 0:
                        val_feat = np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        val_feat = np.concatenate((val_feat, np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_feat:
                            val_feat = create_autocorrelation_dataset(val_feat)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            # drop absolute size column
                            val_feat = np.delete(val_feat, -1, axis=1)
                        elif handle_absolute_size == 'scale_by_dataset':
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            val_feat[:, -1] = (val_feat[:, -1] - val_feat[:, -1].min()) / \
                                              (val_feat[:, -1].max() - val_feat[:, -1].min())
                        elif handle_absolute_size == 'scale_by_batch':
                            bbox_size_idx = val_feat.shape[1] - 1
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            val_feat = np.delete(val_feat, -2, axis=1)

                # load knn features
                knn_feat = np.array([])
                for input_feat in inputs[1]:
                    if knn_feat.size == 0:
                        knn_feat = np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        knn_feat = np.concatenate(
                            (knn_feat, np.load(
                                os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_knn:
                            knn_feat = create_autocorrelation_dataset(knn_feat)
                    # how to handle reid dataset
                    if input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            print('\n Dropped Absolute Size for ReID')
                            knn_feat = np.delete(knn_feat, -1, axis=1)
                        # currently, if the absolute size shall be rescaled it is always rescaled according to the
                        # whole dataset; this is due to the fact that replicating the by_batch setting is difficult
                        # due to the randomization of the DataLoader
                        elif handle_absolute_size == 'scale_by_dataset' or handle_absolute_size == 'scale_by_batch':
                            knn_feat[:, -1] = (knn_feat[:, -1] - knn_feat[:, -1].min()) / \
                                              (knn_feat[:, -1].max() - knn_feat[:, -1].min())
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            knn_feat = np.delete(knn_feat, -2, axis=1)

                # create splits and adjust labeling so that it goes from 0 to num_classes in split
                val_labels = adjust_labeling(meta_val[inputs[3]].to_numpy())

                # load filter dataset
                filter_dataset_val = np.array([])
                for input_feat in knn_filter_dataset_val:
                    if filter_dataset_val.size == 0:
                        filter_dataset_val = np.load(
                            os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        filter_dataset_val = np.concatenate(
                            (filter_dataset_val, np.load(
                                os.path.join(train_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_filter:
                            filter_dataset_val = create_autocorrelation_dataset(filter_dataset_val)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            print('\n Dropped Absolute Size for ReID')
                            filter_dataset_val = np.delete(filter_dataset_val, -1, axis=1)
                        # currently, if the absolute size shall be rescaled it is always rescaled according to the
                        # whole dataset; this is due to the fact that replicating the by_batch setting is difficult
                        # due to the randomization of the DataLoader
                        elif handle_absolute_size == 'scale_by_dataset' or handle_absolute_size == 'scale_by_batch':
                            # TODO: see above
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            filter_dataset_val[:, -1] = (filter_dataset_val[:, -1] -
                                                         filter_dataset_val[:, -1].min()) / \
                                                        (filter_dataset_val[:, -1].max() -
                                                         filter_dataset_val[:, -1].min())
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            filter_dataset_val = np.delete(filter_dataset_val, -2, axis=1)

                # create knn graph
                val_knn_graph = create_knn_graph_dataset(knn_type=knn_type_val,
                                                         features_file=knn_feat,
                                                         neighbors=knn_val,
                                                         knn_calculation=knn_method,
                                                         frames=meta_val['frame'].tolist(),
                                                         frame_dist_forward=knn_frame_dist_fw_val,
                                                         frame_dist_backward=knn_frame_dist_bw_val,
                                                         filter_dataset=filter_dataset_val
                                                         )

                # if combined setting was used, need to append a 0 to the feature saying that all feature records
                # are from the same sequence
                if training_type == 'combined':
                    seq_array = np.zeros((len(val_feat), 1))
                    val_feat = np.concatenate((seq_array, val_feat), axis=1)

                # print shapes of files
                print('VALIDATION SHAPES ' + sequence[0])
                print(val_labels.shape)
                print(val_feat.shape)
                print(val_knn_graph.shape)

                # create knn graph plots
                plot_knn_graph(index=10,
                               run=iteration + 1,
                               detector_name=sequence[0],
                               detector_type='test',
                               log_directory=train_args.log_directory,
                               labels=val_labels,
                               features=val_feat,
                               knn=val_knn_graph,
                               frames=meta_val['frame'].tolist(),
                               k_at_hop=k_at_hop_train,
                               active_connection=active_connections_train,
                               seed=val_args.seed,
                               element_wise_products_feeder=element_wise_product_feeder,
                               element_wise_products_type=element_wise_product_type,
                               absolute_differences=absolute_differences,
                               normalise_distances=normalise_distances
                               )

                plot_knn_graph(index='half',
                               run=iteration + 1,
                               detector_name=sequence[0],
                               detector_type='test',
                               log_directory=train_args.log_directory,
                               labels=val_labels,
                               features=val_feat,
                               knn=val_knn_graph,
                               frames=meta_val['frame'].tolist(),
                               k_at_hop=k_at_hop_train,
                               active_connection=active_connections_train,
                               seed=val_args.seed,
                               element_wise_products_feeder=element_wise_product_feeder,
                               element_wise_products_type=element_wise_product_type,
                               absolute_differences=absolute_differences,
                               normalise_distances=normalise_distances
                               )

                # create embedding plots
                plot_embedding_graph(index=10,
                                     run=iteration + 1,
                                     detector_name=sequence[0],
                                     detector_type='test',
                                     log_directory=train_args.log_directory,
                                     labels=val_labels,
                                     features=val_feat,
                                     knn=val_knn_graph,
                                     frames=meta_val['frame'].tolist(),
                                     k_at_hop=k_at_hop_train,
                                     active_connection=active_connections_train,
                                     seed=val_args.seed,
                                     element_wise_products_feeder=element_wise_product_feeder,
                                     element_wise_products_type=element_wise_product_type,
                                     absolute_differences=absolute_differences,
                                     normalise_distances=normalise_distances
                                     )

                plot_embedding_graph(index='half',
                                     run=iteration + 1,
                                     detector_name=sequence[0],
                                     detector_type='test',
                                     log_directory=train_args.log_directory,
                                     labels=val_labels,
                                     features=val_feat,
                                     knn=val_knn_graph,
                                     frames=meta_val['frame'].tolist(),
                                     k_at_hop=k_at_hop_train,
                                     active_connection=active_connections_train,
                                     seed=val_args.seed,
                                     element_wise_products_feeder=element_wise_product_feeder,
                                     element_wise_products_type=element_wise_product_type,
                                     absolute_differences=absolute_differences,
                                     normalise_distances=normalise_distances
                                     )

                if element_wise_product_feeder:
                    if element_wise_product_type == 'frame_pairwise':
                        if training_type == 'combined':
                            input_channels = ((val_feat.shape[1] - 1) * 2) + 1
                        elif training_type == 'sequential':
                            input_channels = val_feat.shape[1] * 2
                    elif element_wise_product_type == 'pairwise':
                        if training_type == 'combined':
                            input_channels = ((val_feat.shape[1] - 1) * 3) + 1
                        elif training_type == 'sequential':
                            input_channels = val_feat.shape[1] * 3
                else:
                    input_channels = val_feat.shape[1]

                # Testing
                if removed:
                    val_pred, rm_filter, val_losses, val_avg_losses, edges, scores = \
                        val_main(state_dict=train_state_dict,
                                 args=val_args,
                                 input_channels=input_channels,
                                 features=val_feat,
                                 knn_graph=val_knn_graph,
                                 labels=val_labels,
                                 bbox_size_idx=bbox_size_idx,
                                 removed=removed
                                 )
                else:
                    val_pred, val_losses, val_avg_losses, edges, scores = \
                        val_main(state_dict=train_state_dict,
                                 args=val_args,
                                 input_channels=input_channels,
                                 features=val_feat,
                                 knn_graph=val_knn_graph,
                                 labels=val_labels,
                                 bbox_size_idx=bbox_size_idx,
                                 removed=removed
                                 )

                print('\n CREATING 512 FEATURE MAP')
                feature_512 = obtain_512_feature_map(train_state_dict, val_args, input_channels,
                                                     val_feat, val_knn_graph, val_labels, bbox_size_idx)

                # create knn graph plots
                plot_knn_graph(index=10,
                               run=iteration + 1,
                               detector_name=sequence[0],
                               detector_type='test_512',
                               log_directory=train_args.log_directory,
                               labels=val_labels,
                               features=feature_512,
                               knn=val_knn_graph,
                               frames=meta_val['frame'].tolist(),
                               k_at_hop=k_at_hop_train,
                               active_connection=active_connections_train,
                               seed=val_args.seed,
                               element_wise_products_feeder=element_wise_product_feeder,
                               element_wise_products_type=element_wise_product_type,
                               absolute_differences=absolute_differences,
                               normalise_distances=normalise_distances
                               )

                plot_knn_graph(index='half',
                               run=iteration + 1,
                               detector_name=sequence[0],
                               detector_type='test_512',
                               log_directory=train_args.log_directory,
                               labels=val_labels,
                               features=feature_512,
                               knn=val_knn_graph,
                               frames=meta_val['frame'].tolist(),
                               k_at_hop=k_at_hop_train,
                               active_connection=active_connections_train,
                               seed=val_args.seed,
                               element_wise_products_feeder=element_wise_product_feeder,
                               element_wise_products_type=element_wise_product_type,
                               absolute_differences=absolute_differences,
                               normalise_distances=normalise_distances
                               )

                # create embedding plots
                plot_embedding_graph(index=10,
                                     run=iteration + 1,
                                     detector_name=sequence[0],
                                     detector_type='test_512',
                                     log_directory=train_args.log_directory,
                                     labels=val_labels,
                                     features=feature_512,
                                     knn=val_knn_graph,
                                     frames=meta_val['frame'].tolist(),
                                     k_at_hop=k_at_hop_train,
                                     active_connection=active_connections_train,
                                     seed=val_args.seed,
                                     element_wise_products_feeder=element_wise_product_feeder,
                                     element_wise_products_type=element_wise_product_type,
                                     absolute_differences=absolute_differences,
                                     normalise_distances=normalise_distances
                                     )

                plot_embedding_graph(index='half',
                                     run=iteration + 1,
                                     detector_name=sequence[0],
                                     detector_type='test_512',
                                     log_directory=train_args.log_directory,
                                     labels=val_labels,
                                     features=feature_512,
                                     knn=val_knn_graph,
                                     frames=meta_val['frame'].tolist(),
                                     k_at_hop=k_at_hop_train,
                                     active_connection=active_connections_train,
                                     seed=val_args.seed,
                                     element_wise_products_feeder=element_wise_product_feeder,
                                     element_wise_products_type=element_wise_product_type,
                                     absolute_differences=absolute_differences,
                                     normalise_distances=normalise_distances
                                     )

                # save 512 feature maps as numpy arrays
                if val_args.save_feature_map:
                    print('\n SAVING 512 FEATURE MAP')
                    mkdir_if_missing(os.path.join(val_args.log_directory, 'feature_maps'))
                    np.save(os.path.join(val_args.log_directory, 'feature_maps', '512_' + sequence[0] + '.npy'),
                            feature_512)

                # create predictions
                if removed:
                    val_pred, val_pred_rm = create_validation_output_file(meta_val, val_pred, None, rm_filter)
                else:
                    val_pred, val_pred_rm = create_validation_output_file(meta_val, val_pred, None, None)

                # save validation losses to evaluation dataframe
                eval_data.at[iteration, sequence[0] + '_loss'] = val_losses
                eval_data.at[iteration, sequence[0] + '_avg_loss'] = val_avg_losses

                # save prediction results to files as well as edges and scores files
                mkdir_if_missing(os.path.join('logs', log_date, log_timestamp, 'run_' + str(iteration + 1), 'full',
                                              sequence[0]))
                np.savetxt(os.path.join('logs', log_date, log_timestamp, 'run_' + str(iteration + 1), 'full',
                                        sequence[0], 'eval_input.txt'), val_pred, delimiter=' ')
                if removed:
                    mkdir_if_missing(
                        os.path.join('logs', log_date, log_timestamp, 'run_' + str(iteration + 1), 'removed',
                                     sequence[0]))
                    np.savetxt(os.path.join('logs', log_date, log_timestamp, 'run_' + str(iteration + 1), 'removed',
                                            sequence[0], 'eval_input.txt'), val_pred_rm, delimiter=' ')
                if graph_heuristic:
                    if graph_heuristic:
                        create_heurisitc_output_file(output_dir=os.path.join('logs', log_date, log_timestamp,
                                                                             'run_' + str(iteration + 1), 'full',
                                                                             sequence[0]),
                                                     det_file=val_det_file,
                                                     edges=edges,
                                                     scores=scores,
                                                     add_edges=add_dummy_edges)
        # TESTING
        if not skip_testing:
            for sequence in test_detector:
                print('Testing ' + sequence[0] + '...')

                # load detection file
                test_det_file = np.loadtxt(os.path.join(test_directory, sequence[0], 'det/det.txt'), delimiter=",")

                # load features
                test_feat = np.array([])
                bbox_size_idx = None
                for input_feat in inputs[0]:
                    if test_feat.size == 0:
                        test_feat = np.load(
                            os.path.join(test_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        test_feat = np.concatenate((test_feat, np.load(
                            os.path.join(test_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_feat:
                            test_feat = create_autocorrelation_dataset(test_feat)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            # drop absolute size column
                            test_feat = np.delete(test_feat, -1, axis=1)
                        elif handle_absolute_size == 'scale_by_dataset':
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            test_feat[:, -1] = (test_feat[:, -1] - test_feat[:, -1].min()) / \
                                               (test_feat[:, -1].max() - test_feat[:, -1].min())
                        elif handle_absolute_size == 'scale_by_batch':
                            bbox_size_idx = test_feat.shape[1] - 1
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            test_feat = np.delete(test_feat, -2, axis=1)

                # load knn features
                knn_feat = np.array([])
                for input_feat in inputs[1]:
                    if knn_feat.size == 0:
                        knn_feat = np.load(os.path.join(test_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        knn_feat = np.concatenate(
                            (knn_feat, np.load(
                                os.path.join(test_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_knn:
                            knn_feat = create_autocorrelation_dataset(knn_feat)
                    # how to handle reid dataset
                    if input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            print('\n Dropped Absolute Size for ReID')
                            knn_feat = np.delete(knn_feat, -1, axis=1)
                        # currently, if the absolute size shall be rescaled it is always rescaled according to the
                        # whole dataset; this is due to the fact that replicating the by_batch setting is difficult
                        # due to the randomization of the DataLoader
                        elif handle_absolute_size == 'scale_by_dataset' or handle_absolute_size == 'scale_by_batch':
                            knn_feat[:, -1] = (knn_feat[:, -1] - knn_feat[:, -1].min()) / \
                                              (knn_feat[:, -1].max() - knn_feat[:, -1].min())
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            knn_feat = np.delete(knn_feat, -2, axis=1)

                # load filter dataset
                filter_dataset_val = np.array([])
                for input_feat in knn_filter_dataset_val:
                    if filter_dataset_val.size == 0:
                        filter_dataset_val = np.load(
                            os.path.join(test_directory, sequence[0], 'feat_' + input_feat + '.npy'))
                    else:
                        filter_dataset_val = np.concatenate(
                            (filter_dataset_val, np.load(
                                os.path.join(test_directory, sequence[0], 'feat_' + input_feat + '.npy'))), axis=1)
                    # how to handle spatial dataset
                    if input_feat == 'spa':
                        if auto_correlate_filter:
                            filter_dataset_val = create_autocorrelation_dataset(filter_dataset_val)
                    # how to handle reid dataset
                    elif input_feat == 'extra':
                        if handle_absolute_size == 'drop':
                            print('\n Dropped Absolute Size for ReID')
                            filter_dataset_val = np.delete(filter_dataset_val, -1, axis=1)
                        # currently, if the absolute size shall be rescaled it is always rescaled according to the
                        # whole dataset; this is due to the fact that replicating the by_batch setting is difficult
                        # due to the randomization of the DataLoader
                        elif handle_absolute_size == 'scale_by_dataset' or handle_absolute_size == 'scale_by_batch':
                            # TODO: see above
                            # rescale absolute size by dataset, i.e. between largest bbox being 1 and smallest 0
                            filter_dataset_val[:, -1] = (filter_dataset_val[:, -1] -
                                                         filter_dataset_val[:, -1].min()) / \
                                                        (filter_dataset_val[:, -1].max() -
                                                         filter_dataset_val[:, -1].min())
                        if handle_detector_confidence == 'drop':
                            # drop absolute size column
                            filter_dataset_val = np.delete(filter_dataset_val, -2, axis=1)

                # create knn graph
                test_knn_graph = create_knn_graph_dataset(knn_type=knn_type_val,
                                                          features_file=knn_feat,
                                                          neighbors=knn_val,
                                                          knn_calculation=knn_method,
                                                          frames=test_det_file[:, 0],
                                                          frame_dist_forward=knn_frame_dist_fw_val,
                                                          frame_dist_backward=knn_frame_dist_bw_val,
                                                          filter_dataset=filter_dataset_val
                                                          )

                # if combined setting was used, then we need to append a 0 to the feature saying that all feature
                # records are from the same sequence
                if training_type == 'combined':
                    seq_array = np.zeros((len(test_feat), 1))
                    test_feat = np.concatenate((seq_array, test_feat), axis=1)

                # print shapes of files
                print('TESTING SHAPES ' + sequence[0])
                print(test_feat.shape)
                print(test_knn_graph.shape)

                if element_wise_product_feeder:
                    if element_wise_product_type == 'frame_pairwise':
                        if training_type == 'combined':
                            input_channels = ((test_feat.shape[1] - 1) * 2) + 1
                        elif training_type == 'sequential':
                            input_channels = test_feat.shape[1] * 2
                    elif element_wise_product_type == 'pairwise':
                        if training_type == 'combined':
                            input_channels = ((test_feat.shape[1] - 1) * 3) + 1
                        elif training_type == 'sequential':
                            input_channels = test_feat.shape[1] * 3
                else:
                    input_channels = test_feat.shape[1]

                # Testing
                test_pred, edges, scores = test_main(state_dict=train_state_dict,
                                                     args=val_args,
                                                     input_channels=input_channels,
                                                     features=test_feat,
                                                     knn_graph=test_knn_graph,
                                                     bbox_size_idx=bbox_size_idx
                                                     )

                # create predictions
                test_pred = create_testing_output_file(test_det_file, test_pred)

                # save prediction results to files as well as edges and scores files
                mkdir_if_missing(os.path.join('logs', log_date, log_timestamp, 'run_' + str(iteration + 1), 'full',
                                              sequence[0]))
                np.savetxt(os.path.join('logs', log_date, log_timestamp, 'run_' + str(iteration + 1), 'full',
                                        sequence[0], 'eval_input.txt'), test_pred, delimiter=' ')
                if graph_heuristic:
                    create_heurisitc_output_file(output_dir=os.path.join('logs', log_date, log_timestamp,
                                                                         'run_' + str(iteration + 1), 'full',
                                                                         sequence[0]),
                                                 det_file=test_det_file,
                                                 edges=edges,
                                                 scores=scores,
                                                 add_edges=add_dummy_edges,
                                                 )

    # calculate time data creation took
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nFinal time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))

# save evaluation dataframe as csv
eval_data.to_csv(os.path.join('logs', log_date, log_timestamp, 'eval_data.csv'), index=False)
