import os
import os.path as osp
import sys
import time

import numpy as np
import pandas as pd

from gcn_clustering.train import train_main
from gcn_clustering.test import test_main

from dataset_creation import create_knn_graph_dataset
from gcn_clustering.utils.logging import Logger
from gcn_clustering.utils.osutils import mkdir_if_missing

# general
from misc import adjust_labeling, create_train_args, create_test_args, create_prediction_output_file, \
    create_modified_ground_truth_file

date = time.strftime("%Y%m%d")
timestamp = time.strftime("%H%M%S")
directory = '../data/MOT/MOT17'

train_sequence = 'MOT17-04'
val_sequence = 'MOT17-09'
test_sequence = 'MOT17-02'

# METADATA PATHS
# metadata paths training
path_meta_train_dpm = '20200704-190814_train_metadata.csv'
path_meta_train_frcnn = '20200704-193423_train_metadata.csv'
path_meta_train_sdp = '20200704-183451_train_metadata.csv'

# metadata paths validation
path_meta_val_dpm = '20200705-115238_valid_metadata.csv'
path_meta_val_frcnn = '20200705-115400_valid_metadata.csv'
path_meta_val_sdp = '20200705-115519_valid_metadata.csv'

# metadata paths testing
path_meta_test_dpm = '20200704-143638_test_metadata.csv'
path_meta_test_frcnn = '20200704-143943_test_metadata.csv'
path_meta_test_sdp = '20200704-144343_test_metadata.csv'

# GROUND TRUTH PATHS
gt_train = np.loadtxt(os.path.join(directory, train_sequence + '-DPM', 'gt/gt.txt'), delimiter=',')
gt_val = np.loadtxt(os.path.join(directory, val_sequence + '-DPM', 'gt/gt.txt'), delimiter=',')
gt_test = np.loadtxt(os.path.join(directory, test_sequence + '-DPM', 'gt/gt.txt'), delimiter=',')

# parameters
# general
gpu_name = 'cuda:3'
workers = 20
print_freq = 100
seed = 1
knn = 200
knn_method = 'brute'
input_channels = 2048 + 5

label_column_train = 'labels_0.7_0.3_zero'
label_column_valid = 'labels_0.5_0.3_zero'
label_column_test = 'labels_0.5_0.3_zero'

iou_column_train = 'fil_train_0.7_0.3'
iou_column_valid = 'fil_valid_0.7_0.3'
iou_column_test = 'fil_test_0.5_0.3'

# training
k_at_hop_training = [200, 10]
active_connections_training = 10
batch_size_training = 16
epochs_training = 4
weight_decay_training = 1e-4
momentum_training = 0.9
learning_rate_training = 1e-2
save_checkpoints = False

# validation/ testing
k_at_hop_testing = [20, 5]
active_connections_testing = 5
learning_rate_testing = 1e-5
momentum_testing = 0.9
weight_decay_testing = 1e-4
epochs_testing = 20
batch_size_testing = 32
use_checkpoint = False

dpm_checkpoint = osp.join('../logs', 'trained_appearance_dpm.ckpt')
frcnn_checkpoint = osp.join('../logs', 'trained_appearance_frcnn.ckpt')
sdp_checkpoint = osp.join('../logs', 'trained_appearance_sdp.ckpt')

# load metadata files
meta_train_dpm = pd.read_csv(osp.join(directory, train_sequence + '-DPM', path_meta_train_dpm))
meta_train_frcnn = pd.read_csv(osp.join(directory, train_sequence + '-FRCNN', path_meta_train_frcnn))
meta_train_sdp = pd.read_csv(osp.join(directory, train_sequence + '-SDP', path_meta_train_sdp))

meta_val_dpm = pd.read_csv(osp.join(directory, val_sequence + '-DPM', path_meta_val_dpm))
meta_val_frcnn = pd.read_csv(osp.join(directory, val_sequence + '-FRCNN', path_meta_val_frcnn))
meta_val_sdp = pd.read_csv(osp.join(directory, val_sequence + '-SDP', path_meta_val_sdp))

meta_test_dpm = pd.read_csv(osp.join(directory, test_sequence + '-DPM', path_meta_test_dpm))
meta_test_frcnn = pd.read_csv(osp.join(directory, test_sequence + '-FRCNN', path_meta_test_frcnn))
meta_test_sdp = pd.read_csv(osp.join(directory, test_sequence + '-SDP', path_meta_test_sdp))

# load appearance features
train_feat_app_dpm = np.load(osp.join(directory, train_sequence + '-DPM', 'feat_app_pool.npy'))
train_feat_app_frcnn = np.load(osp.join(directory, train_sequence + '-FRCNN', 'feat_app_pool.npy'))
train_feat_app_sdp = np.load(os.path.join(directory, train_sequence + '-SDP', 'feat_app_pool.npy'))

val_feat_app_dpm = np.load(os.path.join(directory, val_sequence + '-DPM', 'feat_app_pool.npy'))
val_feat_app_frcnn = np.load(os.path.join(directory, val_sequence + '-FRCNN', 'feat_app_pool.npy'))
val_feat_app_sdp = np.load(os.path.join(directory, val_sequence + '-SDP', 'feat_app_pool.npy'))

test_feat_app_dpm = np.load(os.path.join(directory, test_sequence + '-DPM', 'feat_app_pool.npy'))
test_feat_app_frcnn = np.load(os.path.join(directory, test_sequence + '-FRCNN', 'feat_app_pool.npy'))
test_feat_app_sdp = np.load(os.path.join(directory, test_sequence + '-SDP', 'feat_app_pool.npy'))

# load spatial features
train_feat_spa_dpm = np.load(os.path.join(directory, train_sequence + '-DPM', 'feat_spa.npy'))
train_feat_spa_frcnn = np.load(os.path.join(directory, train_sequence + '-FRCNN', 'feat_spa.npy'))
train_feat_spa_sdp = np.load(os.path.join(directory, train_sequence + '-SDP', 'feat_spa.npy'))

val_feat_spa_dpm = np.load(os.path.join(directory, val_sequence + '-DPM', 'feat_spa.npy'))
val_feat_spa_frcnn = np.load(os.path.join(directory, val_sequence + '-FRCNN', 'feat_spa.npy'))
val_feat_spa_sdp = np.load(os.path.join(directory, val_sequence + '-SDP', 'feat_spa.npy'))

test_feat_spa_dpm = np.load(os.path.join(directory, test_sequence + '-DPM', 'feat_spa.npy'))
test_feat_spa_frcnn = np.load(os.path.join(directory, test_sequence + '-FRCNN', 'feat_spa.npy'))
test_feat_spa_sdp = np.load(os.path.join(directory, test_sequence + '-SDP', 'feat_spa.npy'))

# create combined feature dataset
train_feat_comb_dpm = np.concatenate((train_feat_app_dpm, train_feat_spa_dpm), axis=1)
train_feat_comb_frcnn = np.concatenate((train_feat_app_frcnn, train_feat_spa_frcnn), axis=1)
train_feat_comb_sdp = np.concatenate((train_feat_app_sdp, train_feat_spa_sdp), axis=1)

valid_feat_comb_dpm = np.concatenate((val_feat_app_dpm, val_feat_spa_dpm), axis=1)
valid_feat_comb_frcnn = np.concatenate((val_feat_app_frcnn, val_feat_spa_frcnn), axis=1)
valid_feat_comb_sdp = np.concatenate((val_feat_app_sdp, val_feat_spa_sdp), axis=1)

test_feat_comb_dpm = np.concatenate((test_feat_app_dpm, test_feat_spa_dpm), axis=1)
test_feat_comb_frcnn = np.concatenate((test_feat_app_frcnn, test_feat_spa_frcnn), axis=1)
test_feat_comb_sdp = np.concatenate((test_feat_app_sdp, test_feat_spa_sdp), axis=1)

# CHOOSE HERE WHICH ONES TO USE
train_feat_dpm = train_feat_comb_dpm
train_feat_frcnn = train_feat_comb_frcnn
train_feat_sdp = train_feat_comb_sdp

val_feat_dpm = valid_feat_comb_dpm
val_feat_frcnn = valid_feat_comb_frcnn
val_feat_sdp = valid_feat_comb_sdp

test_feat_dpm = test_feat_comb_dpm
test_feat_frcnn = test_feat_comb_frcnn
test_feat_sdp = test_feat_comb_sdp

del train_feat_app_dpm, train_feat_app_frcnn, train_feat_app_sdp, val_feat_app_dpm, val_feat_app_frcnn, \
    val_feat_app_sdp, test_feat_app_dpm, test_feat_app_frcnn, test_feat_app_sdp, train_feat_spa_dpm, \
    train_feat_spa_frcnn, train_feat_spa_sdp, val_feat_spa_dpm, val_feat_spa_frcnn, val_feat_spa_sdp, \
    test_feat_spa_dpm, test_feat_spa_frcnn, test_feat_spa_sdp, train_feat_comb_dpm, train_feat_comb_frcnn, \
    train_feat_comb_sdp, valid_feat_comb_dpm, valid_feat_comb_frcnn, valid_feat_comb_sdp, test_feat_comb_dpm, \
    test_feat_comb_frcnn, test_feat_comb_sdp

# saves logs to a file (standard output redirected)
sys.stdout = Logger(os.path.join('../logs', date, timestamp, 'log.txt'))

####################################################################################################################
# Summary of settings for log purposes:

print('Training using: ' + train_sequence)
print('Training meta (DPM): ' + path_meta_train_dpm)
print('Training meta (FRCNN): ' + path_meta_train_frcnn)
print('Training meta (SDP): ' + path_meta_train_sdp)

print('Validating using: ' + val_sequence)
print('Validation meta (DPM): ' + path_meta_val_dpm)
print('Validation meta (FRCNN): ' + path_meta_val_frcnn)
print('Validation meta (SDP): ' + path_meta_val_sdp)

print('Testing using: ' + test_sequence)
print('Testing meta (DPM): ' + path_meta_test_dpm)
print('Testing meta (FRCNN): ' + path_meta_test_frcnn)
print('Testing meta (SDP): ' + path_meta_test_sdp)

########################################################################################################################

print('START TRAINING DPM DETECTOR.....')
# create splits and adjust labeling so that it goes from 0 to num_classes in split
train_labels_dpm = adjust_labeling(meta_train_dpm[label_column_train].to_numpy()[meta_train_dpm[iou_column_train]])
val_labels_dpm = adjust_labeling(meta_val_dpm[label_column_valid].to_numpy()[meta_val_dpm[iou_column_valid]])
test_labels_dpm = adjust_labeling(meta_test_dpm[label_column_test].to_numpy()[meta_test_dpm[iou_column_test]])

train_feat_dpm = train_feat_dpm[meta_train_dpm[iou_column_train]]
val_feat_dpm = val_feat_dpm[meta_val_dpm[iou_column_valid]]
test_feat_dpm = test_feat_dpm[meta_test_dpm[iou_column_test]]

# create knn graphs for each split (ensures labeling is correct)
# first checks whether the train/ test/ valid splits are too small for creating a KNN graph.
# if there are less instances the maximum size knn graph is created (i.e. len(split))

if len(train_labels_dpm) > knn:
    train_knn_graph_dpm = create_knn_graph_dataset(train_feat_dpm, knn, knn_method)
else:
    train_knn_graph_dpm = create_knn_graph_dataset(train_feat_dpm, len(train_labels_dpm)-1, knn_method)
if len(val_labels_dpm) > knn:
    val_knn_graph_dpm = create_knn_graph_dataset(val_feat_dpm, knn, knn_method)
else:
    val_knn_graph_dpm = create_knn_graph_dataset(val_feat_dpm, len(val_labels_dpm)-1, knn_method)
if len(test_labels_dpm) > knn:
    test_knn_graph_dpm = create_knn_graph_dataset(test_feat_dpm, knn, knn_method)
else:
    test_knn_graph_dpm = create_knn_graph_dataset(test_feat_dpm, len(test_labels_dpm)-1, knn_method)

# print shapes of files
print("TRAIN SHAPES DPM")
print(train_labels_dpm.shape)
print(train_feat_dpm.shape)
print(train_knn_graph_dpm.shape)

print("VALID SHAPES DPM")
print(val_labels_dpm.shape)
print(val_feat_dpm.shape)
print(val_knn_graph_dpm.shape)

print("TEST SHAPES DPM")
print(test_labels_dpm.shape)
print(test_feat_dpm.shape)
print(test_knn_graph_dpm.shape)

# training
train_args_dpm = create_train_args(input_channels=input_channels,
                                   seed=seed,
                                   workers=workers,
                                   print_freq=print_freq,
                                   gpu=gpu_name,
                                   lr=learning_rate_training,
                                   momentum=momentum_training,
                                   weight_decay=weight_decay_training,
                                   epochs=epochs_training,
                                   batch_size=batch_size_training,
                                   features=train_feat_dpm,
                                   knn_graph=train_knn_graph_dpm,
                                   labels=train_labels_dpm,
                                   k_at_hop=k_at_hop_training,
                                   active_connection=active_connections_training,
                                   save_checkpoints=save_checkpoints,
                                   checkpoint_directory=dpm_checkpoint
                                   )

train_state_dict_dpm = train_main(train_args_dpm)

# validation
val_args_dpm = create_test_args(seed=seed,
                                workers=workers,
                                print_freq=print_freq,
                                gpu=gpu_name,
                                lr=learning_rate_testing,
                                momentum=momentum_testing,
                                weight_decay=weight_decay_testing,
                                epochs=epochs_testing,
                                batch_size=batch_size_testing,
                                features=val_feat_dpm,
                                knn_graph=val_knn_graph_dpm,
                                labels=val_labels_dpm,
                                k_at_hop=k_at_hop_testing,
                                active_connection=active_connections_testing,
                                use_checkpoint=use_checkpoint,
                                checkpoint_directory=dpm_checkpoint,
                                input_channels=input_channels
                                )

val_pred_dpm, val_pred_removed_dpm = test_main(train_state_dict_dpm, val_args_dpm)
mkdir_if_missing(os.path.join('../logs', date, timestamp, val_sequence + '-DPM_val_full'))
#mkdir_if_missing(os.path.join('../logs', date, timestamp, val_sequence + '-DPM_val_removed'))

val_pred_dpm = create_prediction_output_file(meta_val_dpm, val_pred_dpm, iou_column_valid)
#val_pred_removed_dpm = val_pred_removed_dpm
val_mod_gt_dpm = create_modified_ground_truth_file(gt_val, meta_val_dpm, iou_column_valid)

np.savetxt(os.path.join('../logs', date, timestamp, val_sequence + '-DPM_val_full', 'eval_input.txt'), val_pred_dpm, delimiter=' ')
np.savetxt(os.path.join('../logs', date, timestamp, val_sequence + '-DPM_val_full', 'gt.txt'), val_mod_gt_dpm, delimiter=',')
#np.save(os.path.join('../logs', date, timestamp, val_sequence + '-DPM_val_removed', 'pred_removed.npy'), val_pred_removed_dpm)
#np.savetxt(os.path.join('../logs', date, timestamp, val_sequence + '-DPM_val_removed', 'gt.txt'), val_mod_gt_dpm, delimiter=',')

# testing
test_args_dpm = create_test_args(seed=seed,
                                 workers=workers,
                                 print_freq=print_freq,
                                 gpu=gpu_name,
                                 lr=learning_rate_testing,
                                 momentum=momentum_testing,
                                 weight_decay=weight_decay_testing,
                                 epochs=epochs_testing,
                                 batch_size=batch_size_testing,
                                 features=test_feat_dpm,
                                 knn_graph=test_knn_graph_dpm,
                                 labels=test_labels_dpm,
                                 k_at_hop=k_at_hop_testing,
                                 active_connection=active_connections_testing,
                                 use_checkpoint=use_checkpoint,
                                 checkpoint_directory=dpm_checkpoint,
                                 input_channels=input_channels
                                 )

test_pred_dpm, test_pred_removed_dpm = test_main(train_state_dict_dpm, test_args_dpm)
mkdir_if_missing(os.path.join('../logs', date, timestamp, test_sequence + '-DPM_test_full'))
#mkdir_if_missing(os.path.join('../logs', date, timestamp, test_sequence + '-DPM_test_removed'))

test_pred_dpm = create_prediction_output_file(meta_test_dpm, test_pred_dpm, iou_column_test)
#test_pred_removed_dpm = test_pred_removed_dpm
test_mod_gt_dpm = create_modified_ground_truth_file(gt_test, meta_test_dpm, iou_column_test)

np.savetxt(os.path.join('../logs', date, timestamp, test_sequence + '-DPM_test_full', 'eval_input.txt'), test_pred_dpm, delimiter=' ')
np.savetxt(os.path.join('../logs', date, timestamp, test_sequence + '-DPM_test_full', 'gt.txt'), test_mod_gt_dpm, delimiter=',')

#np.save(os.path.join('../logs', date, timestamp, test_sequence + '-DPM_test_removed', 'pred_removed.npy'), test_pred_removed_dpm)
#np.savetxt(os.path.join('../logs', date, timestamp, test_sequence + '-DPM_test_removed', 'gt.txt'), test_mod_gt_dpm, delimiter=',')

del train_labels_dpm, val_labels_dpm, test_labels_dpm, train_feat_dpm, val_feat_dpm, train_knn_graph_dpm, \
    val_knn_graph_dpm, test_knn_graph_dpm, meta_train_dpm, meta_val_dpm, meta_test_dpm, test_pred_dpm, \
    test_pred_removed_dpm, train_state_dict_dpm, train_args_dpm, val_args_dpm, test_args_dpm

########################################################################################################################

print('START TRAINING FRCNN DETECTOR.....')

# create splits and adjust labeling so that it goes from 0 to num_classes in split
train_labels_frcnn = adjust_labeling(meta_train_frcnn[label_column_train].to_numpy()[meta_train_frcnn[iou_column_train]])
val_labels_frcnn = adjust_labeling(meta_val_frcnn[label_column_valid].to_numpy()[meta_val_frcnn[iou_column_valid]])
test_labels_frcnn = adjust_labeling(meta_test_frcnn[label_column_test].to_numpy()[meta_test_frcnn[iou_column_test]])

train_feat_frcnn = train_feat_frcnn[meta_train_frcnn[iou_column_train]]
val_feat_frcnn = val_feat_frcnn[meta_val_frcnn[iou_column_valid]]
test_feat_frcnn = test_feat_frcnn[meta_test_frcnn[iou_column_test]]

# create knn graphs for each split (ensures labeling is correct)
# first checks whether the train/ test/ valid splits are too small for creating a KNN graph.
# if there are less instances the maximum size knn graph is created (i.e. len(split))
if len(train_labels_frcnn) > knn:
    train_knn_graph_frcnn = create_knn_graph_dataset(train_feat_frcnn, knn, knn_method)
else:
    train_knn_graph_frcnn = create_knn_graph_dataset(train_feat_frcnn, len(train_labels_frcnn)-1, knn_method)
if len(val_labels_frcnn) > knn:
    val_knn_graph_frcnn = create_knn_graph_dataset(val_feat_frcnn, knn, knn_method)
else:
    val_knn_graph_frcnn = create_knn_graph_dataset(val_feat_frcnn, len(val_labels_frcnn)-1, knn_method)
if len(test_labels_frcnn) > knn:
    test_knn_graph_frcnn = create_knn_graph_dataset(test_feat_frcnn, knn, knn_method)
else:
    test_knn_graph_frcnn = create_knn_graph_dataset(test_feat_frcnn, len(test_labels_frcnn)-1, knn_method)

# print shapes of files
print("TRAIN SHAPES FRCNN")
print(train_labels_frcnn.shape)
print(train_feat_frcnn.shape)
print(train_knn_graph_frcnn.shape)

print("VALID SHAPES FRCNN")
print(val_labels_frcnn.shape)
print(val_feat_frcnn.shape)
print(val_knn_graph_frcnn.shape)

print("TEST SHAPES FRCNN")
print(test_labels_frcnn.shape)
print(test_feat_frcnn.shape)
print(test_knn_graph_frcnn.shape)

# training
train_args_frcnn = create_train_args(input_channels=input_channels,
                                     seed=seed,
                                     workers=workers,
                                     print_freq=print_freq,
                                     gpu=gpu_name,
                                     lr=learning_rate_training,
                                     momentum=momentum_training,
                                     weight_decay=weight_decay_training,
                                     epochs=epochs_training,
                                     batch_size=batch_size_training,
                                     features=train_feat_frcnn,
                                     knn_graph=train_knn_graph_frcnn,
                                     labels=train_labels_frcnn,
                                     k_at_hop=k_at_hop_training,
                                     active_connection=active_connections_training,
                                     save_checkpoints=save_checkpoints,
                                     checkpoint_directory=frcnn_checkpoint
                                     )

train_state_dict_frcnn = train_main(train_args_frcnn)

# validation
val_args_frcnn = create_test_args(seed=seed,
                                  workers=workers,
                                  print_freq=print_freq,
                                  gpu=gpu_name,
                                  lr=learning_rate_testing,
                                  momentum=momentum_testing,
                                  weight_decay=momentum_testing,
                                  epochs=epochs_testing,
                                  batch_size=batch_size_testing,
                                  features=val_feat_frcnn,
                                  knn_graph=val_knn_graph_frcnn,
                                  labels=val_labels_frcnn,
                                  k_at_hop=k_at_hop_testing,
                                  active_connection=active_connections_testing,
                                  use_checkpoint=use_checkpoint,
                                  checkpoint_directory=frcnn_checkpoint,
                                  input_channels=input_channels
                                  )

val_pred_frcnn, val_pred_removed_frcnn = test_main(train_state_dict_frcnn, val_args_frcnn)
mkdir_if_missing(os.path.join('../logs', date, timestamp, val_sequence + '-FRCNN_val_full'))
#mkdir_if_missing(os.path.join('../logs', date, timestamp, val_sequence + '-FRCNN_val_removed'))

val_pred_frcnn = create_prediction_output_file(meta_val_frcnn, val_pred_frcnn, iou_column_valid)
#val_pred_removed_frcnn = val_pred_removed_frcnn
val_mod_gt_frcnn = create_modified_ground_truth_file(gt_val, meta_val_frcnn, iou_column_valid)

np.savetxt(os.path.join('../logs', date, timestamp, val_sequence + '-FRCNN_val_full', 'eval_input.txt'), val_pred_frcnn, delimiter=' ')
np.savetxt(os.path.join('../logs', date, timestamp, val_sequence + '-FRCNN_val_full', 'gt.txt'), val_mod_gt_frcnn, delimiter=',')

#np.save(os.path.join('../logs', date, timestamp, val_sequence + '-FRCNN_val_removed', 'pred_removed.npy'), val_pred_removed_frcnn)
#np.savetxt(os.path.join('../logs', date, timestamp, val_sequence + '-FRCNN_val_removed', 'gt.txt'), val_mod_gt_frcnn, delimiter=',')

# testing
test_args_frcnn = create_test_args(seed=seed,
                                   workers=workers,
                                   print_freq=print_freq,
                                   gpu=gpu_name,
                                   lr=learning_rate_testing,
                                   momentum=momentum_testing,
                                   weight_decay=momentum_testing,
                                   epochs=epochs_testing,
                                   batch_size=batch_size_testing,
                                   features=test_feat_frcnn,
                                   knn_graph=test_knn_graph_frcnn,
                                   labels=test_labels_frcnn,
                                   k_at_hop=k_at_hop_testing,
                                   active_connection=active_connections_testing,
                                   use_checkpoint=use_checkpoint,
                                   checkpoint_directory=frcnn_checkpoint,
                                   input_channels=input_channels
                                   )

test_pred_frcnn, test_pred_removed_frcnn = test_main(train_state_dict_frcnn, test_args_frcnn)
mkdir_if_missing(os.path.join('../logs', date, timestamp, test_sequence + '-FRCNN_test_full'))
#mkdir_if_missing(os.path.join('../logs', date, timestamp, test_sequence + '-FRCNN_test_removed'))

test_pred_frcnn = create_prediction_output_file(meta_test_frcnn, test_pred_frcnn, iou_column_test)
#test_pred_removed_frcnn = test_pred_removed_frcnn
test_mod_gt_frcnn = create_modified_ground_truth_file(gt_test, meta_test_frcnn, iou_column_test)

np.savetxt(os.path.join('../logs', date, timestamp, test_sequence + '-FRCNN_test_full', 'eval_input.txt'), test_pred_frcnn, delimiter=' ')
np.savetxt(os.path.join('../logs', date, timestamp, test_sequence + '-FRCNN_test_full', 'gt.txt'), test_mod_gt_frcnn, delimiter=',')

#np.save(os.path.join('../logs', date, timestamp, test_sequence + '-FRCNN_test_removed', 'pred_removed.npy'), test_pred_removed_frcnn)
#np.savetxt(os.path.join('../logs', date, timestamp, test_sequence + '-FRCNN_test_removed', 'gt.txt'), test_mod_gt_frcnn, delimiter=',')

del train_labels_frcnn, val_labels_frcnn, test_labels_frcnn, train_feat_frcnn, val_feat_frcnn, train_knn_graph_frcnn, \
    val_knn_graph_frcnn, test_knn_graph_frcnn, meta_train_frcnn, meta_val_frcnn, meta_test_frcnn, \
    test_pred_frcnn, test_pred_removed_frcnn, train_state_dict_frcnn, train_args_frcnn, val_args_frcnn, test_args_frcnn

########################################################################################################################

print('START TRAINING SDP DETECTOR.....')
# create splits and adjust labeling so that it goes from 0 to num_classes in split
train_labels_sdp = adjust_labeling(meta_train_sdp[label_column_train].to_numpy()[meta_train_sdp[iou_column_train]])
val_labels_sdp = adjust_labeling(meta_val_sdp[label_column_valid].to_numpy()[meta_val_sdp[iou_column_valid]])
test_labels_sdp = adjust_labeling(meta_test_sdp[label_column_test].to_numpy()[meta_test_sdp[iou_column_test]])

train_feat_sdp = train_feat_sdp[meta_train_sdp[iou_column_train]]
val_feat_sdp = val_feat_sdp[meta_val_sdp[iou_column_valid]]
test_feat_sdp = test_feat_sdp[meta_test_sdp[iou_column_test]]

# create knn graphs for each split (ensures labeling is correct)
# first checks whether the train/ test/ valid splits are too small for creating a KNN graph.
# if there are less instances the maximum size knn graph is created (i.e. len(split))
if len(train_labels_sdp) > knn:
    train_knn_graph_sdp = create_knn_graph_dataset(train_feat_sdp, knn, knn_method)
else:
    train_knn_graph_sdp = create_knn_graph_dataset(train_feat_sdp, len(train_labels_sdp)-1, knn_method)
if len(val_labels_sdp) > knn:
    val_knn_graph_sdp = create_knn_graph_dataset(val_feat_sdp, knn, knn_method)
else:
    val_knn_graph_sdp = create_knn_graph_dataset(val_feat_sdp, len(val_labels_sdp)-1, knn_method)
if len(test_labels_sdp) > knn:
    test_knn_graph_sdp = create_knn_graph_dataset(test_feat_sdp, knn, knn_method)
else:
    test_knn_graph_sdp = create_knn_graph_dataset(test_feat_sdp, len(test_labels_sdp)-1, knn_method)

# print shapes of files
print("TRAIN SHAPES SDP")
print(train_labels_sdp.shape)
print(train_feat_sdp.shape)
print(train_knn_graph_sdp.shape)

print("VALID SHAPES SDP")
print(val_labels_sdp.shape)
print(val_feat_sdp.shape)
print(val_knn_graph_sdp.shape)

print("TEST SHAPES SDP")
print(test_labels_sdp.shape)
print(test_feat_sdp.shape)
print(test_knn_graph_sdp.shape)

# training
train_args_sdp = create_train_args(input_channels=input_channels,
                                   seed=seed,
                                   workers=workers,
                                   print_freq=print_freq,
                                   gpu=gpu_name,
                                   lr=learning_rate_training,
                                   momentum=momentum_training,
                                   weight_decay=weight_decay_training,
                                   epochs=epochs_training,
                                   batch_size=batch_size_training,
                                   features=train_feat_sdp,
                                   knn_graph=train_knn_graph_sdp,
                                   labels=train_labels_sdp,
                                   k_at_hop=k_at_hop_training,
                                   active_connection=active_connections_training,
                                   save_checkpoints=save_checkpoints,
                                   checkpoint_directory=sdp_checkpoint
                                   )

train_state_dict_sdp = train_main(train_args_sdp)

# validation
val_args_sdp = create_test_args(seed=seed,
                                workers=workers,
                                print_freq=print_freq,
                                gpu=gpu_name,
                                lr=learning_rate_testing,
                                momentum=momentum_testing,
                                weight_decay=momentum_testing,
                                epochs=epochs_testing,
                                batch_size=batch_size_testing,
                                features=val_feat_sdp,
                                knn_graph=val_knn_graph_sdp,
                                labels=val_labels_sdp,
                                k_at_hop=k_at_hop_testing,
                                active_connection=active_connections_testing,
                                use_checkpoint=use_checkpoint,
                                checkpoint_directory=sdp_checkpoint,
                                input_channels=input_channels
                                )

val_pred_sdp, val_pred_removed_sdp = test_main(train_state_dict_sdp, val_args_sdp)
mkdir_if_missing(os.path.join('../logs', date, timestamp, val_sequence + '-SDP_val_full'))
#mkdir_if_missing(os.path.join('../logs', date, timestamp, val_sequence + '-SDP_val_removed'))

val_pred_sdp = create_prediction_output_file(meta_val_sdp, val_pred_sdp, iou_column_valid)
val_pred_removed_sdp = val_pred_removed_sdp
val_mod_gt_sdp = create_modified_ground_truth_file(gt_val, meta_val_sdp, iou_column_valid)

np.savetxt(os.path.join('../logs', date, timestamp, val_sequence + '-SDP_val_full', 'eval_input.txt'), val_pred_sdp, delimiter=' ')
np.savetxt(os.path.join('../logs', date, timestamp, val_sequence + '-SDP_val_full', 'gt.txt'), val_mod_gt_sdp, delimiter=',')

#np.save(os.path.join('../logs', date, timestamp, val_sequence + '-SDP_val_removed', 'pred_removed.npy'), val_pred_removed_sdp)
#np.savetxt(os.path.join('../logs', date, timestamp, val_sequence + '-SDP_val_removed', 'gt.txt'), val_mod_gt_sdp, delimiter=',')

# testing
test_args_sdp = create_test_args(seed=seed,
                                 workers=workers,
                                 print_freq=print_freq,
                                 gpu=gpu_name,
                                 lr=learning_rate_testing,
                                 momentum=momentum_testing,
                                 weight_decay=momentum_testing,
                                 epochs=epochs_testing,
                                 batch_size=batch_size_testing,
                                 features=test_feat_sdp,
                                 knn_graph=test_knn_graph_sdp,
                                 labels=test_labels_sdp,
                                 k_at_hop=k_at_hop_testing,
                                 active_connection=active_connections_testing,
                                 use_checkpoint=use_checkpoint,
                                 checkpoint_directory=sdp_checkpoint,
                                 input_channels=input_channels
                                 )

test_pred_sdp, test_pred_removed_sdp = test_main(train_state_dict_sdp, test_args_sdp)
mkdir_if_missing(os.path.join('../logs', date, timestamp, test_sequence + '-SDP_test_full'))
#mkdir_if_missing(os.path.join('../logs', date, timestamp, test_sequence + '-SDP_test_removed'))

test_pred_sdp = create_prediction_output_file(meta_test_sdp, test_pred_sdp, iou_column_test)
test_pred_removed_sdp = test_pred_removed_sdp
test_mod_gt_sdp = create_modified_ground_truth_file(gt_test, meta_test_sdp, iou_column_test)

np.savetxt(os.path.join('../logs', date, timestamp, test_sequence + '-SDP_test_full', 'eval_input.txt'), test_pred_sdp, delimiter=' ')
np.savetxt(os.path.join('../logs', date, timestamp, test_sequence + '-SDP_test_full', 'gt.txt'), test_mod_gt_sdp, delimiter=',')

#np.save(os.path.join('../logs', date, timestamp, test_sequence + '-SDP_test_removed', 'pred_removed.npy'), test_pred_removed_sdp)
#np.savetxt(os.path.join('../logs', date, timestamp, test_sequence + '-SDP_test_removed', 'gt.txt'), test_mod_gt_sdp, delimiter=',')
