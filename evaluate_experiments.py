import os
import sys
import json
import moviepy.video.io.ImageSequenceClip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation.evaluateTracking_modified import evaluateTracking
from gcn_clustering.utils.logging import Logger
from gcn_clustering.utils.osutils import mkdir_if_missing
from misc import split

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def evaluate_tracking(eval_runs, removed_instances, is_split, log_files_path, data_path, sequence_maps, detectors):
    """
    Function that evaluates tracking results using the official MOT Metrics (using py-motmetrics). It evaluates
    per Run and if removed_instances is True also evaluates the prediction files where singleton clusters are removed.

    Args:
        evaluation_runs -- number of runs that are to be evaluated
        removed_instances -- boolean whether to calculate evaluation metrics for singleton removal scenario
        is_split -- boolean indicating that modified version of GT is to be used (because of split)
        log_files_path -- path to where log files of run(s) to be evaluated are stored
        data_path -- path to where data is located that was used during prediction
        sequence_maps -- list of sequence maps to use for evaluation (see evaluation/seqmaps for examples)
        detectors -- list of detector names to evaluate (e.g. MOT17-02-DPM)

    Returns:
        Prints evaluation results; stores to eval_data and returns eval_data dataframe
    """
    # load eval_data dataframe from log_directory
    eval_data = pd.read_csv(os.path.join(log_files_path, 'eval_data.csv'), index_col=['run'])
    # saves logs to a file (standard output redirected)
    sys.stdout = Logger(os.path.join(log_files_path, 'eval_log.txt'))
    # initialise columns in eval_data dataframe with all zeros
    for detector in detectors:
        eval_data[detector + '_mota'] = 0.0
        eval_data[detector + '_IDS'] = 0.0
        if removed_instances:
            eval_data[detector + '_mota_rm'] = 0.0
            eval_data[detector + '_IDS_rm'] = 0.0
    # iterate over all runs
    for i in range(eval_runs):
        print('RUN: ' + str(i + 1))
        # go through all seqmaps (which contain detectors to evaluate)
        for seqmap in sequence_maps:
            print('EVALUATING SEQMAP: ' + seqmap)
            print('EVALUATION RESULTS')
            eval_results = evaluateTracking(os.path.join(log_files_path, 'run_' + str(i + 1), 'full'),
                                            data_path, seqmap, is_split)
            # if true, setting where singletons were removed is evaluated
            if removed_instances:
                print('EVALUATION RESULTS (REMOVED)')
                eval_results_rm = evaluateTracking(os.path.join(log_files_path, 'run_' + str(i + 1), 'removed'),
                                                   data_path, seqmap, is_split)
                # save results to eval_data dataframe
                for detector in eval_results_rm.index.tolist():
                    eval_data.at['run_' + str(i + 1), detector + '_mota_rm'] = eval_results_rm.loc[detector, 'mota']
                    eval_data.at['run_' + str(i + 1), detector + '_IDS_rm'] = eval_results_rm.loc[
                        detector, 'num_switches']
            # save results to eval_data dataframe
            for detector in eval_results.index.tolist():
                eval_data.at['run_' + str(i + 1), detector + '_mota'] = eval_results.loc[detector, 'mota']
                eval_data.at['run_' + str(i + 1), detector + '_IDS'] = eval_results.loc[detector, 'num_switches']
    # save eval_data again to csv file
    eval_data.to_csv(os.path.join(log_files_path, 'eval_data.csv'), index=['run'])
    return eval_data


def create_plots(log_directory_path, eval_file, train_epochs, train_detector_names, val_detector_names, detector_types,
                 experiment_type):
    """
    Function that creates plots used to analyse overfitting. Saves the plots into a plot folder wihtin the log
    directory. Creates:
        - Average and batch-wise training and validation losses across epochs.
        - Average validation loss of each sequence (x-axis) compared to the MOTA score of said sequence (y-axis)
        - Average loss on the validation set of the training data (x-axis) compared against the MOTA score of each
          validation sequence (y-axis).
        - Average training loss (x-axis) compared against the MOTA score of each validation sequence (y-axis).
        - Average loss on the validation set of the training data sequence divided by the average training loss (x-axis)
          against the MOTA score of each validation sequence (y-axis).

    Args:
        log_directory_path -- directory where log files of experiment are located
        eval_file -- eval_data file that was created during the experiments
        train_epochs -- no. epochs during training
        train_detector_names -- name of detectors used for training
        val_detector_names -- name of detectors used for validation
        detector_types -- list of detector types e.g. ['DPM', 'FRCNN', 'SDP']
        experiment_type -- type of experiment ('single' or 'combined' setting)
    """
    # create directories where plots are saved to after creation
    mkdir_if_missing(os.path.join(log_directory_path, 'plots', 'loss_curves'))
    mkdir_if_missing(os.path.join(log_directory_path, 'plots', 'MOTA'))
    mkdir_if_missing(os.path.join(log_directory_path, 'plots', 'IDS'))

    # filter eval_data columns for all training loss columns
    training_loss_cols = [col for col in eval_file.columns if 'train' in col]

    # create training loss curves
    for epoch in range(train_epochs):
        for training_loss_col in training_loss_cols:
            for index, run in eval_file.iterrows():
                # convert to series for plotting
                training_losses = json.loads(run[training_loss_col])
                plt.plot(training_losses[epoch])
                # indicate via y_label of plot if average or batch_wise loss
                if 'avg' in training_loss_col:
                    plt.ylabel('average training loss')
                else:
                    plt.ylabel('training loss')
                plt.xlabel('batch number')
                plt.title(training_loss_col + ': epoch ' + str(epoch + 1))
            plt.legend(eval_file.index.tolist())
            plt.ylim((min(training_losses[epoch]) - (min(training_losses[epoch]) / 2),
                      max(training_losses[epoch]) + (max(training_losses[epoch]) / 2)))
            # save plot
            plt.savefig(os.path.join(log_directory_path, 'plots', 'loss_curves',
                                     training_loss_col + '_epoch_' + str(epoch + 1) + '.png'))
            plt.close()

    # create validation loss curves
    for detector in train_detector_names + val_detector_names:
        for index, run in eval_file.iterrows():
            val_loss = json.loads(run[detector + '_loss'])
            plt.plot(val_loss)
            plt.ylabel('validation loss')
            plt.xlabel('batch number')
            plt.ylim((min(val_loss) - (min(val_loss) / 2), max(val_loss) + (max(val_loss) / 2)))
            plt.title('validation loss ' + detector)
        plt.legend(eval_file.index.tolist())
        plt.savefig(os.path.join(log_directory_path, 'plots', 'loss_curves', 'val_loss_' + detector + '.png'))
        plt.close()

    # create average validation loss curves
    for detector in train_detector_names + val_detector_names:
        for index, run in eval_file.iterrows():
            val_loss = json.loads(run[detector + '_avg_loss'])
            plt.plot(val_loss)
            plt.ylabel('average validation loss')
            plt.xlabel('batch number')
            plt.ylim((min(val_loss) - (min(val_loss) / 2), max(val_loss) + (max(val_loss) / 2)))
            plt.title('average validation loss ' + detector)
        plt.legend(eval_file.index.tolist())
        plt.savefig(os.path.join(log_directory_path, 'plots', 'loss_curves', 'avg_val_loss_' + detector + '.png'))
        plt.close()

    # create loss plots (all epochs + val loss after each epoch)
    for detector in detector_types:
        min_loss = 99999999999.9
        max_loss = 0.0
        plt.figure(figsize=(10, 10))
        for index, run in eval_file.iterrows():
            loss_column = [k for k in training_loss_cols if 'avg' not in k and detector in k and 'val' not in k]
            val_column = [k for k in training_loss_cols if 'avg' not in k and detector in k and 'val' in k]

            losses = []
            for epoch_loss in json.loads(run[loss_column][0]):
                split_epoch = split(epoch_loss, 2)
                losses.append(np.mean(split_epoch[0]))
                losses.append(np.mean(epoch_loss))

            val_losses = []
            for epoch_loss in json.loads(run[val_column][0]):
                split_epoch = split(epoch_loss, 2)
                val_losses.append(np.mean(split_epoch[0]))
                val_losses.append(np.mean(epoch_loss))

            plt.plot(np.arange(0.5, epochs + .5, 0.5), losses)
            plt.plot(np.arange(0.5, epochs + .5, 0.5), val_losses)

            plt.xlabel('batch number')
            plt.ylabel('loss')
            plt.title('Train Loss + Val Loss ' + detector)
            if min(losses) < min_loss:
                min_loss = min(losses)
            elif min(val_losses) < min_loss and min(val_losses) < min(losses):
                min_loss = min(val_losses)
            if max(losses) > max_loss:
                max_loss = max(losses)
            elif max(val_losses) > max_loss and max(val_losses) > max(losses):
                max_loss = max(val_losses)
        legend = []
        for run in eval_file.index.tolist():
            legend.append(run + '_train')
            legend.append(run + '_val')
        plt.legend(legend)
        plt.savefig(os.path.join(log_directory_path, 'plots', 'loss_curves', 'train_val_loss_' + detector + '.png'))
        plt.ylim((min_loss - (min_loss / 2)), max_loss + (max_loss / 2))

        plt.close()

    """
    OVERFITTING ANALYSIS
    """
    # iterate over all validation detectors
    for val_detector in val_detector_names:
        for train_detector in train_detector_names:
            train_val_loss_avg = []
            train_loss_avg = []
            val_loss_avg = []
            train_div_train_val_loss_avg = []
            for detector_type in detector_types:
                if detector_type in val_detector:
                    # if combined setting use loss of final GCN
                    if experiment_type == 'single':
                        loss_column = [col for col in eval_file.columns if
                                       'train' in col and detector_type in col and not 'avg' in col]
                    else:
                        loss_column = [col for col in eval_file.columns if
                                       'train' in col and detector_type in col and 'final' in col and not 'avg' in col]
                    for index, run in eval_file.iterrows():
                        train_loss = json.loads(run[loss_column][0])[-1]
                        train_val_loss = json.loads(run[train_detector + '_loss'])
                        val_loss = json.loads(run[val_detector + '_loss'])
                        # create final value by taking average of last two train/ validation loss values
                        train_loss_avg.append((train_loss[-2] + train_loss[-1]) / 2)
                        train_val_loss_avg.append((train_val_loss[-2] + train_val_loss[-1]) / 2)
                        val_loss_avg.append((val_loss[-2] + val_loss[-1]) / 2)
                        # create final value by taking average of last two train and validation loss values
                        # and dividing averages
                        train_div_train_val_loss_avg.append(
                            ((train_loss[-2] + train_loss[-1]) / 2) / (train_val_loss[-2] + train_val_loss[-1]) / 2)
                    mota = eval_file[val_detector + '_mota'].tolist()
                    ids = eval_file[val_detector + '_IDS'].tolist()

                    ### MOTA ###

                    # combine three plots to one plot with the subplots being next to each other
                    fig, axs = plt.subplots(1, 3)
                    fig.set_size_inches(30, 10)
                    plt.title(val_detector)
                    plt.autoscale()

                    # VAL LOSS TRAINING VS MOTA VAL SEQUENCES
                    # create list of pairs which resemble points for each run within the scatter plot
                    combined = [list(a) for a in zip(train_val_loss_avg, mota)]
                    for pair in combined:
                        axs[0].scatter(pair[0], pair[1])
                    axs[0].set_title('Training Validation Loss vs. MOTA Validation Sequences')
                    axs[0].set_xlabel('Val Loss ' + train_detector)
                    axs[0].set_ylabel('MOTA ' + val_detector)
                    axs[0].legend(eval_file.index.tolist(), prop={'size': 15})
                    axs[0].set_xlim((min(val_loss_avg) - (min(val_loss_avg) / 4), max(val_loss_avg) + (max(val_loss_avg) / 4)))
                    axs[0].set_ylim((min(mota) - (min(mota) / 32), max(mota) + (max(mota) / 32)))

                    # TRAIN LOSS VS MOTA VAL SEQUENCES
                    # create list of pairs which resemble points for each run within the scatter plot
                    combined = [list(a) for a in zip(train_loss_avg, mota)]
                    for pair in combined:
                        axs[1].scatter(pair[0], pair[1])
                    axs[1].set_title('Average Train Loss vs. MOTA Validation Sequences')
                    axs[1].set_xlabel('Train Loss')
                    axs[1].set_ylabel('MOTA ' + val_detector)
                    axs[1].legend(eval_file.index.tolist(), prop={'size': 15})
                    axs[1].set_xlim(
                        (min(train_loss_avg) - (min(train_loss_avg) / 4), max(train_loss_avg) + (max(train_loss_avg) / 4)))
                    axs[1].set_ylim((min(mota) - (min(mota) / 32), max(mota) + (max(mota) / 32)))

                    # TRAIN LOSS / VAL LOSS VS MOTA VAL SEQUENCES
                    # create list of pairs which resemble points for each run within the scatter plot
                    combined = [list(a) for a in zip(train_val_loss_avg, mota)]
                    for pair in combined:
                        axs[2].scatter(pair[0], pair[1])
                    axs[2].legend(eval_file.index.tolist(), prop={'size': 15})
                    axs[2].set_title('train loss / val loss vs. MOTA')
                    axs[2].set_xlabel('train loss / val loss')
                    axs[2].set_ylabel('MOTA')
                    axs[2].set_xlim((min(train_val_loss_avg) - (min(train_val_loss_avg) / 4),
                                     max(train_val_loss_avg) + (max(train_val_loss_avg) / 4)))
                    axs[2].set_ylim((min(mota) - (min(mota) / 32), max(mota) + (max(mota) / 32)))

                    plt.savefig(os.path.join(log_directory_path, 'plots', 'MOTA', val_detector + '.png'))
                    plt.close()

                    ### IDS ###

                    # combine three plots to one plot with the subplots being next to each other
                    fig, axs = plt.subplots(1, 3)
                    fig.set_size_inches(30, 10)
                    plt.title(val_detector)
                    plt.autoscale()

                    # VAL LOSS TRAINING VS IDS VAL SEQUENCES
                    # create list of pairs which resemble points for each run within the scatter plot
                    combined = [list(a) for a in zip(train_val_loss_avg, ids)]
                    for pair in combined:
                        axs[0].scatter(pair[0], pair[1])
                    axs[0].set_title('Training Validation Loss vs. IDS Validation Sequences')
                    axs[0].set_xlabel('Val Loss ' + train_detector)
                    axs[0].set_ylabel('IDS ' + val_detector)
                    axs[0].legend(eval_file.index.tolist(), prop={'size': 15})
                    axs[0].set_xlim((min(train_val_loss_avg) - (min(train_val_loss_avg) / 4),
                                     max(train_val_loss_avg) + (max(train_val_loss_avg) / 4)))
                    axs[0].set_ylim((min(ids) - (min(ids) / 32), max(ids) + (max(ids) / 32)))

                    # TRAIN LOSS VS IDS VAL SEQUENCES
                    # create list of pairs which resemble points for each run within the scatter plot
                    combined = [list(a) for a in zip(train_loss_avg, ids)]
                    for pair in combined:
                        axs[1].scatter(pair[0], pair[1])
                    axs[1].set_title('Average Train Loss vs. IDS Validation Sequences')
                    axs[1].set_xlabel('Train Loss')
                    axs[1].set_ylabel('IDS ' + val_detector)
                    axs[1].legend(eval_file.index.tolist(), prop={'size': 15})
                    axs[1].set_xlim((min(train_loss_avg) - (min(train_loss_avg) / 4),
                                     max(train_loss_avg) + (max(train_loss_avg) / 4)))
                    axs[1].set_ylim((min(ids) - (min(ids) / 32), max(ids) + (max(ids) / 32)))

                    # TRAIN LOSS / VAL LOSS VS IDS VAL SEQUENCES
                    # create list of pairs which resemble points for each run within the scatter plot
                    combined = [list(a) for a in zip(train_div_train_val_loss_avg, ids)]
                    for pair in combined:
                        axs[2].scatter(pair[0], pair[1])
                    axs[2].legend(eval_file.index.tolist(), prop={'size': 15})
                    axs[2].set_title('train loss / val loss vs. IDS')
                    axs[2].set_xlabel('train loss / val loss')
                    axs[2].set_ylabel('IDS')
                    axs[2].set_xlim((min(train_div_train_val_loss_avg) - (min(train_div_train_val_loss_avg) / 4),
                                     max(train_div_train_val_loss_avg) + (max(train_div_train_val_loss_avg) / 4)))
                    axs[2].set_ylim((min(ids) - (min(ids) / 32), max(ids) + (max(ids) / 32)))

                    plt.savefig(os.path.join(log_directory_path, 'plots', 'IDS', val_detector + '.png'))
                    plt.close()

    # Create training validation loss vs MOTA
    for detector in train_detector_names + val_detector_names:
        val_loss_avg = []
        for index, run in eval_file.iterrows():
            val_loss = json.loads(run[detector + '_loss'])
            # create final value by taking average of last two train loss values
            val_loss_avg.append((val_loss[-2] + val_loss[-1]) / 2)
        mota = eval_file[detector + '_mota'].tolist()
        ids = eval_file[detector + '_IDS'].tolist()

        ### MOTA ###

        # create list of pairs which resemble points for each run within the scatter plot
        combined = [list(a) for a in zip(val_loss_avg, mota)]
        for pair in combined:
            plt.scatter(pair[0], pair[1])
        plt.legend(eval_file.index.tolist())
        plt.title('Validation loss ' + detector + ' vs. MOTA ' + detector)
        plt.xlabel('val loss ' + detector)
        plt.ylabel('MOTA ' + detector)
        plt.xlim((min(val_loss_avg) - (min(val_loss_avg) / 4),
                  max(val_loss_avg) + (max(val_loss_avg) / 4)))
        plt.ylim((min(mota) - (min(mota) / 32), max(mota) + (max(mota) / 32)))

        plt.savefig(os.path.join(log_directory_path, 'plots', 'MOTA', detector + '_2.png'))
        plt.close()

        ### IDS ###

        # create list of pairs which resemble points for each run within the scatter plot
        combined = [list(a) for a in zip(val_loss_avg, ids)]
        for pair in combined:
            plt.scatter(pair[0], pair[1])
        plt.legend(eval_file.index.tolist())
        plt.title('Validation loss ' + detector + ' vs. MOTA ' + detector)
        plt.xlabel('val loss ' + detector)
        plt.ylabel('MOTA ' + detector)
        plt.xlim((min(val_loss_avg) - (min(val_loss_avg) / 4),
                  max(val_loss_avg) + (max(val_loss_avg) / 4)))
        plt.ylim((min(ids) - (min(ids) / 32), max(ids) + (max(ids) / 32)))

        plt.savefig(os.path.join(log_directory_path, 'plots', 'IDS', detector + '_2.png'))
        plt.close()


def create_videos(data_directory, detectors, fps):
    """
    Function that creates the sequence videos from the images within the cluster_vis folders within the logs

    Params:
        data_directory -- directory where detector folders are located in that contain the cluster_vis folder
        detectors -- list of detectors to process (i.e. folders contained in the data_directory
        fps --  list of fps of the detectors (needs to be same size as detectors variable)
    """
    for detector, fps in zip(detectors, fps):
        # save videos in 'videos' folder within the data_directory
        mkdir_if_missing(os.path.join(data_directory, 'videos'))
        # load images contained within cluster_vis folder and sort by name
        images = [data_directory + '/' + detector + '/cluster_vis/' + img for img in
                  os.listdir(os.path.join(data_directory, detector, 'cluster_vis')) if img.endswith(".jpg")]
        images.sort()
        # create video from said images and write to folder
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
        clip.write_videofile(os.path.join(data_directory, 'videos', detector + '.mp4'))


if __name__ == '__main__':
    log_pth = 'logs/20201113/142220'
    data_pth = 'data/MOT/MOT17/train'
    seqmap_path = 'evaluation/seqmaps'
    removed = True
    is_split_experiment = False
    train_type = 'single'
    runs = 1
    epochs = 4
    run_to_visualize = 'run_1/full'
    seqmap = 'MOT17-train.txt'  # Seqmap which contains train sequence
    seqmap_no_train = 'MOT17-no-train.txt'  # Seqmap which does not contain train sequence

    # detector names
    train_det = ['MOT17-04-DPM', 'MOT17-04-FRCNN', 'MOT17-04-SDP']

    val_det = ['MOT17-02-DPM', 'MOT17-02-FRCNN', 'MOT17-02-SDP',
               'MOT17-04-DPM', 'MOT17-04-FRCNN', 'MOT17-04-SDP',
               'MOT17-05-DPM', 'MOT17-05-FRCNN', 'MOT17-05-SDP',
               'MOT17-09-DPM', 'MOT17-09-FRCNN', 'MOT17-09-SDP',
               'MOT17-10-DPM', 'MOT17-10-FRCNN', 'MOT17-10-SDP',
               'MOT17-11-DPM', 'MOT17-11-FRCNN', 'MOT17-11-SDP',
               'MOT17-13-DPM', 'MOT17-13-FRCNN', 'MOT17-13-SDP']

    test_det = ['MOT17-01-DPM', 'MOT17-01-FRCNN', 'MOT17-01-SDP',
                'MOT17-03-DPM', 'MOT17-03-FRCNN', 'MOT17-03-SDP',
                'MOT17-06-DPM', 'MOT17-06-FRCNN', 'MOT17-06-SDP',
                'MOT17-07-DPM', 'MOT17-07-FRCNN', 'MOT17-07-SDP',
                'MOT17-08-DPM', 'MOT17-08-FRCNN', 'MOT17-08-SDP',
                'MOT17-12-DPM', 'MOT17-12-FRCNN', 'MOT17-12-SDP',
                'MOT17-14-DPM', 'MOT17-14-FRCNN', 'MOT17-14-SDP']

    # array containing frame rate of all sequences that one wants to create videos for
    det_fps = [30, 30, 30, 30, 30, 30, 14, 14, 14, 30, 30, 30, 30, 30, 30, 30, 30, 30, 25, 25, 25]
    # array containig detector types used during experiment
    det_types = ['DPM', 'FRCNN', 'SDP']

    eval_data = evaluate_tracking(eval_runs=runs,
                                  removed_instances=removed,
                                  is_split=is_split_experiment,
                                  log_files_path=log_pth,
                                  data_path=data_pth,
                                  sequence_maps=[os.path.join(seqmap_path, seqmap_no_train),
                                                 os.path.join(seqmap_path, seqmap)],
                                  detectors=train_det + val_det
                                  )

    create_plots(log_directory_path=log_pth,
                 eval_file=eval_data,
                 train_detector_names=train_det,
                 val_detector_names=val_det,
                 detector_types=det_types,
                 train_epochs=epochs,
                 experiment_type=train_type
                 )

    # Uncomment if want to create videos for train and test sequences (need images in cluster_vis folder)
    # NOTE: takes up a lot of disk space!
    # create_videos(os.path.join(log_pth, run_to_visualize), val_det, det_fps)
    # create_videos(os.path.join(log_path, run_path), test_det, det_fps)
