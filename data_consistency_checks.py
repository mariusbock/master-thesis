import os
import pandas as pd
import numpy as np

from dataset_creation import create_modified_detection_file
from utils import iou


def check_mod_det_file(det_file, mod_det_file):
    """
    Function that checks whether the modified detection file is consistent with the original detection file.
    Goes through each row and compared by equality (except added index column)
    """
    print('Checking Modified Detection File...')
    for i in range(len(mod_det_file)):
        # If rows are not the same then user is notified
        if (mod_det_file[i, 1:] != det_file[i, :]).all():
            print('Rows' + str(i) + 'not the same')


def get_all_metadata_files(filepath):
    """
    Function that returns the relative paths to all metadata files within a detector directory.
    Return type is a list of strings
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
    """
    print('Checking Metadata File Length...')
    if len(det_file) != len(metadata_file):
        print("Metadata and detection file are not the same length!")


def check_detections(det_file, metadata_file):
    """
    Function that checks whether the idx, frame and detection coordinates in metadata file match the ones from the
    original detection file. Iterates over each row and checks with corresponding row in detection
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


def check_labeling(metadata_file, gt_file):
    """
    Function that checks whether the labeling of the metadata file with the ground truth file. Checks for each row in
    the metadata file whether:
    1. Ground truth bounding box in metadata row is contained in ground truth file
    2. If the corresponding bbox in the ground truth file is not excluded
    3. If the label assigned to the bbox is consistent with the one from the ground truth file
    4. If the IoU calculated between the detection and ground truth file is the same
    """
    print('Checking Labeling of Metadata File...')
    # need to round metadata file (otherwise format conflict between numpy and pandas)
    rounded_metadata = metadata_file.round(3)
    # number of detection i.e. length of metadata file
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
        # check for unassigned bboxes i.e. coordinates all -1, iou equal to 0 and label equal to -1
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


def check_splits(metadata_file):
    """
    Function that checks whether the splitting is consistent. That is it is checked whether:
    1. Splits are consistent with IoU and background labeling is consistent with chosen method
    2. A detection is never part of more than one split (class/ background/ excluded)
    3. A detection is never part of more than one split (train/ valid/ test)
    4. For each filtered iou column it is checked whether the record actually satifies the condition of being either
    larger or smaller than the iou thresholds defined for the filtering
    """
    print('Checking Splits...')
    num_lines_det = len(metadata_file)
    label_col = [col for col in metadata_file.columns if 'labels_' in col]
    # column names of all filtered columns (start with fil)
    fil_cols = [col for col in metadata_file.columns if 'fil' in col]
    iou_upper = label_col.split("_")[-3]
    iou_lower = label_col.split("_")[-2]
    bg_handling = label_col.split("_")[-3]
    for i in range(len(metadata_file)):
        # check for each type of split if it meets the iou criterion
        if metadata_file.iloc[i]['is_class']:
            if metadata_file.iloc[i]['iou'] <= iou_upper:
                print("IoU does not satisfy is_class threshold in column row " + str(i))
        # for background handling additionally check whether the labeling is consistent with the chosen method
        if metadata_file.iloc[i]['is_background']:
            if bg_handling == "zero":
                if metadata_file.iloc[i][label_col] != 0:
                    print("Labeling is inconsistent with background handling in column row " + str(i))
            # TODO: come up with check-up for Singletons
            if metadata_file.iloc[i]['iou'] >= iou_lower:
                print("IoU does not satisfy is_background threshold in column row " + str(i))
        if metadata_file.iloc[i]['is_excluded']:
            if metadata_file.iloc[i]['iou'] > iou_upper or metadata_file.iloc[i]['iou'] < iou_lower:
                print("IoU does not satisfy is_excluded threshold in column row " + str(i))
        # if more than one of the is_class, is_background and is_excluded column is true, then the user is notified
        if int(metadata_file.iloc[i]['is_class']) + int(metadata_file.iloc[i]['is_background']) + int(
                metadata_file.iloc[i]['is_excluded']) > 1:
            print("Row " + str(i) + "is included in more than one split (class/ background/ excluded)!")
        # if none of the is_class, is_background and is_excluded column is true, then the user is notified
        if int(metadata_file.iloc[i]['is_class']) + int(metadata_file.iloc[i]['is_background']) + int(
                metadata_file.iloc[i]['is_excluded']) == 1:
            print("Row " + str(i) + "is included in no split (class/ background/ excluded)!")
        # if more than one of the is_train, is_valid and is_test column is true, then the user is notified
        if int(metadata_file.iloc[i]['is_train']) + int(metadata_file.iloc[i]['is_valid']) + int(
                metadata_file.iloc[i]['is_test']) > 1:
            print("Row " + str(i) + "is included in more than one split (train/ valid/ test)!")
        # if none of the is_train, is_valid and is_test column is true, then the user is notified
        if int(metadata_file.iloc[i]['is_train']) + int(metadata_file.iloc[i]['is_valid']) + int(
                metadata_file.iloc[i]['is_test']) == 1:
            print("Row " + str(i) + "is included in no split (train/ valid/ test)!")
        # for the filter columns the thresholds are obtained via the column name (fil_[lower bound]_[upper bound])
        for fil_col in fil_cols:
            fil_iou_upper = float(fil_col.split("_")[-2])
            fil_iou_lower = float(fil_col.split("_")[-1])
            # if iou does not fit requirements of the filter column user is notified
            if metadata_file.iloc[i][fil_col] and fil_iou_lower <= metadata_file.iloc[i]['iou'] <= fil_iou_upper:
                print("IoU does not satisfy thresholds in filtered column " + fil_col + " row " + str(i))
        if i % 2500 == 0:
            print("Detection processed: " + str(i) + "/" + str(num_lines_det))


def check_metadata_files(det_file, gt_file, metadata_paths):
    """
    Function that put together all individual functions and applies each of them on each metadata file. First creates
    modified detection file and checks its consistency; then loops over each metadata file.
    """
    # modify detection file (add index as first column)
    mod_det_file_mot = create_modified_detection_file(det_file)
    check_mod_det_file(det_file, mod_det_file_mot)

    # per metadata file apply all funtions that check consistency
    for metadata_path in metadata_paths:
        print("CHECKING METADATA FILE: " + metadata_path)
        metadata_file = pd.read_csv(metadata_path)
        check_metadata_file_length(mod_det_file_mot, metadata_file)
        check_detections(mod_det_file_mot, metadata_file)
        check_labeling(metadata_file, gt_file)
        check_splits(metadata_file)
        print("DONE CHECKING METADATA FILE: " + metadata_path)
    print("DONE CHECKING FILES")


if __name__ == '__main__':
    # Detection parameters
    sequence = "MOT/MOT17/MOT17-02"
    detector = "dpm"

    detection_file = np.loadtxt(os.path.join("data", sequence, detector, "det.txt"), delimiter=",")
    metadata_files = get_all_metadata_files(os.path.join("data", sequence, detector))
    ground_truth_file = np.loadtxt(os.path.join("data", sequence, "gt.txt"), delimiter=",")

    check_metadata_files(detection_file, ground_truth_file, metadata_files)
