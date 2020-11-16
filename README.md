# Multiple Object Tracking by Link Prediction using Graph Convolution Networks

This repository contains the code for the master thesis "Multiple Object Tracking by Link Prediction using Graph Convolution Networks" of Marius Bock. The thesis is submitted to Professor Margret Keuper at the University of Mannheim.

## Introduction

## Requirements
- numpy 1.18.1
- matplotlib 3.1.2
- torchvision 0.5.0
- pandas 1.0.3
- pillow 7.1.2
- moviepy 1.0.1
- torch 1.4.0
- motmetrics 1.2.0
- scipy 1.4.1
- scikit-learn 0.23.2

## Dataset Creation
In order to create dataset weights for different CNNs need to be downloaded and put into 'networks/pretrained_models':
- From https://github.com/WangWenhao0716/Adapted-Center-and-Scale-Prediction download both 'ACSP(Smooth L1).pth.tea' and 'ResNet101v2+SN(8,32).pth'
- From https://github.com/layumi/Person_reID_baseline_pytorch download 'ft_net.pth'

Main script of dataset creation is dataset_creation.py. See hyperparameters provided in file what settings are possible. Note: this creation process was only tested for MOT17 and MOT20 data. Please place the files downloaded from the MOT website as is into a folder (e.g. 'data') within the main directory of this repository.

## Experiments
Currently, only the non-ensemble setup is supported. One can read up on the proposed ensemble setup, but would need to adjust workflow to fit new methods.

For single experiments, see the experiments_single.py file. See hyperparameters provided in file what settings are possible. The setting provided in the experiments file is the one producing the best result reported in the best setting section of the master thesis.

To use the modified version of the detection dataset as described in the thesis please apply the preprocessing as described in the Tracktor tracker on top of the detections (https://github.com/phil-bergmann/tracking_wo_bnw).

## Evaluation
The main functions for evaluation can be found in the evaluate_experiments.py file. The postprocessing script (evaluation/postprocessing_script.m) is only available using MatLab and needs to be applied seperately. The primal feasible search heuristic code (see graph folder) needs to be compiled in a 'build' directory using cmake and make. To apply the heurisitc for a sequence, run:

    ./solve-regular -i [absolute path to heurisitic_input_file]/heuristic_input.txt 
                    -o  [absolute path to same directory as before]/heuristic_output.txt 
                    -b [threshold to apply]
                    
Currently, the code needs to be run for each sequence individually. Once this is done, 'heuristic' can be set to 'true' in MatLab script and proposed clustering by the heurisitc proposed by Keuper et al. will be executed. If other setup wanted for heurisitc (see thesis for mentioned setups), see the src/command-line-tools/solve-regular script and change as described in file.

## Ackownledgement
GCN code is based on implementation provided by Wang et al. (https://github.com/Zhongdao/gcn_clustering). Person dataset is calculated using implementation provided by Wang (https://github.com/WangWenhao0716/Adapted-Center-and-Scale-Prediction). Reidentification dataset is calculated using implementation provided by Zheng et al. (https://github.com/layumi/Person_reID_baseline_pytorch). Primal Feasible Search Heuristic is based upon implementation provided by Keuper et al. code (https://github.com/bjoern-andres/graph).
