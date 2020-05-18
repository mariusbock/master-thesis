import os
import sys

import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

path = "/Users/mariusbock/Desktop/MOT17 raw/train/"


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


for image_folder in listdir_nohidden(path):
    with open(path + image_folder + "/det/det.txt", "r") as a_file:
        print(path + image_folder)
        for line in a_file:
            stripped_line = line.strip()

data = np.load('/Users/mariusbock/git/master-thesis/data/facedata/512.fea.npy')
labels = np.load('/Users/mariusbock/git/master-thesis/data/facedata/512.labels.npy')
graph = np.load('/Users/mariusbock/git/master-thesis/data/facedata/knn.graph.512.bf.npy')

#print(data)
#print(labels)
#print(graph)