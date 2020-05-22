import random
import yaml
import csv

from time import sleep
from os import listdir
from os.path import join
from collections import OrderedDict
from itertools import islice

import cv2
import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing

from preparations import prepare, display_sidebyside

def get_image_pairs():
    test_dir = './CW_data/test/'
    training_dir = './CW_data/training/'
    answers = OrderedDict()
    with open('test-training_map.txt', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if 's' in row['training']:
                continue
            answers[ join(test_dir, row['test']) ] = join(training_dir, row['training'])

    return answers

def visualize(test_training_mapping, tests, trainings):

    for i in range(len(tests)):
        # test_dots = tests[i]['dots_img']
        test_pose = tests[i]['pose_img']

        # training_dots = trainings[i]['dots_img']
        training_pose = trainings[i]['pose_img']

        # display_sidebyside([test_dots, training_dots, test_pose, training_pose])

        rcnn_test_pose = tests[i]['rcnn_pose_img']
        rcnn_training_pose = trainings[i]['rcnn_pose_img']
        display_sidebyside([rcnn_test_pose, test_pose, rcnn_training_pose, training_pose])

def main():
    test_training_mapping = get_image_pairs()

    num_of_pairs = 22
    sliced = islice(test_training_mapping.items(), num_of_pairs)
    trimmed_mapping = OrderedDict(sliced)

    trainings = prepare(list(trimmed_mapping.keys()))
    tests = prepare(list(trimmed_mapping.values()))

    while True:
        visualize(test_training_mapping, tests, trainings)



if __name__ == "__main__":
    main()
