import torchvision
import torch
import csv

from os.path import join
from collections import OrderedDict
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO


def get_image_pairs():
    test_dir = './CW_data/test/'
    training_dir = './CW_data/training/'
    answers = OrderedDict()
    with open('test-training_map.txt', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if 'f' in row['training']:
                continue
            answers[ join(test_dir, row['test']) ] = join(training_dir, row['training'])

    return answers

def do_img(filename):
    image = Image.open(filename)

    image_tensor = torchvision.transforms.functional.to_tensor(image)

    # pass a list of (potentially different sized) tensors
    # to the model, in 0-1 range. The model will take care of
    # batching them together and normalizing
    output = do_img.model([image_tensor])
    # output is a list of dict, containing the postprocessed predictions

    plt.subplots(figsize=(50,100))
    tmp = np.array(image)
    mask = torch.zeros(tmp.shape)
    for keypoints in output[0]["keypoints"]:
        for keypoint in keypoints:
            x,y,_ = keypoint.data
            # print(int(y), int(x))
            for dx in range(-5,6):
                # mask[int(y), int(x),1] = 1
                mask[int(y+dx), int(x+dx),1] = 1
                mask[int(y-dx), int(x+dx),1] = 1
        break
    mask = mask  + tmp / 255  *0.6
    # plt.subplot(2,1,1)
    # plt.imshow(tmp)
    # plt.subplot(2,1,1)
    plt.imshow(mask)
    plt.show()

def main():

    do_img.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()
    test_training_mapping = get_image_pairs()

    num_of_pairs = 22
    sliced = islice(test_training_mapping.items(), num_of_pairs)
    trimmed_mapping = OrderedDict(sliced)

    for test,training in trimmed_mapping.items():
        print (test, training)
        do_img(test)
        do_img(training)


if __name__ == "__main__":
    main()

