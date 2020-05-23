import random
import yaml
import csv

from time import sleep
from os import listdir
from os.path import join
from copy import copy

import cv2
import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing

from preparations import prepare, display_sidebyside

def get_test_image_names():
    test_folder = "./CW_data/test"

    filenames = []
    for file in listdir(test_folder):
        if 'person' in file:
            continue
        elif (int(file[-5])%2==0 and file[-7:]!="186.JPG") or file[-7:]=='185.JPG': # 'f' goes through this
        #     continue
        # else:
            filenames.append(file)

    return [ join(test_folder, filename) for filename in filenames]

def import_features(import_filename='features_training.yaml'):

    features_data = {}

    with open(import_filename, 'r') as infile:
        features_data = yaml.load(infile, Loader=yaml.FullLoader)

    return features_data

def knn_features_front(acc, i, d):

    fj=0
    for name, value in sorted(d.items()):
        # print (fj, name, value)
        if (name == 'name' or 
        name == 'ankle_width' or 
        name == 'knee_width' or 
        name == 'shoulder_width' or 
        # name == 'neck_width' or 
        # name == 'head_width' or 
        # name == 'person_height' or 
        name == 'hip_width' or 
        name == 'orientation'):
            continue
        # elif name == 'neck_width':
        #     acc[i][fj] = value[0]
        #     fj+=1
        # elif name == 'knee_width':
        #     acc[i][fj] = value[0]
        #     fj+=1
        elif type(value) == int:
            acc[i][fj] = value
            fj += 1
        else:
            for v in value:
                acc[i][fj] = v
                fj += 1
    # print(acc[i])

def knn_features_side(acc, i ,d):
    fj=0
    for name, value in sorted(d.items()):
        # print (name)
        if name == 'name' or name == 'orientation':
            continue
        elif name == 'hip_width':
            acc[i][fj] = value[1]
            fj+=1
        elif name == 'knee_width':
            acc[i][fj] = value[0]
            fj+=1
        elif type(value) == int:
            acc[i][fj] = value
            fj += 1
        else:
            for v in value:
                acc[i][fj] = v
                fj += 1


def load_knn_models(features_data):

    side_pics = 0
    for f in features_data:
        if f['orientation'] == 's':
            side_pics += 1

    # start front knn
    if side_pics != len(features_data):
        knn_front = cv2.ml.KNearest_create()
        trainDataFront = np.zeros((len(features_data) - side_pics, len(features_data[0])*2  ),dtype=np.float32)
        i=0
        for d in features_data:
            if d['orientation'] == 's':
                continue
            knn_features_front(trainDataFront, i, d)
            i+=1

        # scaler_front = preprocessing.MinMaxScaler()
        # scaler_front = preprocessing.StandardScaler()
        scaler_front = preprocessing.RobustScaler()

        scaler_front.fit(trainDataFront)
        trainDataFrontNorm = scaler_front.transform(trainDataFront)

        responses_front = np.empty((len(features_data)-side_pics,1),dtype=np.float32)
        index = 0
        for i in range(len(features_data)):
            if features_data[i]['orientation'] == 'f':
                responses_front[index][0] = i
                index += 1

        knn_front.train(trainDataFrontNorm, cv2.ml.ROW_SAMPLE, responses_front)
    else:
        knn_front = scaler_front = None
    # end front knn

    # start side knn
    if side_pics > 0:
        knn_side = cv2.ml.KNearest_create()
        trainDataSide = np.zeros((side_pics, len(features_data[0])*2),dtype=np.float32)
        i=0
        for d in features_data:
            if d['orientation'] == 'f':
                continue
            knn_features_side(trainDataSide, i, d)
            i+=1

        # scaler_side = preprocessing.MinMaxScaler()
        # scaler_side = preprocessing.StandardScaler()
        scaler_side = preprocessing.RobustScaler()

        scaler_side.fit(trainDataSide)
        trainDataSideNorm = scaler_side.transform(trainDataSide)

        responses_side = np.empty((side_pics,1),dtype=np.float32)
        index = 0
        for i in range(len(features_data)):
            if features_data[i]['orientation'] == 's':
                responses_side[index][0] = i
                index += 1

        knn_side.train(trainDataSideNorm, cv2.ml.ROW_SAMPLE, responses_side)
    else:
        knn_side = scaler_side = None
    # end side knn

    return knn_front, knn_side, scaler_front, scaler_side

def get_correct_answers():
    answers = {}
    with open('test-training_map.txt', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            answers[row['test']] = row['training']
    return answers

def identify(human, features_data, knn_front, knn_side, scaler_front, scaler_side, verbose = False):
    if 'contour' in human:
        del human['contour']
        del human['initial_img']
        del human['person_mask']
        del human['dots_img']
        del human['pose_img']

    new = np.zeros((1,len(features_data[0])*2),dtype=np.float32)
    if human['orientation'] == 'f':
        knn_features_front(new, 0, human)

        newNorm = scaler_front.transform(new)
        ret, results, neighbours, dist = knn_front.findNearest( newNorm, 1)
    else:
        knn_features_side(new, 0, human)

        newNorm = scaler_side.transform(new)
        ret, results, neighbours, dist = knn_side.findNearest( newNorm, 1)

    guess_id = int(neighbours[0,0])
    guess_name = features_data[guess_id]['name']
    if verbose:
        print (f"Matching: {human['name']}")
        print (f"Guessed answer {guess_name}")

        print( "result:  {}\n".format(results) )
        print( "neighbours:  {}\n".format(neighbours) )
        for nei in neighbours[0]:
            print ( list(features_data)[int(nei)]['name'] )
        print ("")
        print ( "human:", human )
        print( "distance:  {}\n".format(dist) )

    return guess_id, guess_name, dist[0]


def authenticate(features_data, human_data, threshold = 100000000, check_answers = True, verbose = False, vverbose = False):
    knn_front, knn_side, scaler_front, scaler_side = load_knn_models(features_data)

    answers = get_correct_answers()

    correct_guesses = []
    i = 1
    for human in human_data:
        correct_answer = answers[ human['name'].split('/')[-1] ]

        guess_id, guess_name, dist = identify(human, features_data, knn_front, knn_side, scaler_front, scaler_side, vverbose)

        if (correct_answer == guess_name.split('/')[-1] or not check_answers) and dist < threshold:
            print (f"{i}. YES {dist} {human['name']}") if vverbose else True

            correct_guesses.append(i)
        else:
            print (f"{i}. NO {dist} {human['name']}") if vverbose else True

        i+=1

    if verbose:
        if check_answers:
            print ("Number of correct answers:",len(correct_guesses), "Out of:", len(human_data), correct_guesses)
        else:
            print("Number of authentications:", len(correct_guesses))

    return len(correct_guesses)


def main():
    features_data = import_features()

    # human_data = prepare(get_test_image_names(), 'features_test.yaml')
    human_data = import_features('features_test.yaml')

    answers = get_correct_answers()
    features_data_far = features_data.copy()
    for _,training in answers.items():
        for i in range(len(features_data_far)):
            if training in features_data_far[i]['name']:
                del features_data_far[i]
                break

    far = {}
    frr = {}
    for threshold in np.arange(0, 2,0.01):
        authenticated_real = authenticate(features_data, human_data, threshold)
        frr[threshold] = (len(human_data) - authenticated_real) / len(human_data)

        authenticated_false = authenticate(features_data_far, human_data, threshold, check_answers = False)
        far[threshold] = authenticated_false / len(human_data)

    x_frr, y_frr = zip(*sorted(frr.items()))
    x_far, y_far = zip(*sorted(far.items()))

    idx = np.argwhere( np.diff(np.sign( np.array(y_frr) - np.array(y_far) )) ).flatten()
    if len(idx) > 0:
        idx = idx[0]
        authenticated_real = authenticate(features_data, human_data, x_frr[idx], verbose = True)
        print (f"EER = {round(y_far[idx]*100,2)}%")
    else:
        print("No intersection")

    pyplot.plot(x_frr, y_frr, 'b', label='FRR')
    pyplot.plot(x_far, y_far, 'r', label='FAR')
    pyplot.yticks(np.arange(0,1.1,0.1))
    pyplot.legend()
    pyplot.grid()
    pyplot.show()


if __name__ == "__main__":
    main()
