import random
import yaml
import csv
import itertools

from time import sleep
from os import listdir
from os.path import join
from copy import copy
from collections import OrderedDict

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
            continue
        else:
            filenames.append(file)

    return [ join(test_folder, filename) for filename in filenames]

def import_features(import_filename='features_training.yaml'):

    features_data = OrderedDict()

    with open(import_filename, 'r') as infile:
        features_data = yaml.load(infile, Loader=yaml.FullLoader)

    return features_data

def knn_features_front(acc, i, d):

    fj=0

    if knn_features_front.selected_features:
        for feature in knn_features_front.selected_features:
            feature_name, pos = feature.split(":")

            value = d[feature_name]
            if pos == '':
                acc[i][fj] = value
                fj += 1
            else:
                pos = int(pos)
                acc[i][fj] = value[pos]
                fj += 1

    else:
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
            name == 'neck_pos' or 
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

    if knn_features_side.selected_features:
        for feature in knn_features_side.selected_features:
            feature_name, pos = feature.split(":")

            value = d[feature_name]
            if pos == '':
                acc[i][fj] = value
                fj += 1
            else:
                pos = int(pos)
                acc[i][fj] = value[pos]
                fj += 1

    else:
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
                name == 'neck_pos' or 
                name == 'orientation'):
                    continue
                # elif name == 'neck_width':
                #     acc[i][fj] = value[0]
                #     fj+=1
                # elif name == 'knee_width':
                #     acc[i][fj] = value[0]
                #     fj+=1
                elif type(value) == int or type(value) == float:
                    acc[i][fj] = value
                    fj += 1
                else:
                    for v in value:
                        acc[i][fj] = v
                        fj += 1
    # print(acc[i])

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
        scaler_front = preprocessing.StandardScaler()
        # scaler_front = preprocessing.RobustScaler()

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

        scaler_side = preprocessing.MinMaxScaler()
        # scaler_side = preprocessing.StandardScaler()
        # scaler_side = preprocessing.RobustScaler()

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
        ret, results, neighbours, dist = knn_front.findNearest( newNorm, 3)
    else:
        knn_features_side(new, 0, human)

        newNorm = scaler_side.transform(new)
        ret, results, neighbours, dist = knn_side.findNearest( newNorm, 3)

    guess_id = int(neighbours[0,0])
    guess_name = features_data[guess_id]['name']
    if verbose:
        print (f"Matching: {human['name']}")
        print (f"Guessed answer {guess_name}")

        print( "result:  {}".format(results) )
        print( "neighbours:  {}".format(neighbours) )
        for nei in neighbours[0]:
            print ( list(features_data)[int(nei)]['name'] )
        # print ("")
        # print ( "human:", human )
        print( "distance:  {}".format(dist) )

    return guess_id, guess_name, dist[0]


def authenticate(features_data, human_data, threshold = 100000000, check_answers = True, verbose = False, vverbose = False):
    knn_front, knn_side, scaler_front, scaler_side = load_knn_models(features_data)

    answers = get_correct_answers()

    correct_guesses = []
    i = 1
    for human in human_data:
        correct_answer = answers[ human['name'].split('/')[-1] ]
        # if 's' in correct_answer:
        #     continue

        guess_id, guess_name, dist = identify(human, features_data, knn_front, knn_side, scaler_front, scaler_side, verbose = vverbose)

        if len(dist) > 1:
            dist = dist[0]

        if (correct_answer == guess_name.split('/')[-1] or not check_answers) and dist < threshold:
            print (f"{i}. YES {dist} {human['name']}\n\n") if vverbose else True

            correct_guesses.append(i)
        else:
            print (f"{i}. NO {dist} {human['name']}\n\n") if vverbose else True

        i+=1

    if verbose:
        if check_answers:
            print ("Number of correct answers:",len(correct_guesses), "Out of:", len(human_data), correct_guesses)
        else:
            print("Number of authentications:", len(correct_guesses))

    return len(correct_guesses)

def correct_classification_rate():
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

    best_feature_score = {}
    best_feature_label = {}
    features = ['ankle_width:0', 'head_width:0', 'hip_width:0', 'knee_width:0', 'neck_pos:', 'neck_width:0', 'person_height:', 'shoulder_width:0', 'ankle_width:1', 'head_width:1', 'hip_width:1', 'knee_width:1', 'neck_width:1', 'shoulder_width:1']
    for L in range(1, len(features)+1):
    # for L in range(1, 5):
        for selected_features in itertools.combinations(features, L):
            # print(selected_features)
            knn_features_front.selected_features = selected_features
            knn_features_side.selected_features = selected_features
            classified_correct = authenticate(features_data, human_data) / len(human_data) * 100

            if L not in best_feature_score or best_feature_score[L] < classified_correct:
                best_feature_score[ L ] = classified_correct
                best_feature_label[ L ] = selected_features
        print (L)

    feature_scores = []
    for k,v in best_feature_score.items():
        feature_scores.append( (k,v) )

    feature_scores.sort(key=lambda x:x[1])
    x_f, y_f = zip(*sorted(feature_scores))

    print (best_feature_score)
    print (best_feature_label)
    pyplot.bar(x_f, y_f, align='center', alpha=0.5)

    pyplot.ylabel('Max Correct classification Rate (%)')
    pyplot.xlabel('Number of features used')
    pyplot.yticks(np.arange(0,101,5))
    pyplot.xticks(np.arange(0,len(feature_scores)+1,1))
    pyplot.grid()
    pyplot.show()
    return

    # authenticated_real = authenticate(features_data, human_data, verbose = True, vverbose=True)
    # return


def equal_error_rate():

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

    knn_features_front.selected_features = False
    knn_features_side.selected_features = False
    far = {}
    frr = {}
    acc = {}
    max_correct = 0
    max_threshold = None
    for threshold in np.arange(0, 0.4,0.001):
        authenticated_real = authenticate(features_data, human_data, threshold, check_answers= False)
        frr[threshold] = (len(human_data) - authenticated_real) / len(human_data) * 100

        authenticated_false = authenticate(features_data_far, human_data, threshold, check_answers = False)
        far[threshold] = authenticated_false / len(human_data) * 100

        correct = authenticated_real + (len(human_data) - authenticated_false)
        if max_correct < correct:
            max_correct = correct
            max_threshold = threshold

        acc[threshold] = correct / (len(human_data)*2) * 100

    x_frr, y_frr = zip(*sorted(frr.items()))
    x_far, y_far = zip(*sorted(far.items()))
    x_acc, y_acc = zip(*sorted(acc.items()))

    authenticate(features_data, human_data, threshold=max_threshold, verbose = True)
    idx = np.argwhere( np.diff(np.sign( np.array(y_frr) - np.array(y_far) )) ).flatten()
    id = idx[1]
    pyplot.plot(x_frr[id], y_frr[id], 'go')
    print (f"EER = {round(y_frr[id],2)}%")
    authenticate(features_data, human_data, threshold=x_far[id], verbose = True)

    pyplot.plot(x_acc, y_acc, 'y', label='Accuracy')
    pyplot.plot(x_frr, y_frr, 'b', label='FRR')
    pyplot.plot(x_far, y_far, 'r', label='FAR')
    pyplot.ylabel('Percentage (%)')
    pyplot.xlabel('Threshold value')
    pyplot.yticks(np.arange(0,101,5))
    pyplot.legend()
    pyplot.grid()
    pyplot.show()


def main():
    # equal_error_rate()
    correct_classification_rate()

if __name__ == "__main__":
    main()
