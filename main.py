import random
import yaml
import csv

from time import sleep
from os import listdir
from os.path import join

import cv2
import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

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

    features_data = {}

    with open(import_filename, 'r') as infile:
        features_data = yaml.load(infile, Loader=yaml.FullLoader)

    return features_data

def knn_features_front(acc, i, d):

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

def knn_features_side(acc, i ,d):
    fj=0
    for name, value in sorted(d.items()):
        # print (name)
        if name == 'name' or name == 'orientation':
            if name == 'name':
                print (value)
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

def get_fastDTW(entry, base_cases):

    name = ''
    min_dist = None
    i = 1
    for base_case in base_cases:
        distance, path = fastdtw(base_case['wavelet'], entry['wavelet'], radius=4, dist=euclidean)
        print(i, distance, base_case['name'])
        if not min_dist or distance < min_dist:
            min_dist = distance
            name = base_case['name']
        i += 1

    return min_dist, name

def load_knn_models(features_data):

    # start front knn
    knn_front = cv2.ml.KNearest_create()
    trainDataFront = np.zeros((int(len(features_data)/2), len(features_data[0])*2  ),dtype=np.float32)
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

    responses_front = np.empty((int(len(features_data)/2),1),dtype=np.float32)
    index = 0
    for i in range(len(features_data)):
        if features_data[i]['orientation'] == 'f':
            responses_front[index][0] = i
            index += 1

    knn_front.train(trainDataFrontNorm, cv2.ml.ROW_SAMPLE, responses_front)
    # end front knn

    # start side knn
    knn_side = cv2.ml.KNearest_create()
    trainDataSide = np.zeros((int(len(features_data)/2), len(features_data[0])*2),dtype=np.float32)
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

    responses_side = np.empty((int(len(features_data)/2),1),dtype=np.float32)
    index = 0
    for i in range(len(features_data)):
        if features_data[i]['orientation'] == 's':
            responses_side[index][0] = i
            index += 1

    knn_side.train(trainDataSideNorm, cv2.ml.ROW_SAMPLE, responses_side)
    # end side knn

    return knn_front, knn_side, scaler_front, scaler_side

def identify():
    features_data = import_features()

    # human_data = prepare(get_test_image_names(), 'features_test.yaml')
    human_data = import_features('features_test.yaml')

    # knn_front, knn_side, scaler_front, scaler_side = load_knn_models(features_data)

    answers = {}
    with open('test-training_map.txt', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            answers[row['test']] = row['training']

    correct_guesses = []
    i = 1
    for human in human_data:
        if 'contour' in human:
            del human['contour']
            del human['initial_img']
            del human['person_mask']

        new = np.zeros((1,len(features_data[0])*2),dtype=np.float32)
        if human['orientation'] == 'f':
            knn_features_front(new, 0, human)

            newNorm = scaler_front.transform(new)
            ret, results, neighbours, dist = knn_front.findNearest( newNorm, 3)
        else:
            # knn_features_side(new, 0, human)

            # newNorm = scaler_side.transform(new)
            # ret, results, neighbours, dist = knn_side.findNearest( newNorm, 3)

            dist, name = get_fastDTW(human, features_data)
            print("BEST MATCH:", dist, name, human['name'])
            print()

        # print( "result:  {}\n".format(results) )
        # print( "neighbours:  {}\n".format(neighbours) )
        # for nei in neighbours[0]:
        #     print ( list(features_data)[int(nei)] )
        # print ("")
        # print ( "human:", human )
        # print( "distance:  {}\n".format(dist) )

        correct_answer = answers[ human['name'].split('/')[-1] ]
        print(f"correct answ: {correct_answer}, given asnwer - {name.split('/')[-1]}")
        if correct_answer == name.split('/')[-1]:
            print (f"{i}. YES")
            correct_guesses.append(i)
        else:
            print (f"{i}. NO")


        # img_training = [ cv2.imread( list(features_data)[int(n)]['name'], cv2.IMREAD_COLOR) for n in neighbours[0] ]
        # img_test = cv2.imread(human['name'] , cv2.IMREAD_COLOR )
        # display_sidebyside([img_test] + img_training, title='t '+str(i), wait=True)

        i+=1

    print ("Number of correct answers:",len(correct_guesses), "Out of:", len(human_data), correct_guesses)

def main():

    identify()


if __name__ == "__main__":
    main()
