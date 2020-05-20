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

from preparations import prepare, display_sidebyside

def get_test_image_names():
    test_folder = "./CW_data/test"

    filenames = []
    for file in listdir(test_folder):
        # if 'person' not in file:
        if ('person' not in file and int(file[-5])%2==0 and file[-7:]!="186.JPG") or file[-7:]=='185.JPG':
            filenames.append(file)

    return [ join(test_folder, filename) for filename in filenames]

def import_features(import_filename='features_training.yaml'):

    features_data = {}

    with open(import_filename, 'r') as infile:
        features_data = yaml.load(infile, Loader=yaml.FullLoader)

    return features_data

def identify():
    features_data = import_features()

    answers = {}
    with open('test-training_map.txt', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            answers[row['test']] = row['training']

    # human_data = prepare(get_test_image_names(), 'features_test.yaml')
    human_data = import_features('features_test.yaml')

    knn = cv2.ml.KNearest_create()

    # trainData = np.empty((len(features_data),10),dtype=np.float32)
    trainData = np.empty((len(features_data),9),dtype=np.float32)
    i=0
    heads = np.empty((len(features_data),1),dtype=np.float32)
    for d in features_data:
        trainData[i][0] = d['head_width'][0]
        trainData[i][1] = d['head_width'][1]
        # #trainData[i][2] = d['hip_width'][0]
        # trainData[i][2] = 0
        # trainData[i][3] = d['hip_width'][1]
        # trainData[i][4] = d['neck_width'][0]
        # trainData[i][5] = d['neck_width'][1]
        # trainData[i][6] = d['shoulder_width'][0]
        # trainData[i][7] = d['shoulder_width'][1]
        # trainData[i][8] = d['person_height']
        #
        trainData[i][2] = d['hip_width'][1]
        trainData[i][3] = d['neck_width'][0]
        trainData[i][4] = d['neck_width'][1]
        trainData[i][5] = d['shoulder_width'][0]
        trainData[i][6] = d['shoulder_width'][1]
        trainData[i][7] = d['person_height']
        trainData[i][8] = d['knee_width'][0]
        # trainData[i][9] = d['knee_width'][1]
        heads[i][0] = d['head_width'][0]
        i+=1



# ---------------------------------------------------


    trainData1 = np.empty((len(features_data),(len(features_data[0])-1)*2 - 1  ),dtype=np.float32) #-1 for 'name', *2 for tuples, -1 for 'height'
    i=0
    for d in features_data:
        fj=0
        for name, value in sorted(d.items()):
            print (name)
            if name == 'name':
                continue
            #elif name == 'hip_width':
            #    continue
            elif type(value) == int:
                trainData1[i][fj] = value
                fj += 1
            else:
                for v in value:
                    trainData1[i][fj] = v
                    fj += 1
        i+=1

# -----------------------------------------------

    # trainDataNorm = cv2.normalize(trainData, None, norm_type=cv2.NORM_INF)
    # trainDataNorm = trainData


    # scaler = preprocessing.MinMaxScaler()
    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.RobustScaler()

    print(scaler.fit(trainData))

    trainDataNorm = scaler.transform(trainData)

# --------
    scaler1 = preprocessing.RobustScaler()

    print(scaler1.fit(trainData1))

    trainDataNorm1 = scaler1.transform(trainData1)
# ------

    responses = np.empty((len(features_data),1),dtype=np.float32)
    for i in range(len(features_data)):
        responses[i][0] = i

    knn.train(trainDataNorm1, cv2.ml.ROW_SAMPLE, responses)

    correct_guesses = []
    i = 1
    for human in human_data:
        if 'contour' in human:
            del human['contour']
            del human['initial_img']
            del human['person_img']


        new = np.empty((1,trainData.shape[1]), dtype=np.float32)
        new[0][0] = human['head_width'][0]
        new[0][1] = human['head_width'][1]
        #new[0][2] = human['hip_width'][0]
        # new[0][2] = 0
        # new[0][3] = human['hip_width'][1]
        # new[0][4] = human['neck_width'][0]
        # new[0][5] = human['neck_width'][1]
        # new[0][6] = human['shoulder_width'][0]
        # new[0][7] = human['shoulder_width'][1]
        # new[0][8] = human['person_height']
        new[0][2] = human['hip_width'][1]
        new[0][3] = human['neck_width'][0]
        new[0][4] = human['neck_width'][1]
        new[0][5] = human['shoulder_width'][0]
        new[0][6] = human['shoulder_width'][1]
        new[0][7] = human['person_height']
        new[0][8] = human['knee_width'][0]
        # new[0][9] = human['knee_width'][1]
        # newNorm = cv2.normalize(np.vstack([trainData, new]), None, norm_type=cv2.NORM_INF)[-1:]

        newNorm = scaler.transform(new)
# -----------------------------------------------
        new1 = np.empty((1,trainData1.shape[1] ),dtype=np.float32)
        fj = 0
        for name, value in sorted(human.items()):
            print (name)
            if name == 'name':
                continue
            elif name == 'hip_width':
                continue
            elif type(value) == int:
                new1[0][fj] = value
                fj += 1
            else:
                for v in value:
                    new1[0][fj] = v
                    fj += 1
# -----------------------------------------------

        newNorm1 = scaler1.transform(new1)
        # newNorm = new

        ret, results, neighbours, dist = knn.findNearest( newNorm1, 3)
        # print( "result:  {}\n".format(results) )
        # print( "neighbours:  {}\n".format(neighbours) )
        # for nei in neighbours[0]:
        #     print ( list(features_data)[int(nei)] )
        # print ("")
        # print ( "human:", human )
        print( "distance:  {}\n".format(dist) )

        correct_answer = answers[ human['name'].split('/')[-1] ]
        if correct_answer == features_data[int(neighbours[0, 0])]['name'].split('/')[-1]:
            print (f"{i}. YES")
            correct_guesses.append(i)


        img_training = [ cv2.imread( list(features_data)[int(n)]['name'], cv2.IMREAD_COLOR) for n in neighbours[0] ]
        img_test = cv2.imread(human['name'] , cv2.IMREAD_COLOR )
        # display_sidebyside([img_test] + img_training, title='t', wait=True)

        i+=1

    print ("Number of correct answers:",len(correct_guesses), "Out of:", len(human_data), correct_guesses)

def main():
    #prepare()

    identify()


if __name__ == "__main__":
    main()
