import random
import yaml
import csv

from time import sleep
from os import listdir
from os.path import join

import cv2
import numpy as np
from matplotlib import pyplot
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

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

    #get correct matches
    answers = {}
    with open('test-training_map.txt', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            answers[row['test']] = row['training']

    # human_data = prepare(get_test_image_names(), 'features_test.yaml')
    human_data = import_features('features_test.yaml')

    knn = cv2.ml.KNearest_create()

    trainData = np.empty((len(features_data),(len(features_data[0])-1)*2 - 1  ),dtype=np.float32) #-1 for 'name', *2 for tuples, -1 for 'height'
    i=0
    for d in features_data:
        fj=0
        for name, value in d.items():#wrongly assuming dict items() order is consistent
            print (name)
            if name == 'name':
                continue
            #elif name == 'hip_width':
            #    continue
            elif type(value) == int:
                trainData[i][fj] = value
                fj += 1
            else:
                for v in value:
                    trainData[i][fj] = v
                    fj += 1
        i+=1
    #trainDataNorm = stats.zscore(trainData)

    scaler = MinMaxScaler()
    scaler.fit(trainData)

    trainDataNorm = scaler.transform(trainData)
    # trainDataNorm = trainData


    responses = np.empty((len(features_data),1),dtype=np.float32)
    for i in range(len(features_data)):
        responses[i][0] = i

    knn.train(trainDataNorm, cv2.ml.ROW_SAMPLE, responses)

    correct_guess = 0
    i = 1
    for human in human_data:
        if 'contour' in human:
            del human['contour']
            del human['initial_img']
            del human['person_img']

        print ("")
        new = np.empty((1,trainData.shape[1] ),dtype=np.float32)
        fj = 0
        for name, value in human.items():#wrongly assuming dict items() order is consistent
            print (name)
            if name == 'name':
                continue
            elif name == 'hip_width':
                continue
            elif type(value) == int:
                new[0][fj] = value
                fj += 1
            else:
                for v in value:
                    new[0][fj] = v
                    fj += 1

        #normalize 'new'
        # st_devs = np.std(trainData, axis=0)
        # avgs = np.mean(trainData, axis=0)
        newNorm = np.empty((1,trainData.shape[1]),dtype=np.float32)
        # for j in range(len(new[0])):
        #     newNorm[0, j] = (new[0, j] - avgs[j]) / st_devs[j]

        for j in range(len(new[0])):
            newNorm[0, j] = (new[0, j] - min(trainData[:, j])) / (max(trainData[:, j]) - min(trainData[:, j]))

        # newNorm = new
        ret, results, neighbours, dist = knn.findNearest( newNorm, 3)
        print( "result:  {}\n".format(results) )
        print( "neighbours:  {}\n".format(neighbours) )
        for nei in neighbours[0]:
            print ( list(features_data)[int(nei)] )
        print ("")
        print ( "human:", human )
        print( "distance:  {}\n".format(dist) )

        correct_answer = answers[ human['name'].split('/')[-1] ]
        if correct_answer == features_data[int(neighbours[0, 0])]['name'].split('/')[-1]:
            print (f"{i}. YES")
            correct_guess += 1


        img_training = [ cv2.imread( list(features_data)[int(n)]['name'], cv2.IMREAD_COLOR) for n in neighbours[0] ]
        img_test = cv2.imread(human['name'] , cv2.IMREAD_COLOR )
        # display_sidebyside([img_test] + img_training, title='t', wait=True)

        i+=1

    print ("Number of correct answers:",correct_guess, "Out of:", len(human_data))

def main():
    #prepare()

    identify()


if __name__ == "__main__":
    main()
