import random
import yaml

from time import sleep
from os import listdir
from os.path import join

import cv2
import numpy as np
from matplotlib import pyplot

from preparations import prepare, get_contour, get_person_height, display_sidebyside, display_together

def get_test_image_names(orientation = 'f'):# 's' - side, 'f' - front view
    test_folder = "./CW_data/test"

    filenames = []
    a=False
    for file in listdir(test_folder):
        a = not a
        if a:
            continue
        else:
            filenames.append(file)

    return [ join(test_folder, filename) for filename in filenames]

def import_features(import_filename='features.yaml'):

    features_data = {}

    with open(import_filename, 'r') as infile:
        features_data = yaml.load(infile, Loader=yaml.FullLoader)

    return features_data

def identify():
    features_data = import_features()

    a = cv2.imread('./CW_data/training/021z005pf.jpg')
    b = cv2.imread('./CW_data/test/DSC00174.JPG')

    display_together([a,b], title='aaaa')
    display_sidebyside([a,b], title='bbb')
    cv2.waitKey(00)

    human_data = []
    img_names = get_test_image_names()

    images = []
    for img_name in img_names:
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        imgBil = cv2.bilateralFilter(img, 9, 75, 75)
        #display_sidebyside([img, imgMed, imgBil], title=img_name)

        images.append(imgBil)

        human_data.append( {'name':img_name, 'initial_img':imgBil} )

    humans_thredmills = human_treadmill_extraction(images)
    for i in range(len(human_data)):
        human_data[i] = {**human_data[i], **{'foreground':humans_thredmills[i]}}

    shapes, contours = get_contours(humans_thredmills)
    for i in range(len(human_data)):
        human_data[i] = {**human_data[i], **{'shape':shapes[i], 'contour':contours[i]}}


    heights = get_total_heights(contours)
    for i in range(len(human_data)):
        human_data[i] = {**human_data[i], **{'total_height':heights[i]}}



    for _, height in features_data.items():
        pyplot.scatter(height,[5])

    newcomer = (human_data[0]['name'], human_data[0]['total_height'])
    pyplot.scatter(newcomer[1],[4])

    #for human in human_data:
    #    pyplot.scatter(human['total_height'],[4],marker='4')


    knn = cv2.ml.KNearest_create()

    trainData = np.empty((len(features_data),1),dtype=np.float32)
    i=0
    for v in features_data.values():
        trainData[i][0] = v
        i+=1

    responses = np.empty((len(features_data),1),dtype=np.float32)
    for i in range(len(features_data)):
        responses[i][0] = i

    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses )
    #knn.train(np.array([ h for h in features_data.values() ]), cv2.ml.ROW_SAMPLE, np.array([ n for n in features_data ]) )
    new = np.empty((1,1), dtype=np.float32)
    new[0] = newcomer[1]
    ret, results, neighbours, dist = knn.findNearest( new, 8)
    print( "result:  {}\n".format(results) )
    print( "neighbours:  {}\n".format(neighbours) )
    print ( list(features_data)[int(neighbours.item(0))] )
    print ( newcomer[0] )
    print( "distance:  {}\n".format(dist) )


    img_training = [ cv2.imread( list(features_data)[int(n)], cv2.IMREAD_COLOR) for n in neighbours[0] ]
    img_test = cv2.imread(newcomer[0] , cv2.IMREAD_COLOR )
    display_sidebyside([img_test] + img_training, title='t')

    #pyplot.show()

    while True:
        cv2.waitKey(0)


def main():
    prepare()

    #identify()


#canny_threshold_identifier(humans_thredmills)

#display_together(images, 'initial')

#aligned_images = align_images(images)

#display_img(aligned_images[0], title='0')
#display_img(aligned_images[1], title='1')

#bg = display_together(aligned_images, 'alligned')

#get_background(images)
#fgMask = get_fg_mask(aligned_images, bg)
#display_img(fgMask)


if __name__ == "__main__":
    main()
