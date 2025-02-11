
import random
import yaml

from time import sleep
from os import listdir
from os.path import join

import cv2
import numpy as np
from matplotlib import pyplot

hl = 56
hh = 71

sl = 98
sh = 255

vl = 46
vh = 226


canny_l_t = 100
canny_h_t = 200

def get_training_image_names(orientation = 'f'):# 's' - side, 'f' - front view
    training_folder = "./CW_data/training"

    filenames = []
    for file in listdir(training_folder):
        if orientation in file:
            filenames.append(file)

    return [ join(training_folder, filename) for filename in filenames]

def background_identifier(images, title='img'):
    global hl, hh,sl,sh,vl,vh
    height = images[0].shape[0]
    width = images[0].shape[1]
    ratio = height/width

    WIDTH = 800
    HEIGHT = int(WIDTH * ratio)
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)

    def nothing(x):
        pass

    cv2.createTrackbar('Hl',title,hl,180,nothing)
    cv2.createTrackbar('Hh',title,hh,180,nothing)
    cv2.createTrackbar('Sl',title,sl,255,nothing)
    cv2.createTrackbar('Sh',title,sh,255,nothing)
    cv2.createTrackbar('Vl',title,vl,255,nothing)
    cv2.createTrackbar('Vh',title,vh,255,nothing)
    cv2.resizeWindow(title, (WIDTH, HEIGHT))

    i = 0
    while(1):
        img = images[i]
        #green
        lower = np.array([hl, sl, vl])
        upper = np.array([hh, sh, vh])

        mask1 = cv2.bitwise_not(cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV), lower, upper))

        res = cv2.bitwise_or(img, img, mask = mask1)
        concat_images = np.concatenate((img, cv2.cvtColor(mask1,cv2.COLOR_GRAY2BGR), res), axis=1)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 32:
            i += 1
            if (i>=len(images)):
                i=0

        # get current positions of four trackbars
        hl = cv2.getTrackbarPos('Hl',title)
        hh = cv2.getTrackbarPos('Hh',title)
        sl = cv2.getTrackbarPos('Sl',title)
        sh = cv2.getTrackbarPos('Sh',title)
        vl = cv2.getTrackbarPos('Vl',title)
        vh = cv2.getTrackbarPos('Vh',title)

        cv2.imshow(title,concat_images)
        sleep(0.1)

    cv2.destroyAllWindows()

def human_treadmill_extraction(images):
    width_h_pixel = int(images[0].shape[0]/2)
    height_w_pixel = int(images[0].shape[1]/2)

    lower = np.array([hl, sl, vl])
    upper = np.array([hh, sh, vh])

    min_h = images[0].shape[0]
    min_w = images[0].shape[1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    proc_images = []
    for img in images:
        mask1 = cv2.bitwise_not(cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV), lower, upper))

        #make img grayscale as colors are not needed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #find left right and top green screen limits
        left_limit = 0
        for i in range(len(mask1[width_h_pixel])):
            white = False
            for j in range(3): #number of consequtive white pixels
                if (mask1[width_h_pixel, i+j] == 0):
                    white = True
                else:
                    white = False
                    break

            if (white):
                left_limit = i
                break

        right_limit = 0
        for i in reversed(range(len(mask1[width_h_pixel]))):
            white = False
            for j in range(3): #number of consequtive white pixels
                if (mask1[width_h_pixel, i-j] == 0):
                    white = True
                else:
                    white = False
                    break

            if (white):
                right_limit = i
                break

        top_limit = 0
        for i in range(len(mask1[height_w_pixel])):
            white = False
            for j in range(3): #number of consequtive white pixels
                if (mask1[i+j, width_h_pixel] == 0):
                    white = True
                else:
                    white = False
                    break

            if (white):
                top_limit = i
                break

        #add some more padding to greenscreen limits
        left_limit+=50
        right_limit-=50
        top_limit+=50


        #apply mask to image
        masked_img = cv2.bitwise_or(img, img, mask = mask1)

        #crop masked image
        croped_img = masked_img[top_limit:masked_img.shape[0]-100, left_limit:right_limit]

        #apply histogram normalization
        nrm_img = clahe.apply(croped_img)

        proc_images.append(nrm_img)

        #keep track of minimum cropped img size so that all images could be set to same size later
        min_h = min(nrm_img.shape[0], min_h)
        min_w = min(nrm_img.shape[1], min_w)


    #set all images to the same smallest size
    for i in range(len(proc_images)):
        proc_images[i] = proc_images[i][:min_h, :min_w]

    #get median frame of this all pictures to remove treadmill
    #medianFrame = np.median(proc_images, axis=0).astype(dtype=np.uint8)
    #put in black boundary in usual person location to keep that part of the image
    #cv2.rectangle(medianFrame,(420,0),(900,medianFrame.shape[0]),0,-1)

    return proc_images

def get_contours(proc_images):
    imgs_contours = []
    imgs_shapes = []
    for i in range(len(proc_images)):
        # the median frame
        #proc_images[i] = cv2.absdiff(proc_images[i], medianFrame)
        ret,thres_img = cv2.threshold(proc_images[i],5,255,cv2.THRESH_BINARY)
        #c = cv2.Canny(proc_images[i],100,300)

        contour_thres, _ = cv2.findContours(thres_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        mx_contour = None
        max_contour_len = 0
        for c in contour_thres:
            if len(c) > max_contour_len:
                mx_contour = c
                max_contour_len = len(c)
                #print (len(c))
        contours.append(mx_contour)

        imgs_contours.append(contours)
        imgs_shapes.append(thres_img)
        contour_thres_img = np.zeros(proc_images[i].shape, np.uint8)
        cv2.drawContours(contour_thres_img, contours, -1, 100, 2)
        display_sidebyside([proc_images[i], thres_img, contour_thres_img])
        cv2.waitKey(0)

        '''
        contour_canny, _ = cv2.findContours(c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_canny_img = np.zeros(proc_images[i].shape, np.uint8)
        cv2.drawContours(contour_canny_img, contour_canny, -1, 100, 2)

        contour_nothing, _ = cv2.findContours(proc_images[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_nothing_img = np.zeros(proc_images[i].shape, np.uint8)
        cv2.drawContours(contour_nothing_img, contour_nothing, -1, 100, 2)

        display_sidebyside([proc_images[i], contour_nothing_img],title='proc_images')
        display_sidebyside([thres_img, contour_thres_img, c, contour_canny_img])
        cv2.waitKey(0)
        '''
        #===================== Don't think threshold binarization is needed here
        # Treshold to binarize

        #gives out all the contours of the image with right paramters
        '''
        th3 = cv2.adaptiveThreshold(proc_images[i],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,5)
        contour_th3, _ = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_th3_img = np.zeros(proc_images[i].shape, np.uint8)
        cv2.drawContours(contour_thres_img, contour_th3, -1, 100, 2)
        display_sidebyside([proc_images[i], th3, contour_th3_img], title='median_again')
        cv2.waitKey(0)
        '''

    return imgs_shapes, imgs_contours

    '''
    while True:
        for c in proc_images:
            display_sidebyside([c], title='cropped')
            #pyplot.hist(cl1.ravel(),256,[0,256]); pyplot.show()
            cv2.waitKey(00)
    '''

def get_total_heights(contours):
    heights = []

    for contour in contours:
        highest_y = 100000
        for cnt in contour[0]:
            y = cnt.item(1) #Y coordinate
            if y < highest_y:
                highest_y = y
        heights.append(highest_y)

    return heights

def display_together(imgs, title='img'):

    dst = np.zeros(imgs[0].shape, np.uint8)
    for img in imgs:
        dst = cv2.addWeighted(dst, 1, img, 1/len(imgs), 0)

    display(dst, title)
    cv2.waitKey(1)

    return dst

def display(img, title='img'):
    height = img.shape[0]
    width = img.shape[1]
    ratio = height/width

    WIDTH = 1800
    HEIGHT = int(WIDTH * ratio)
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, (WIDTH, HEIGHT))

    cv2.imshow(title, img)
    cv2.waitKey(1)

def display_sidebyside(imgs, title='img'):
    concat_img = np.concatenate(tuple(imgs), axis=1)
    display(concat_img, title=title)

def export_features(human_data, export_filename='features.yaml'):

    export_data = {}
    for human in human_data:
        export_data[human['name']] = human['total_height']

    with open(export_filename, 'w') as outfile:
        yaml.dump(export_data, outfile, default_flow_style=False)


def prepare():
    human_data = []
    img_names = get_training_image_names()

    images = []
    for img_name in img_names:
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        imgBil = cv2.bilateralFilter(img, 9, 75, 75)
        #display_sidebyside([img, imgMed, imgBil], title=img_name)

        images.append(imgBil)

        human_data.append( {'name':img_name, 'initial_img':imgBil} )

    #background_identifier(images)

    humans_thredmills = human_treadmill_extraction(images)
    for i in range(len(human_data)):
        human_data[i] = {**human_data[i], **{'foreground':humans_thredmills[i]}}

    shapes, contours = get_contours(humans_thredmills)
    for i in range(len(human_data)):
        human_data[i] = {**human_data[i], **{'shape':shapes[i], 'contour':contours[i]}}


    heights = get_total_heights(contours)
    for i in range(len(human_data)):
        human_data[i] = {**human_data[i], **{'total_height':heights[i]}}

    #for data in human_data:
        #cv2.line(data['foreground'], (0,data['total_height']), (1000,data['total_height']), 122, 5)
        #display(data['foreground'])
        #cv2.waitKey(0)

    export_features(human_data)