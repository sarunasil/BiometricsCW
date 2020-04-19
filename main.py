import random

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

def align_images(images_all):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    im2 = images_all[0]
    im2Gray = im2#cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    aligned_images = [im2]
    for im1 in images_all[1:]:
        # Convert images to grayscale
        im1Gray = im1#cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width = im2.shape
        im_aligned = cv2.warpPerspective(im1, h, (width, height))


        aligned_images.append(im_aligned)

    return aligned_images

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

def canny_threshold_identifier(images, title='img'):
    global canny_l_t, canny_h_t

    cv2.namedWindow(title,cv2.WINDOW_NORMAL)

    def nothing(x):
        pass

    cv2.createTrackbar('Lower_threshold', title, canny_l_t, 600, nothing)
    cv2.createTrackbar('Higher_thresold', title, canny_h_t, 600, nothing)

    i = 0
    while(1):
        img = images[i]

        canny_edges = cv2.Canny(img, canny_l_t, canny_h_t)

        concat_images = np.concatenate((img, canny_edges), axis=1)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 32:
            i += 1
            if (i>=len(images)):
                i=0

        # get current positions of four trackbars
        canny_l_t = cv2.getTrackbarPos('Lower_threshold',title)
        canny_h_t = cv2.getTrackbarPos('Higher_threshold',title)

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


    for i in range(len(proc_images)):
        # the median frame
        #proc_images[i] = cv2.absdiff(proc_images[i], medianFrame)
        ret,thres_img = cv2.threshold(proc_images[i],5,255,cv2.THRESH_BINARY)
        #c = cv2.Canny(proc_images[i],100,300)

        contour_thres, _ = cv2.findContours(thres_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        for c in contour_thres:
            if len(c) > 1000:
                contours.append(c)
                print (len(c))
        contour_thres_img = np.zeros(proc_images[i].shape, np.uint8)
        cv2.drawContours(contour_thres_img, contours, -1, 100, 2)
        display_sidebyside([thres_img, contour_thres_img])
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

    return proc_images

    while True:
        for c in proc_images:
            display_sidebyside([c], title='cropped')
            #pyplot.hist(cl1.ravel(),256,[0,256]); pyplot.show()
            cv2.waitKey(00)


def display_together(imgs, title='img'):

    dst = np.zeros(imgs[0].shape, np.uint8)
    for img in imgs:
        dst = cv2.addWeighted(dst, 1, img, 1/len(imgs), 0)

    display_img(dst, title)
    cv2.waitKey(1)

    return dst

def display_img(img, title='img'):
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
    display_img(concat_img, title=title)

def main():
    img_names = get_training_image_names()

    images = []
    for img_name in img_names:
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        imgBil = cv2.bilateralFilter(img, 9, 75, 75)
        #display_sidebyside([img, imgMed, imgBil], title=img_name)

        images.append(imgBil)

    #background_identifier(images)
    humans_thredmills = human_treadmill_extraction(images)

    #canny_threshold_identifier(humans_thredmills)

    #display_together(images, 'initial')

    #aligned_images = align_images(images)

    #display_img(aligned_images[0], title='0')
    #display_img(aligned_images[1], title='1')

    #bg = display_together(aligned_images, 'alligned')

    #get_background(images)
    #fgMask = get_fg_mask(aligned_images, bg)
    #display_img(fgMask)


def fun():
    for img_name in img_names:
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        height = img.shape[0]
        width = img.shape[1]
        ratio = height/width

        HEIGHT = int(WIDTH * ratio)

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #green
        lower = np.array([60 - 10, 100, 100])
        upper = np.array([60 + 10, 255, 255])

        mask1 = cv2.inRange(hsv, lower, upper)

        #white
        lower = np.array([10, 0, 30])
        upper = np.array([40, 20, 70])

        mask2 = cv2.inRange(hsv, lower, upper)

        mask = cv2.bitwise_or(mask1, mask2)

        res = cv2.bitwise_and(img, img, mask = mask2)

        window_name = '1'
        cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, (WIDTH, HEIGHT))

        cv2.imshow(window_name, img)
        cv2.waitKey(1)
        sleep(1)

    while True:
        k = cv2.waitKey(100)

        if k != -1:
            break

        for img_name in img_names:
            if cv2.getWindowProperty(img_name, cv2.WND_PROP_VISIBLE) < 1:
                break


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    while True:
        cv2.waitKey(0)

'''
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    print (ret)
    cv2.imshow('image1',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''