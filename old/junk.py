import random

from time import sleep
from os import listdir
from os.path import join

import cv2
import numpy as np
from matplotlib import pyplot

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

'''
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    print (ret)
    cv2.imshow('image1',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''