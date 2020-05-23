
import random
import yaml
import math

from time import sleep, time
from os import listdir
from os.path import join, exists
from statistics import mean


import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import pyplot
from torchvision import transforms, models
from torchvision.utils import save_image
from sklearn import preprocessing



def get_training_image_names(orientation = 'f'):# 's' - side, 'f' - front view
    training_folder = "./CW_data/training"

    filenames = []
    for file in listdir(training_folder):
        if orientation in file and 'person' not in file:
        # if 'person' not in file:
            filenames.append(file)

    return [ join(training_folder, filename) for filename in filenames]

def get_person(filename):
    """extracts person from the background

    Taken from Pytorch examples:
    https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

    Arguments:
        filename {String} -- person picture filename
    """

    input_image = Image.open(filename)

    #if done previously
    prep_img_name = filename[:-4]+"_person.jpg"
    if exists(prep_img_name):
        print ("Using cached person shape")
        r = Image.open(prep_img_name)

        ret,thresh1 = cv2.threshold( cv2.cvtColor(np.asarray(r), cv2.COLOR_RGB2GRAY) ,127,255,cv2.THRESH_BINARY)
        return cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR), thresh1


    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        get_person.model.to('cuda')

    with torch.no_grad():
        output = get_person.model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    colors = torch.as_tensor([(255,255,255) for i in range(21)])
    colors[0] = torch.as_tensor([0,0,0])
    colors = (colors).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)


    r = r.convert('RGB')
    #save for person image for later
    r.save(prep_img_name)

    ret,thresh1 = cv2.threshold( cv2.cvtColor(np.asarray(r), cv2.COLOR_RGB2GRAY) ,127,255,cv2.THRESH_BINARY)
    return cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR), thresh1

def get_contour(person_mask):

    if len(person_mask.shape) > 2:
        person_mask = cv2.cvtColor(person_mask, cv2.COLOR_BGR2GRAY)
        ret,person_mask = cv2.threshold(person_mask,127,255,cv2.THRESH_BINARY)
    contour_thres, _ = cv2.findContours(person_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours = []
    mx_contour = None
    max_contour_len = 0
    for c in contour_thres:
        if len(c) > max_contour_len:
            mx_contour = c
            max_contour_len = len(c)
            #print (len(c))
    contours.append(mx_contour)

    contour_thres_img = np.zeros(person_mask.shape, np.uint8)
    cv2.drawContours(contour_thres_img, contours, -1, 100, 2)
    #display_sidebyside([person_mask, contour_thres_img])

    return contours

def get_keypoints_rcnn(cv_img):

    image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    # For reversing the operation:
    # im_np = np.asarray(im_pil)

    image_tensor = transforms.functional.to_tensor(image)

    output = get_keypoints_rcnn.model([image_tensor])
    # output is a list of dict, containing the postprocessed predictions

    parts = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
    dots = {}
    for keypoints in output[0]["keypoints"]:
        i = 0
        for keypoint in keypoints:
            x,y,_ = keypoint.data
            # print(int(y), int(x))
            dots[parts[i]] = (int(x), int(y))

            cv2.ellipse(cv_img, dots[parts[i]], (2, 2), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.putText(cv_img, str(i), dots[parts[i]], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            i+=1
        break

    # display(cv_img)
    return cv_img, dots

def get_orientation(body_parts):

    if body_parts['RWrist'] and body_parts['RElbow'] and body_parts['REar'] and abs(body_parts['RShoulder'][0] - body_parts['LShoulder'][0]) > 80:
        return 'f'
    else:
        return 's'

def get_measurements(first_dot, contour_img, body_parts, orientation):

    height = int(first_dot[1])
    cv2.line(contour_img,(first_dot[0]-100,height),(first_dot[0]+100,height),128,2)

    head_width = get_head_width(contour_img, body_parts, orientation)
    neck_width = get_neck_width(contour_img, body_parts, orientation)
    shoulder_width = get_shoulder_width(contour_img, body_parts, orientation)
    hip_width = get_hip_width(contour_img, body_parts, orientation)
    knee_width = get_knee_width(contour_img, body_parts, orientation)
    ankle_width = get_ankle_width(contour_img, body_parts, orientation)

    return {
        "person_height": height,
        "head_width": head_width,
        "neck_width": neck_width,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "knee_width": knee_width,
        "ankle_width": ankle_width
    }

def get_head_width(contour_img, body_parts, orientation):
    head_range = 1#maybe turn off range, since ear detection is accurate?

    max_head_width = 0
    ml = mr = my = None

    if body_parts['REar']:
        initial_head_y = int((body_parts['LEar'][1] + body_parts['REar'][1]) / 2)
    else:
        initial_head_y = int(body_parts['LEar'][1])
    for head_y in range(initial_head_y - head_range, initial_head_y + head_range):

        left = body_parts['REar'][0] if orientation == 'f' else body_parts['LEar'][0]
        for x in reversed(range(0, left)):
            if contour_img[head_y, x] == 0:
                left = x
                break

        right = body_parts['LEar'][0]
        for x in range(right, len(contour_img[0])):
            if contour_img[head_y, x] == 0:
                right = x
                break

        if not max_head_width or max_head_width < right-left:
            max_head_width = right-left
            mr = right
            ml = left
            my = head_y

    cv2.ellipse(contour_img, (ml, my), (8, 8), 0, 0, 360, 127, cv2.FILLED)
    cv2.ellipse(contour_img, (mr, my), (8, 8), 0, 0, 360, 127, cv2.FILLED)
    #display(contour_img, wait=True)

    return (max_head_width, my)

def get_neck_width(contour_img, body_parts, orientation):
    nose = body_parts['Nose']
    shoulder_height = int((body_parts['LShoulder'][1] + body_parts['RShoulder'][1]) / 2)

    min_neck = (len(contour_img[0]), nose[1])

    if orientation == 'f':
        ml = mr = my = None

        initial_neck_y = min_neck[1]
        for neck_y in range(initial_neck_y, shoulder_height):

            left = right = nose[0]
            for x in reversed(range(0, left)):
                if contour_img[neck_y, x] == 0:
                    left = x
                    break

            for x in range(right, len(contour_img[0])):
                if contour_img[neck_y, x] == 0:
                    right = x
                    break

            if min_neck[0] > right-left:
                min_neck = (right-left, neck_y)
                mr = right
                ml = left
                my = neck_y
        cv2.ellipse(contour_img, (ml, my), (8, 8), 0, 0, 360, 127, cv2.FILLED)
        cv2.ellipse(contour_img, (mr, my), (8, 8), 0, 0, 360, 127, cv2.FILLED)
    else:
        min_neck = (100,100)
        # initial_neck = body_parts['Neck']
        # l_ear = body_parts['LEar']

        # adj_node = ( l_ear[0], initial_neck[1] )

        # hip = math.sqrt((initial_neck[0] - l_ear[0])**2 + (initial_neck[1] - l_ear[1])**2)
        # adj = math.sqrt((adj_node[0] - initial_neck[0])**2 + (adj_node[1] - initial_neck[1])**2)
        # neck_angle = math.acos( adj/hip )
        # print(neck_angle)

    # display(contour_img)

    return min_neck

def get_shoulder_width(contour_img, body_parts, orientation):
    if orientation == 'f':
        shoulder_height = int((body_parts['LShoulder'][1] + body_parts['RShoulder'][1]) / 2)

        left = body_parts['RShoulder'][0]
        right = body_parts['LShoulder'][0]
    else:
        shoulder_height = int(body_parts['RShoulder'][1])

        left = right = body_parts['LShoulder'][0]


    for x in reversed(range(0, left)):
        if contour_img[shoulder_height, x] == 0:
            left = x
            break

    for x in range(right, len(contour_img[0])):
        if contour_img[shoulder_height, x] == 0:
            right = x
            break

    cv2.ellipse(contour_img, (left, shoulder_height), (8, 8), 0, 0, 360, 75, cv2.FILLED)
    cv2.ellipse(contour_img, (right, shoulder_height), (8, 8), 0, 0, 360, 75, cv2.FILLED)
    # display(contour_img, wait=True)

    return (right - left, shoulder_height)

def get_hip_width(contour_img, body_parts, orientation):

    if orientation == 'f':
        hip_height = int((body_parts['LHip'][1] + body_parts['RHip'][1]) / 2)
        left = body_parts['RHip'][0]
        right = body_parts['LHip'][0]

        for x in reversed(range(0, left)):
            if contour_img[hip_height, x] == 0:
                left = x
                break
            if x <= body_parts['RWrist'][0]:
                left = x#since it's the middle of the RWrist, maybe add another half of the wrist width to compensate?
                break

        for x in range(right, len(contour_img[0])):
            if contour_img[hip_height, x] == 0:
                right = x
                break
            if x >= body_parts['LWrist'][0]:
                right = x#since it's the middle of the RWrist, maybe add another half of the wrist width to compensate?
                break
    else:
        hip_height = int(body_parts['LHip'][1])
        left = right = body_parts['LHip'][0]

    cv2.ellipse(contour_img, (left, hip_height), (8, 8), 0, 0, 360, 191, cv2.FILLED)
    cv2.ellipse(contour_img, (right, hip_height), (8, 8), 0, 0, 360, 191, cv2.FILLED)

    # display(contour_img, wait=True)

    return (right - left, hip_height)

def get_knee_width(contour_img, body_parts, orientation):

    if orientation == 'f':
        knee_height = int((body_parts['LKnee'][1] + body_parts['RKnee'][1]) / 2)
        left = body_parts['RKnee'][0]
        right = body_parts['LKnee'][0]

        for x in reversed(range(0, left)):
            if contour_img[knee_height, x] == 0:
                left = x
                break

        for x in range(right, len(contour_img[0])):
            if contour_img[knee_height, x] == 0:
                right = x
                break

    else:
        knee_height = int(body_parts['LKnee'][1])
        left = right = body_parts['LKnee'][0]


    cv2.ellipse(contour_img, (left, knee_height), (8, 8), 0, 0, 360, 191, cv2.FILLED)
    cv2.ellipse(contour_img, (right, knee_height), (8, 8), 0, 0, 360, 191, cv2.FILLED)
    # display(contour_img, wait=True)

    return (right - left, knee_height)

def get_ankle_width(contour_img, body_parts, orientation):
    ankle_range = 20
    min_ankle_width = 0
    ml = mr = my = None

    if orientation == 'f':
        initial_ankle_height = int((body_parts['LAnkle'][1] + body_parts['RAnkle'][1]) / 2)
        left = body_parts['RAnkle'][0]
        right = body_parts['LAnkle'][0]

        for ankle_y in range(initial_ankle_height - ankle_range, initial_ankle_height + ankle_range):
            for x in reversed(range(0, left)):
                if contour_img[ankle_y, x] == 0:
                    left = x
                    break

            for x in range(right, len(contour_img[0])):
                if contour_img[ankle_y, x] == 0:
                    right = x
                    break

            if not min_ankle_width or min_ankle_width > right-left:
                min_ankle_width = right-left
                mr = right
                ml = left
                my = ankle_y
        cv2.ellipse(contour_img, (ml, my), (8, 8), 0, 0, 360, 60, cv2.FILLED)
        cv2.ellipse(contour_img, (mr, my), (8, 8), 0, 0, 360, 60, cv2.FILLED)

    else:
        ankle_y = int(body_parts['LAnkle'][1])
        left = right = body_parts['LAnkle'][0]

    # display(contour_img)

    return (min_ankle_width, ankle_y)




def display_together(imgs, title='img', wait=True):

    dst = np.zeros(imgs[0].shape, np.uint8)
    for img in imgs:
        dst = cv2.addWeighted(dst, 1, img, 1/len(imgs), 0)

    display(dst, title, wait)

    return dst

def display_sidebyside(imgs, title='img', wait=True):
    concat_img = np.concatenate(tuple(imgs), axis=1)
    display(concat_img, title, wait)

def display(img, title='img', wait=True):
    height = img.shape[0]
    width = img.shape[1]
    ratio = height/width

    WIDTH = 1800
    HEIGHT = int(WIDTH * ratio)
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, (WIDTH, HEIGHT))

    cv2.imshow(title, img)
    cv2.waitKey( 0 if wait else 1 )

def export_features(human_data, export_filename):

    for i in range(len(human_data)):
        del human_data[i]['initial_img']
        del human_data[i]['person_mask']
        del human_data[i]['contour']
        del human_data[i]['dots_img']
        del human_data[i]['pose_img']

    with open(export_filename, 'w') as outfile:
        yaml.dump(human_data, outfile, default_flow_style=False)


def prepare(img_names, export_filename=None):
    human_data = []

    #prepare get_person() model and save it as function attribute
    get_person.model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True).eval()

    #prepare rcnn model
    get_keypoints_rcnn.model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

    # img_names = ['./CW_data/training/022z077ps.jpg']
    for i in range(len(img_names)):
        img_name = img_names[i]
        print (f"{i+1}/{len(img_names)}", img_name)
        initial_img, person_mask = get_person(img_name)

        contour = get_contour(person_mask)

        #crop and only leave person in the picture - helps OpenPose models
        cropped_size = (1400,800)
        left = right = top = down = None
        for p in contour[0]:
            point = p[0]
            if not left or point[0] < left:
                left = point[0]
            elif not right or point[0] > right:
                right = point[0]

            if not down or point[1] > down:
                down = point[1]
            elif not top or point[1] < top:
                top = point[1]
        center_x = mean([left,right])

        initial_img = initial_img[down-cropped_size[0]:down, center_x - int(cropped_size[1]/2):center_x + int(cropped_size[1]/2)]
        person_mask = person_mask[down-cropped_size[0]:down, center_x - int(cropped_size[1]/2):center_x + int(cropped_size[1]/2)]
        # display_sidebyside([initial_img,cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR)],wait=True)

        pose_img, body_parts = get_keypoints_rcnn(cv2.bitwise_or(initial_img, initial_img, mask=person_mask))

        orientation = get_orientation(body_parts)
        mes_dots_img = person_mask.copy()
        contour = get_contour(person_mask)
        measurements = get_measurements(contour[0][0][0], mes_dots_img, body_parts, orientation)


        # display_sidebyside([initial_img, cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR), pose_img, cv2.cvtColor(mes_dots_img, cv2.COLOR_GRAY2BGR)], title='prep main display')
        data = {
            'name':img_name,
            'initial_img':initial_img,
            'person_mask':cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR),
            'pose_img':pose_img,
            'dots_img':cv2.cvtColor(mes_dots_img, cv2.COLOR_GRAY2BGR),
            'contour':contour,
            'orientation':orientation
        }
        data = {**data, **measurements}

        human_data.append(data)

    if export_filename:
        export_features(human_data, export_filename)

    return human_data


if __name__ == "__main__":
    prepare(get_training_image_names(), 'features_training.yaml')
