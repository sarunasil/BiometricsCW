
import random
import yaml
import math

from time import sleep
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



def get_training_image_names(orientation = 's'):# 's' - side, 'f' - front view
    training_folder = "./CW_data/training"

    filenames = []
    for file in listdir(training_folder):
        # if orientation in file and 'person' not in file:
        if 'person' not in file:
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

    # fig, axs = pyplot.subplots(1,2, figsize=(50,100))
    # axs[1].imshow(r)

    # pyplot.show()


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


def get_pose(img):
    """Get person's body parts positions

    Reworked from https://github.com/legolas123/cv-tricks.com/blob/master/OpenCV/Pose_Estimation/run_pose.py

    Arguments:
        img {[type]} -- person imageget_person_height

    Returns:
        [type] -- [description]
    """
    proto = "pose/body_25/body_25_deploy.prototxt"
    model = "pose/body_25/pose_iter_584000.caffemodel"
    dataset = "BODY"

    # proto = "pose/coco/deploy_coco.prototxt"
    # model = "pose/coco/pose_iter_440000.caffemodel"
    # dataset = "COCO"

    # proto = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    # model = "pose/mpi/pose_iter_160000.caffemodel"
    # dataset = "MPI"

    threshold = 0.01
    width = 368
    height = 368

    if dataset == 'COCO':
        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

        POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    elif dataset=='MPI':
        BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                    "Background": 15 }

        POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                    ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    elif dataset == "BODY":
        BODY_PARTS ={"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,
                    "LElbow":6,"LWrist":7,"MidHip":8,"RHip":9,"RKnee":10,"RAnkle":11,"LHip":12,
                    "LKnee":13,"LAnkle":14,"REye":15,"LEye":16,"REar":17,"LEar":18,"LBigToe":19,
                    "LSmallToe":20,"LHeel":21,"RBigToe":22,"RSmallToe":23,"RHeel":24,"Background":25}

        POSE_PAIRS =[ ["Neck","MidHip"],   ["Neck","RShoulder"],   ["Neck","LShoulder"],
                    ["RShoulder","RElbow"],   ["RElbow","RWrist"],   ["LShoulder","LElbow"],
                    ["LElbow","LWrist"],   ["MidHip","RHip"],   ["RHip","RKnee"],  ["RKnee","RAnkle"],
                    ["MidHip","LHip"],  ["LHip","LKnee"], ["LKnee","LAnkle"],  ["Neck","Nose"],
                    ["Nose","REye"], ["REye","REar"],  ["Nose","LEye"], ["LEye","LEar"],
                    ["RShoulder","REar"],  ["LShoulder","LEar"],   ["LAnkle","LBigToe"],["LBigToe","LSmallToe"],
                    ["LAnkle","LHeel"], ["RAnkle","RBigToe"],["RBigToe","RSmallToe"],["RAnkle","RHeel"] ]

    inWidth = width
    inHeight = height

    net = cv2.dnn.readNetFromCaffe(proto, model)

    frameWidth = img.shape[1]
    frameHeight = img.shape[0]

    inp = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
    # blobb = inp.reshape(inp.shape[2] * inp.shape[1], inp.shape[3], 1)
    # cv2.imshow('inp', blobb)
    # cv2.waitKey(0)

    net.setInput(inp)
    out = net.forward()

    # print(inp.shape)
    assert(len(BODY_PARTS) <= out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)-1):
        # # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > threshold else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(img, points[idFrom], points[idTo], (255, 74, 0), 2)
            cv2.ellipse(img, points[idFrom], (2, 2), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(img, points[idTo], (2, 2), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.putText(img, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),1,cv2.LINE_AA)
            cv2.putText(img, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),3,cv2.LINE_AA)

    if dataset=="COCO":
        part_coord = {
            "Nose":points[BODY_PARTS['Nose']],
            "Neck":points[BODY_PARTS['Neck']],
            "RShoulder":points[BODY_PARTS['RShoulder']],
            "LShoulder":points[BODY_PARTS['LShoulder']],
            "RHip":points[BODY_PARTS['RHip']],
            "LHip":points[BODY_PARTS['LHip']],
            "RKnee":points[BODY_PARTS['RKnee']],
            "LKnee":points[BODY_PARTS['LKnee']],
            "REar":points[BODY_PARTS['REar']],
            "LEar":points[BODY_PARTS['LEar']],
            "RWrist":points[BODY_PARTS['RWrist']],
            "RElbow":points[BODY_PARTS["RElbow"]]
        }
    elif dataset=='MPI':
        part_coord = {
            "Head":points[BODY_PARTS['Head']],
            "Neck":points[BODY_PARTS['Neck']],
            "RShoulder":points[BODY_PARTS['RShoulder']],
            "LShoulder":points[BODY_PARTS['LShoulder']],
            "RHip":points[BODY_PARTS['RHip']],
            "LHip":points[BODY_PARTS['LHip']],
            "RKnee":points[BODY_PARTS['RKnee']],
            "LKnee":points[BODY_PARTS['LKnee']],
            "Chest":points[BODY_PARTS["Chest"]],
            # "REar":points[BODY_PARTS['REar']],
            # "LEar":points[BODY_PARTS['LEar']],
            "RWrist":points[BODY_PARTS['RWrist']],
            "RElbow":points[BODY_PARTS["RElbow"]]
        }
    elif dataset == "BODY":
        part_coord = {
            "Nose":points[BODY_PARTS['Nose']],
            "Neck":points[BODY_PARTS['Neck']],
            "RShoulder":points[BODY_PARTS['RShoulder']],
            "LShoulder":points[BODY_PARTS['LShoulder']],
            "RHip":points[BODY_PARTS['RHip']],
            "LHip":points[BODY_PARTS['LHip']],
            "RKnee":points[BODY_PARTS['RKnee']],
            "LKnee":points[BODY_PARTS['LKnee']],
            "REar":points[BODY_PARTS['REar']],
            "LEar":points[BODY_PARTS['LEar']],
            "RWrist":points[BODY_PARTS['RWrist']],
            "RElbow":points[BODY_PARTS["RElbow"]]
        }
    return img, part_coord


def get_keypoints_rcnn(cv_img):

    image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    # For reversing the operation:
    # im_np = np.asarray(im_pil)

    image_tensor = transforms.functional.to_tensor(image)

    output = get_keypoints_rcnn.model([image_tensor])
    # output is a list of dict, containing the postprocessed predictions

    parts = ['nose','leye','reye','lear','rear''lshoulder','rshoulder','lelbow','relbow','lwrist','rwrist','lhip','rhip','lknee','rknee','lankle','rankle','nan']
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

def get_measurements(contour_img, body_parts, orientation):

    height = contour_img.shape[0]
    shoulder_width = get_shoulder_width(contour_img, body_parts, orientation)
    neck_width = get_neck_width(contour_img, body_parts, orientation)
    hip_width = get_hip_width(contour_img, body_parts, orientation)
    knee_width = get_knee_width(contour_img, body_parts, orientation)
    head_width = get_head_width(contour_img, body_parts, orientation)

    return {
        "person_height": height,
        "shoulder_width": shoulder_width,
        "neck_width": neck_width,
        "knee_width": knee_width,
        "hip_width": hip_width,
        "head_width": head_width
    }



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

def get_neck_width(contour_img, body_parts, orientation):
    neck_range = 100

    min_neck_width = len(contour_img[1])

    if orientation == 'f' or orientation == 's':
        ml = mr = my = None

        initial_neck_y = body_parts['Neck'][1]
        for neck_y in range(initial_neck_y - neck_range, initial_neck_y):

            left = right = body_parts['Neck'][0]
            for x in reversed(range(0, left)):
                if contour_img[neck_y, x] == 0:
                    left = x
                    break

            for x in range(right, len(contour_img[0])):
                if contour_img[neck_y, x] == 0:
                    right = x
                    break

            if not min_neck_width or min_neck_width > right-left:
                min_neck_width = right-left
                mr = right
                ml = left
                my = neck_y
    else:
        pass
        # initial_neck = body_parts['Neck']
        # l_ear = body_parts['LEar']

        # adj_node = ( l_ear[0], initial_neck[1] )

        # hip = math.sqrt((initial_neck[0] - l_ear[0])**2 + (initial_neck[1] - l_ear[1])**2)
        # adj = math.sqrt((adj_node[0] - initial_neck[0])**2 + (adj_node[1] - initial_neck[1])**2)
        # neck_angle = math.acos( adj/hip )
        # print(neck_angle)

    cv2.ellipse(contour_img, (ml, my), (8, 8), 0, 0, 360, 127, cv2.FILLED)
    cv2.ellipse(contour_img, (mr, my), (8, 8), 0, 0, 360, 127, cv2.FILLED)
    #display(contour_img, wait=False)

    return (min_neck_width, my)

def get_hip_width(contour_img, body_parts, orientation):

    if orientation == 'f':
        hip_height = int((body_parts['LHip'][1] + body_parts['RHip'][1]) / 2)
        left = body_parts['RHip'][0]
        right = body_parts['LHip'][0]
    else:
        hip_height = int(body_parts['LHip'][1])
        left = right = body_parts['LHip'][0]


    for x in reversed(range(0, left)):
        if contour_img[hip_height, x] == 0:
            left = x
            break

    for x in range(right, len(contour_img[0])):
        if contour_img[hip_height, x] == 0:
            right = x
            break

    cv2.ellipse(contour_img, (left, hip_height), (8, 8), 0, 0, 360, 191, cv2.FILLED)
    cv2.ellipse(contour_img, (right, hip_height), (8, 8), 0, 0, 360, 191, cv2.FILLED)
    # display(contour_img, wait=True)

    return (right - left, hip_height)

def get_knee_width(contour_img, body_parts, orientation):

    if orientation == 'f':
        knee_height = int((body_parts['LKnee'][1] + body_parts['RKnee'][1]) / 2)
        left = body_parts['RKnee'][0]
        right = body_parts['LKnee'][0]
    else:
        knee_height = int(body_parts['LKnee'][1])
        left = right = body_parts['LKnee'][0]

    for x in reversed(range(0, left)):
        if contour_img[knee_height, x] == 0:
            left = x
            break

    for x in range(right, len(contour_img[0])):
        if contour_img[knee_height, x] == 0:
            right = x
            break

    cv2.ellipse(contour_img, (left, knee_height), (8, 8), 0, 0, 360, 191, cv2.FILLED)
    cv2.ellipse(contour_img, (right, knee_height), (8, 8), 0, 0, 360, 191, cv2.FILLED)
    # display(contour_img, wait=True)

    return (right - left, knee_height)

def get_head_width(contour_img, body_parts, orientation):
    head_range = 20

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


# ------------------------
        rcnn_pose_img, rcnn_body_parts = get_keypoints_rcnn(initial_img.copy())
# ------------------------


        pose_img, body_parts = get_pose(cv2.bitwise_or(initial_img, initial_img, mask=person_mask))

        orientation = get_orientation(body_parts)
        mes_dots_img = person_mask.copy()
        measurements = get_measurements(mes_dots_img, body_parts, orientation)


        # display_sidebyside([initial_img, cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR), pose_img, cv2.cvtColor(mes_dots_img, cv2.COLOR_GRAY2BGR), rcnn_pose_img], wait=True)
        data = {
            'name':img_name,
            'initial_img':initial_img,
            'person_mask':cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR),
            'pose_img':pose_img,
            'dots_img':cv2.cvtColor(mes_dots_img, cv2.COLOR_GRAY2BGR),
            'rcnn_pose_img':rcnn_pose_img,
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
