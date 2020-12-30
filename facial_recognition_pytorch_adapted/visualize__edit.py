"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

import dlib
from imutils import face_utils
import cv2

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


video_capture = cv2.VideoCapture(0)

# rect for frontal face detector and rect.rect for cnn face detector
# Actually, since dlib is build with CUDA support and GPU is used, frontal
# face detector runs slower than its cnn counterpart!
# face_detect = dlib.get_frontal_face_detector() 
face_detect = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

net = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

while True:

    ret, frame = video_capture.read()
    raw_img = frame
    # raw_img = io.imread('images/1.jpg')
    # raw_img = io.imread('/home/activreg/Tomasz_files/test_data/photos/Malakas.jpeg')
    # raw_img = io.imread('/home/activreg/Downloads/grief_cycle.jpeg')

    # Undesired for camera input, but required for io.imread images
    # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    rects = face_detect(raw_img, 1)
    for (i, rect) in enumerate(rects):   
        (face_x, face_y, face_w, face_h) = face_utils.rect_to_bb(rect.rect)

        shape = sp(raw_img, rect.rect)
        face_chip = dlib.get_face_chip(raw_img, shape) 
        #face_chip = raw_img[face_y:face_y+face_h,face_x:face_x+face_w] # Testing purposes only

        gray_face_chip = rgb2gray(face_chip)
        gray_face_chip = resize(gray_face_chip, (48,48), mode='symmetric').astype(np.uint8)
        
        img = gray_face_chip[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        #score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)

        #print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))

        cv2.putText(raw_img, (class_names[int(predicted.cpu().numpy())]),(face_x+face_w-10,face_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(raw_img, (face_x, face_y, face_w, face_h), (255, 0, 0), 2)

        cv2.imshow("Face chip", face_chip) # For testing purposes only! (A.k.a., to be ultimately commented out)
        
    cv2.imshow("Video", raw_img)    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

print("I don't like when core gets dumped!")
