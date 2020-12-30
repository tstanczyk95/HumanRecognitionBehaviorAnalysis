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

import facial_recognition_pytorch_adapted.transforms as transforms
from skimage import io
from skimage.transform import resize
from facial_recognition_pytorch_adapted.models import *

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


def initialize_emotion_recognition_net():
    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('facial_recognition_pytorch_adapted', 'FER2013_VGG19', 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()

    return net

def predict_emotion_from_face_chip(net, face_chip):
    gray_face_chip = rgb2gray(face_chip)
    gray_face_chip = resize(gray_face_chip, (48,48), mode='symmetric').astype(np.uint8)
    
    img = gray_face_chip[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    return outputs_avg
