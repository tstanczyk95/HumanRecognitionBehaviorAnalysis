import sys
import os
import dlib
import glob

import cv2
from array import array
import numpy as np



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Assumption: the input image contains photo of only one person!  #
# In case of multiple faces detected, only the first one will     #
# be processed and save to the file.                              #
#                                                                 #
# The detected face will be displayed while processing            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if len(sys.argv) != 5:
    print(
        "Call this program like this:\n"
        "   ./face_descriptor_generator.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat /path/to/the/input/image \"name of the person\"\n")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
image_path = sys.argv[3]
person_name = sys.argv[4]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
##detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
win = dlib.image_window()

# Now process the image
print("Processing file: {}".format(image_path))
img = dlib.load_rgb_image(image_path)

win.clear_overlay()
win.set_image(img)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

# Now process each face we found.
for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()))
    # Get the landmarks/parts for the face in box d.
    shape = sp(img, d.rect)
    
    # Draw the face landmarks on the screen so we can see what face is currently being processed.
    win.clear_overlay()
    win.add_overlay(d.rect)
    win.add_overlay(shape)
    dlib.hit_enter_to_continue()


    # Compute the 128D vector that describes the face in img identified by
    # shape.  In general, if two face descriptor vectors have a Euclidean
    # distance between them less than 0.6 then they are from the same
    # person, otherwise they are from different people. Here we just print
    # the vector to the screen.
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    print(face_descriptor)
    


    print("Computing descriptor on aligned image ..")
    
    # Let's generate the aligned image using get_face_chip
    face_chip = dlib.get_face_chip(img, shape)        
    
    win.set_image(face_chip)
    cv2.imwrite("face_chip_{}.jpg".format(person_name.replace(" ", "_").lower()), 
                cv2.cvtColor(face_chip, cv2.COLOR_BGR2RGB))
    
    
    # Now we simply pass this chip (aligned image) to the api
    face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)                
    print(face_descriptor_from_prealigned_image) 
    
    
    with open("face_descriptor_{}.txt".format(person_name.replace(" ", "_").lower()), "w") as f:
        f.write("{}\n".format(person_name))
        for el in face_descriptor_from_prealigned_image:
            f.write("{}\n".format(el))
            
    
    
    dlib.hit_enter_to_continue()
    
    break

