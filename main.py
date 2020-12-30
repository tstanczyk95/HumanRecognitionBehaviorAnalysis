from __future__ import division

import dlib
import cv2
import os
import numpy as np
import copy
import sys
import argparse
import torch
from torch.autograd import Variable
from enum import Enum
from datetime import datetime

from pytorch_yolo_adapted.models import *
from pytorch_yolo_adapted.utils.utils import *
from pytorch_yolo_adapted.utils.datasets import *

from lightweight_openpose_adapted.demo import initialize_open_pose
from lightweight_openpose_adapted.demo import  get_network_input_from_video

from facial_recognition_pytorch_adapted.predict_emotion import initialize_emotion_recognition_net
from facial_recognition_pytorch_adapted.predict_emotion import predict_emotion_from_face_chip

from action_recognition_network import Processor


class RegisteringStatus(Enum):
    idle = 0
    active = 1
    just_finished = 2
    aborted = 3 # e.g. when tracked_person leaves the scene

reg_status = RegisteringStatus.idle
REGISTERED_SEQUENCE_FRAMES_LENGTH = 120
NUMBER_OF_SEQUENCES = 3
EMOTION_SHOT_FRAME_FREQUENCY = 30
SIMILARITY_THRESHOLD = 0.55
MIN_TRACKED_PEOPLE_NUMBER = 2
TEXT_VERTICAL_SHIFT = 20
DETECTION_SKIP_FRAMES = 15

with open('./data/action_recognition/classes_pure_labels.txt', 'r') as f:
    text_labels = f.read().splitlines()

processor = Processor()

def match_face_boxes_to_person_boxes(person_cords, face_data):
    person_pool = copy.copy(person_cords)
    face_pool = copy.copy(face_data)
    
    # List which will contain person and their face (possibly 1:1, 1:0, 0:1)
    person_face_list = []
    
    while True:
        # For each face box, find all the person boxes containing it
        persons_per_face = []
        for fd in face_pool:
            potential_persons = []
            
            for pc in person_pool:
                # Check if given person bounding box includes given face bounding box (left, top, right, bottom)
                if pc[0] <= fd[1][0] and \
                    pc[1] <= fd[1][1] and \
                    pc[2] >= fd[1][2] and \
                    pc[3] >= fd[1][3]: 
                        potential_persons.append(pc)
                        
            persons_per_face.append((fd, potential_persons))
            
        # Remove possible faces and persons from the pools 
        # and add them to the main (return) list
        zero_person_face_counter = 0
        one_person_face_counter = 0
        
        for ppf in persons_per_face:
            single_face_data, potential_persons = ppf
            
            if len(potential_persons) == 0:
                person_face_list.append((None, single_face_data[1], "NON-TRACKABLE"))
                face_pool.remove(single_face_data)
                zero_person_face_counter += 1
                
            elif len(potential_persons) == 1:
                person_face_list.append((potential_persons[0], single_face_data[1],
                                        single_face_data[0]))
                face_pool.remove(single_face_data)
                if potential_persons[0] in person_pool: 
                    person_pool.remove(potential_persons[0])
                one_person_face_counter += 1
                
        if zero_person_face_counter == 0 and \
            one_person_face_counter == 0:
                for pc in person_pool:
                    related_to_some_person = False
                    
                    for ppf in persons_per_face:
                        if pc in ppf[1]:
                            person_face_list.append((pc, None, "AMBIGUOUS"))
                            related_to_some_person = True
                            break
                            # No need to remove this pc from the pool, as the main 
                            # while loop is about to stop after this big outer if with counters
                    if not related_to_some_person:
                        person_face_list.append((pc, None, "NO MATCH"))
                        # As above with unnecessary removing
                break   
            
        if len(face_pool) == 0:
            for pc in person_pool:
                person_face_list.append((pc, None, "NO MATCH"))
                # As above with unnecessary removing
            break
    
    return person_face_list


def check_if_keypoints_within_bounding_box(keypoints, bounding_box, im_width, im_height):
    left, top, right, bottom = bounding_box

    left_normalized = float(left / im_width)
    right_normalized = float(right / im_width)
    top_normalized = float(top / im_height)
    bottom_normalized = float(bottom / im_height)



    all_zeros = True
    
    for i in [0, 1, 2, 3, 5, 6]: # selected upper keypoints only
        if keypoints[2, i] == 0:
            continue

        all_zeros = False
        
        keypoint_x = keypoints[0, i]
        keypoint_y = keypoints[1, i]
        
        print(left_normalized, keypoint_x, right_normalized)
        print(top_normalized, keypoint_y, bottom_normalized)


        if not(keypoint_x >= left_normalized
            and keypoint_x <= right_normalized
            and keypoint_y >= top_normalized
            and keypoint_y <= bottom_normalized):
            return False

    if all_zeros:
        return False

    return True
            

def transform_single_sample_to_motion(data):
    C, T, V, M = data.shape
    data_transformed = np.zeros((3, T, V, M))

    for t in list(range(T - 1)):
        data_transformed[:, t, :, :] = np.abs(data[:, t + 1, :, :] - data[:, t, :, :])
    data_transformed[:, T - 1, :, :] = 0

    return data_transformed


def compute_bounding_boxes_difference(bb1, bb2):
    left1, top1, right1, bottom1 = bb1
    left2, top2, right2, bottom2 = bb2

    return np.abs(left1 - left2) + np.abs(top1 - top2) + np.abs(right1 - right2) + np.abs(bottom1 - bottom2)


def get_tracked_person_max_counter_label(tracked_person):
    if tracked_person is None:
        raise Exception('tracked_person set to None, but attempted to reach its max counter label!')

    # if tracked_person is not None, then it always has at least one person with count of at least 1.
    max_counter = 0
    max_counter_label = None
    for label, counter in tracked_person["label"].items():
        if counter > max_counter:
            max_counter = counter
            max_counter_label = label

    return max_counter_label


def check_if_subject_left_scene(bounding_box, frame_width):
    left, top, right, bottom = bounding_box

    if (left < 0 and not(right > frame_width)) or (not(left < 0) and right > frame_width):
        return True

    return False


# Load and store all face_descriptors with their labels
face_descriptors = []

face_descriptor_path = "./data/face_recognition/face_descriptors/"
for file in os.listdir(face_descriptor_path):
    #print(file)
    if not file.endswith(".txt"):
        continue
    current_file = os.path.join(face_descriptor_path, file)
    with open(current_file, "r") as cf:
        content = cf.readlines()
        
    label = content[0][:-1] #[:-1] so as to exclude the '\n' char
    print("Loading descriptor for {}...".format(label)) 
    content = [float(c.strip()) for c in content[1:]]

    # (Try to) remove the bracket part from label
    bracket_index = label.find('(')
    if bracket_index != -1:
        label = label[:bracket_index].strip()
    
    face_descriptors.append((label, np.array(content)))
        
predictor_path = "./data/face_recognition/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "./data/face_recognition/dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.cnn_face_detection_model_v1("./data/face_recognition/mmod_human_face_detector.dat")
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


# Set up YOLO object detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet("pytorch_yolo_adapted/config/yolov3.cfg", img_size=416).to(device) # Set up model
model.load_darknet_weights("pytorch_yolo_adapted/weights/yolov3.weights") # Load darknet weights
model.eval()  # Set in evaluation mode
classes = load_classes("pytorch_yolo_adapted/data/coco.names")  # Extracts class labels from file
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("/path/to/video") # alternatively
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
FRAME_CENTER_X = frame_width / 2
FRAME_CENTER_Y = frame_height / 2


# Prepare date and time string
now = datetime.now()
year = str(now.year)
month = str(now.month)
day = str(now.day)
hour = str(now.hour)
minute = str(now.minute)
second = str(now.second)

datetime_string = year + "_" + month + "_" + day + "_" + hour + "_" + minute + "_" + second

out = cv2.VideoWriter(
        "output_processed_video_{}.avi".format(datetime_string), cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
        (frame_width, frame_height))

tracker_label_list = []

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
# Reference to the (single) person being tracked in form of a dictionary containing that person's possible labels and his/her #
# bounding box: { "label": {person1: counter1, person2: counter2, ...}, "bb": (left, top, right, bottom) }. Initially None.   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
tracked_person = None
already_tracked_labels = []

num_person_in = 5
num_person_out = 2

# Initialize sequence registered for action recognition
data_numpy = np.empty((3, 0, 18, num_person_in))
openpose_net = initialize_open_pose()


class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_recognition_net = initialize_emotion_recognition_net()

# Main processing loop (including camera frame reading)
frame_no = 0
registered_frames = 0
sequences_with_predicted_actions = 0
action_output_sum = None
emotion_output_sum = None
emotion_output_sum_list = []

file_short = open("activity_logs_{}.txt".format(datetime_string), "w")

start_time_pipeline_video = time.time()

while True:
    ret, frame_read = cap.read()
        
    if not ret:
        break
        
    frame_no += 1    
    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
    
    if reg_status == RegisteringStatus.active:
        #print(">>> RECORDING ACTION <<<")
        frame_data = get_network_input_from_video(openpose_net, frame_rgb)
        data_numpy = np.append(data_numpy, frame_data, 1)

        registered_frames += 1

        if registered_frames % EMOTION_SHOT_FRAME_FREQUENCY == 0:
            (left, top, right, bottom) = tracked_person["bb"]
            frame_person_cropped = frame_rgb[int(top):int(bottom), int(left):int(right), :]
            
            try:
                face_emo_dets = detector(frame_person_cropped, 1)
                if len(face_emo_dets) == 0:
                    print("Could not find face for emotion recognition")
                else:
                    # Get the landmarks/parts for the first detected face 
                    # (since it is bb-cropped image, at most one face is expected)
                    shape = sp(frame_rgb, face_emo_dets[0].rect)       
                    face_emo_chip = dlib.get_face_chip(frame_rgb, shape)
                    face_emotion_outputs = predict_emotion_from_face_chip(emotion_recognition_net, face_emo_chip)

                    if emotion_output_sum is None:
                        emotion_output_sum = face_emotion_outputs
                    else:
                        emotion_output_sum += face_emotion_outputs

            except RuntimeError as e:
                print("Could not detect face due to detector failure\n(most likely a problem from CUDA's side)")

        if registered_frames % REGISTERED_SEQUENCE_FRAMES_LENGTH == 0:
            reg_status = RegisteringStatus.just_finished

    elif reg_status == RegisteringStatus.just_finished:

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[1:2] = -data_numpy[1:2]
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                        0))
        data_numpy = data_numpy[:, :, :, 0:num_person_out]

        result = processor.start(data_numpy)

        if action_output_sum is None:
            action_output_sum = result
        else:
            action_output_sum += result

        print("\nSequence {} outcome:".format(registered_frames // REGISTERED_SEQUENCE_FRAMES_LENGTH))
        file_short.write("Sequence {} outcome:\n".format(registered_frames // REGISTERED_SEQUENCE_FRAMES_LENGTH))
        predict_labels = torch.argsort(result.data, descending=True)[:5]
        for i, label in enumerate(predict_labels):
            label_item = label.item()
            predict_text_label = text_labels[label_item]
            print("{} [{:>7.4f}] {} (A{})".format(i + 1, result[label], predict_text_label, str(label_item + 1).zfill(3)))
            file_short.write("{} [{:>7.4f}] {} (A{})\n".format(i + 1, result[label], predict_text_label, str(label_item + 1).zfill(3)))

        sequences_with_predicted_actions += 1
        data_numpy = np.empty((3, 0, 18, num_person_in))

        if emotion_output_sum is None:
            print("[Could not register facial emotion expression]")
        else:
            _, predicted = torch.max(emotion_output_sum.data, 0)
            print("The facial expression emotion is %s" %str(class_names[int(predicted.cpu().numpy())]))
            file_short.write("Facial expresion: {}\n\n".format(class_names[int(predicted.cpu().numpy())]))
            print(emotion_output_sum)
            emotion_output_sum_list.append(emotion_output_sum)
            emotion_output_sum = None

        if registered_frames % (NUMBER_OF_SEQUENCES * REGISTERED_SEQUENCE_FRAMES_LENGTH) == 0:
            reg_status = RegisteringStatus.idle
            registered_frames = 0

            tracked_person_label = get_tracked_person_max_counter_label(tracked_person)
            already_tracked_labels.append(tracked_person_label)

            tracked_person = None

            action_output_sum /= sequences_with_predicted_actions
            print("The average of {} outcome(s):".format(sequences_with_predicted_actions))
            file_short.write("\nThe average of {} outcome(s):\n".format(sequences_with_predicted_actions))
            predict_labels = torch.argsort(action_output_sum.data, descending=True)[:5]
            for i, label in enumerate(predict_labels):
                label_item = label.item()
                predict_text_label = text_labels[label_item]
                print("{} [{:>7.4f}] {} (A{})".format(i + 1, action_output_sum[label], predict_text_label, str(label_item + 1).zfill(3)))
                file_short.write("{} [{:>7.4f}] {} (A{})\n".format(i + 1, action_output_sum[label], predict_text_label, str(label_item + 1).zfill(3)))

            sequences_with_predicted_actions = 0
            action_output_sum = None


            if len(emotion_output_sum_list) == 0:
                print("[Could not register facial emotion expression for any action sequence]")
            else:
                emotion_output_sequence_sum = None
                for eo in emotion_output_sum_list:
                    if emotion_output_sequence_sum is None:
                        emotion_output_sequence_sum = eo
                    else:
                        emotion_output_sequence_sum += eo
                
                _, predicted = torch.max(emotion_output_sequence_sum.data, 0)
                print("The average facial expression emotion is %s" %str(class_names[int(predicted.cpu().numpy())]))
                file_short.write("Average facial expression: {}\n".format(class_names[int(predicted.cpu().numpy())]))
                print(emotion_output_sequence_sum)
                emotion_output_sum_list.clear()

                file_short.write("####################\n\n")

        else:
            reg_status = RegisteringStatus.active

    elif reg_status == RegisteringStatus.aborted:
        reg_status = RegisteringStatus.idle
        registered_frames = 0

        # print("Sequence {} registering aborted (subject left scene)".format(sequences_with_predicted_actions + 1))

        if action_output_sum is None:
            print("No actions registered")
        else:
            action_output_sum /= sequences_with_predicted_actions
            print("The average of {} outcome(s):".format(sequences_with_predicted_actions))
            file_short.write("\nThe average of {} outcome(s):\n".format(sequences_with_predicted_actions))
            predict_labels = torch.argsort(action_output_sum.data, descending=True)[:5]
            for i, label in enumerate(predict_labels):
                label_item = label.item()
                predict_text_label = text_labels[label_item]
                print("{} [{:>7.4f}] {} (A{})".format(i + 1, action_output_sum[label], predict_text_label, str(label_item + 1).zfill(3)))
                file_short.write("{} [{:>7.4f}] {} (A{})\n".format(i + 1, action_output_sum[label], predict_text_label, str(label_item + 1).zfill(3)))

            action_output_sum = None

            tracked_person_label = get_tracked_person_max_counter_label(tracked_person)
            already_tracked_labels.append(tracked_person_label)

        sequences_with_predicted_actions = 0
        tracked_person = None 


        if len(emotion_output_sum_list) == 0:
                print("[Could not register facial emotion expression for any action sequence]")
        else:
            emotion_output_sequence_sum = None
            for eo in emotion_output_sum_list:
                if emotion_output_sequence_sum is None:
                    emotion_output_sequence_sum = eo
                else:
                    emotion_output_sequence_sum += eo
            
            _, predicted = torch.max(emotion_output_sequence_sum.data, 0)
            print("The average facial expression emotion is %s" %str(class_names[int(predicted.cpu().numpy())]))
            file_short.write("Average facial expression: {}\n".format(class_names[int(predicted.cpu().numpy())]))
            print(emotion_output_sequence_sum)
            emotion_output_sum_list.clear()

            file_short.write("####################\n\n")

    matching_persons = None
    are_matching_persons_trackers = False

    if (frame_no - 1) % DETECTION_SKIP_FRAMES == 0:
        tracker_label_list = [] 

        img = transforms.ToTensor()(frame_read) # Extract image as PyTorch tensor
        img, _ = pad_to_square(img, 0) # Pad to square resolution
        img = resize(img, 416) # Resize

        input_img = img
        input_img.unsqueeze_(0)
        input_img = Variable(input_img.type(Tensor))

        with torch.no_grad():
            detections = model(input_img)
            detections = non_max_suppression(detections, 0.8, 0.4) #conf threshold, iou threshold

        detections = detections[0]

        person_cords = []
        if detections is not None:
            detections = rescale_boxes(detections, 416, frame_read.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if classes[int(cls_pred)].lower() == "person":
                    person_cords.append((x1, y1, x2, y2))
                
        print("Number of people detected: {}".format(len(person_cords)))
        
        dets = detector(frame_rgb, 1)
        print("Number of faces detected: {}".format(len(dets)))
        
        face_data = []
        for k, d in enumerate(dets):
            
            # Get the landmarks/parts for the face in box d.
            shape = sp(frame_rgb, d.rect)       
                            
            face_chip = dlib.get_face_chip(frame_rgb, shape)
            
            # Now we simply pass this chip (aligned image) to the api
            face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)                
            
            
            # Find matching person by calculating Euclidean distances and picking the person with the smallest one
            # (and less than SIMILARITY_THRESHOLD)
            fdfpi_array = np.array(face_descriptor_from_prealigned_image)
            min_distance = SIMILARITY_THRESHOLD
            min_distance_label = "UNKNOWN"
            for fd in face_descriptors:
                current_distance = np.linalg.norm(fd[1] - fdfpi_array)
                
                if current_distance < min_distance:
                    min_distance = current_distance
                    min_distance_label = fd[0]
            
            face_data.append((min_distance_label, (d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())))
        
        frame_marked = frame_rgb.copy()
                
        for pc in person_cords:
            cv2.rectangle(frame_marked, (pc[0], pc[1]), (pc[2], pc[3]), (255, 0, 255), 2)
        
        for fc in face_data:
            cv2.rectangle(frame_marked, (fc[1][0], fc[1][1]), (fc[1][2], fc[1][3]), (0, 255, 255), 2)
        
        face_per_person_list = match_face_boxes_to_person_boxes(person_cords, face_data)
             
        tracked_person_just_set = False
        bounding_box_min_distance = 1e4
        min_distance_bb = None
        
        for person_bb, face_bb, face_label in face_per_person_list:
            if face_label == "NON-TRACKABLE":
                continue
            tracker = dlib.correlation_tracker()
            tracker.start_track(frame_rgb, dlib.rectangle(person_bb[0], person_bb[1], person_bb[2], person_bb[3]))
            tracker_label_list.append((tracker, face_label))
            
            if tracked_person is None and face_label not in ["AMBIGUOUS", "NO MATCH", "UNKNOWN"]:
                
                if face_label not in already_tracked_labels:
                    tracked_person = {"label": {face_label: 1}, "bb": (person_bb[0], person_bb[1], person_bb[2], person_bb[3])}
                    tracked_person_just_set = True
                    reg_status = RegisteringStatus.active
                    file_short.write("####################\n")
                else:
                    print("{} has already been registered for action => NOT TRACKING THIS PERSON".format(face_label))

            if tracked_person is not None and not tracked_person_just_set:

                if face_label not in ["AMBIGUOUS", "NO MATCH", "UNKNOWN"]:
                    if face_label not in tracked_person["label"].keys():
                        tracked_person["label"][face_label] = 1
                    else:
                        tracked_person["label"][face_label] += 1


                bb_distance = compute_bounding_boxes_difference(tracked_person["bb"], (person_bb[0], person_bb[1], person_bb[2], person_bb[3]))
                if bb_distance < bounding_box_min_distance:
                    bounding_box_min_distance = bb_distance
                    min_distance_bb = (person_bb[0], person_bb[1], person_bb[2], person_bb[3])

        if tracked_person is not None and not tracked_person_just_set:
            # update face label of yolo (and related tracker), and bb of tracked_person
            if min_distance_bb is not None:
                tracked_person["bb"] = min_distance_bb

                update_index = -1
                old_face_label = None

                for i, (person_bb, face_bb, face_label) in enumerate(face_per_person_list):
                    if person_bb is not None and compute_bounding_boxes_difference(min_distance_bb, (person_bb[0], person_bb[1], person_bb[2], person_bb[3])) == 0:
                        update_index = i
                        break

                if update_index >= 0:
                    old_detection_face_label = face_per_person_list[update_index][2]

                    max_counter_label = get_tracked_person_max_counter_label(tracked_person)
                    new_entry = (face_per_person_list[update_index][0], face_per_person_list[update_index][1], max_counter_label) 
                    face_per_person_list[update_index] = new_entry

                update_index = -1

                for i, (tracker, face_label) in enumerate(tracker_label_list):
                    if old_detection_face_label == face_label:
                        update_index = i

                if update_index >= 0:

                    max_counter_label = get_tracked_person_max_counter_label(tracked_person)
                    new_entry = (tracker_label_list[update_index][0], max_counter_label)
                    tracker_label_list[update_index] = new_entry


        if reg_status == RegisteringStatus.just_finished:
            matching_persons = face_per_person_list
            
    else:
        for tracker, face_label in tracker_label_list:
        
            tracker.update(frame_rgb)
            
            pos = tracker.get_position()
        
            left = int(pos.left())
            top = int(pos.top())
            right = int(pos.right())
            bottom = int(pos.bottom())
            
            cv2.rectangle(frame_rgb, (left, top), (right, bottom), (255, 255, 0), 2)
            cv2.putText(frame_rgb, face_label,
                (left, top - TEXT_VERTICAL_SHIFT),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)

            if tracked_person is not None:
                max_counter_label = get_tracked_person_max_counter_label(tracked_person)

                if max_counter_label == face_label:
                    tracked_person["bb"] = (left, top, right, bottom)
     
        if reg_status == RegisteringStatus.just_finished:
            matching_persons = tracker_label_list
            are_matching_persons_trackers = True

    
    if reg_status == RegisteringStatus.just_finished:
        print("Action participant:")
        current_time = str(datetime.now())
        file_short.write("{}\n".format(current_time))
        file_short.write("Action participant: ")

        # Last frame keypoints
        first_person_keypoints = data_numpy[:, -1, :, 0]
        second_person_keypoints = data_numpy[:, -1, :, 1]

        if tracked_person is not None:
            is_first_person_participant = check_if_keypoints_within_bounding_box(first_person_keypoints, tracked_person["bb"], frame_rgb.shape[1], frame_rgb.shape[0])
            is_second_person_participant = check_if_keypoints_within_bounding_box(second_person_keypoints, tracked_person["bb"], frame_rgb.shape[1], frame_rgb.shape[0])

            if is_first_person_participant or is_second_person_participant:
                action_participant = get_tracked_person_max_counter_label(tracked_person)
                print(action_participant)
                file_short.write("{}\n".format(action_participant))
            else:
                print("[Action performed by non-tracked person]")
                file_short.write("NON-TRACKED PERSON\n")

        else:
            print("[No tracking person registered for action]")

    
    if tracked_person is not None and check_if_subject_left_scene(tracked_person["bb"], frame_width):
        reg_status = RegisteringStatus.aborted
        print("Sequence {} registering aborted (subject left scene)".format(sequences_with_predicted_actions + 1))


    if tracked_person is not None and reg_status == RegisteringStatus.active:
        max_counter_label = get_tracked_person_max_counter_label(tracked_person)
        if tracked_person["label"][max_counter_label] >= 5 and max_counter_label in already_tracked_labels:
            reg_status = RegisteringStatus.aborted
            print("Flickering person that is new (less) and that has already been tracked (more). -> NOT TRACKING")


    # Mark tracked person with magenta
    if tracked_person is not None:
        cv2.rectangle(frame_rgb, (tracked_person["bb"][0], tracked_person["bb"][1]), (tracked_person["bb"][2], tracked_person["bb"][3]), (255, 0, 255), 2)
        max_counter_label = get_tracked_person_max_counter_label(tracked_person)
        cv2.putText(frame_rgb, max_counter_label,
            (tracked_person["bb"][0], tracked_person["bb"][1] - TEXT_VERTICAL_SHIFT),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)

    if reg_status == RegisteringStatus.active:
        cv2.putText(frame_rgb, "REGISTERING SEQUENCE {}/{}".format(sequences_with_predicted_actions + 1, NUMBER_OF_SEQUENCES),
                (0, 0 + TEXT_VERTICAL_SHIFT), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
     
    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)    
    out.write(frame_rgb)
    cv2.imshow("video", frame_rgb)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print(tracked_person)
        break
    
    # Development and analysis purposes only
    if key == ord('r'):
        already_tracked_labels.clear()
        print("Cleared already tracked people")

end_time_pipeline_video = time.time()

time_pipeline_video = end_time_pipeline_video - start_time_pipeline_video
print("Processed frames: {}".format(frame_no))
print("Processing time: {}\n{} FPS".format(time_pipeline_video, frame_no / time_pipeline_video))


file_short.close()

out.release()
cap.release()
cv2.destroyAllWindows()