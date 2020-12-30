import argparse

import cv2
import numpy as np
import torch

from lightweight_openpose_adapted.models.with_mobilenet import PoseEstimationWithMobileNet
from lightweight_openpose_adapted.modules.keypoints import extract_keypoints, group_keypoints
from lightweight_openpose_adapted.modules.load_state import load_state
from lightweight_openpose_adapted.modules.pose import Pose, track_poses
from lightweight_openpose_adapted.val import normalize, pad_width

import time
import os


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, img, height_size, cpu, track, smooth):
    stride = 8
    upsample_ratio = 4
    num_person_in = 5
    num_keypoints = Pose.num_kpts

    frame_data = np.zeros((3, 1, 18, num_person_in))

    im_height, im_width, _ = img.shape

    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

    number_of_persons = pose_entries.shape[0]
    person_data = np.zeros((number_of_persons, 18, 3))

    for j in range(number_of_persons):

        if j >= 5:
            print("Skipping person nr {} counting from 0".format(j))
            continue

        person = pose_entries[j]

        # 18 keypoints
        for i in range(18):
            all_keypoint_id = int(person[i])
            if all_keypoint_id != -1:
                # Include normalizing to scale [0,1]
                frame_data[0][0][i][j] = round(float(all_keypoints[all_keypoint_id][0] / im_width), 3)
                frame_data[1][0][i][j] = round(float(all_keypoints[all_keypoint_id][1] / im_height), 3)
                frame_data[2][0][i][j] = round(float(all_keypoints[all_keypoint_id][2]), 3)

    return frame_data


def initialize_open_pose():
    load_start_time = time.time()
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('./lightweight_openpose_adapted/models/checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)
    load_end_time = time.time()
    print("Net and weights loading time: {}s".format(load_end_time - load_start_time))

    net_start_time = time.time()
    net = net.eval()
    net = net.cuda()
    net_end_time = time.time()
    print("Net eval and setup time: {}s".format(net_end_time - net_start_time))

    return net


def get_network_input_from_video(net, img):
    net_input = run_demo(net, img, 256, False, 1, 1)
    
    return net_input
