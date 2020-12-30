#!/usr/bin/env python
from __future__ import print_function

import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

paris = {
    'kinetics': ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                 (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
}


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transform_stacked_samples_joint_to_bone(data):
    N, C, T, V, M = data.shape
    data_transformed = np.zeros((N, 3, T, V, M))

    data_transformed[:, :C, :, :, :] = data
    for v1, v2 in list(paris['kinetics']):
        data_transformed[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

    return data_transformed

def pad_null_frames(data):
    data_expanded = np.expand_dims(data, axis=0)

    N, C, T, V, M = data_expanded.shape
    s = np.transpose(data_expanded, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    #pad the null frames with the previous frames
    for i_s, skeleton in enumerate(list(s)):
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    data_expanded = np.transpose(s, [0, 4, 2, 3, 1])
    data_orig_shape = np.squeeze(data_expanded, axis=0)
    return data_orig_shape

def extend_array_to_300_frames(data_numpy):
    data_numpy_zeros_extended = np.zeros((3, 300, 18, 2))
    data_numpy_zeros_extended[:, 0:data_numpy.shape[1], :, :] = data_numpy

    return data_numpy_zeros_extended


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self):
        init_seed(0)
        self.load_model()

    def load_model(self):
        output_device = 0
        self.output_device = output_device
        Model = import_class('model.aagcn.Model') # model with attention
        Model2 = import_class('model.agcn.Model') # model without attention

        model_args = {
            'num_class': 60,
            'num_person': 2,
            'num_point': 18,
            'graph': 'graph.kinetics.Graph',
            'graph_args': {'labeling_mode': 'spatial'}    
        }

        self.model = Model(**model_args).cuda(output_device)
        self.model2 = Model2(**model_args).cuda(output_device)
        self.model3 = Model(**model_args).cuda(output_device)
        self.model4 = Model2(**model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        # joints with attention (xview60) padded with frames
        weights_path = './runs/run_from_9_06_xview_60_classes_joints_padded_data_acc90proc/ki_aagcn_joint-49-29600.pt'
        # bones without attention (xview60) padded with frames
        weights2_path = './runs/run_from_9_06_xview_60_classes_bones_padded_data_acc89proc/ki_agcn_bone-49-29600.pt'
        # joints motion with attention (xview60) padded with zeros
        weights3_path = './runs/run_from_2_06_xview_60_classes_attention_acc90proc/ki_aagcn_joint-49-29600.pt'
        # bones motion without attention (xview60) padded with zeros
        weights4_path = './runs/run_from_3_06_xview_60_classes_bones_acc89proc/ki_agcn_bone-49-29600.pt'

        weights = torch.load(weights_path)
        weights = OrderedDict(
            [[k.split('module.')[-1],
                v.cuda(output_device)] for k, v in weights.items()])
        self.model.load_state_dict(weights)

        weights2 = torch.load(weights2_path)
        weights2 = OrderedDict(
            [[k.split('module.')[-1],
                v.cuda(output_device)] for k, v in weights2.items()])
        self.model2.load_state_dict(weights2)

        weights3 = torch.load(weights3_path)
        weights3 = OrderedDict(
            [[k.split('module.')[-1],
                v.cuda(output_device)] for k, v in weights3.items()])
        self.model3.load_state_dict(weights3)

        weights4 = torch.load(weights4_path)
        weights4 = OrderedDict(
            [[k.split('module.')[-1],
                v.cuda(output_device)] for k, v in weights4.items()])
        self.model4.load_state_dict(weights4)
        
    
    def predict_only(self, data_numpy):
        with open('./data/action_recognition/classes_pure_labels.txt', 'r') as f:
            text_labels = f.read().splitlines()

        self.model.eval()
        self.model2.eval()
        self.model3.eval()
        self.model4.eval()

        with torch.no_grad():
            network_input = data_numpy

            window_step = 30
            window_sizes = list(range(60, 121, 30))

            number_of_frames = network_input.shape[1]

            stacked_network_input_joint_frame_padded = np.zeros((0, 3, 300, 18, 2))
            stacked_network_input_joint_zero_padded = np.zeros((0, 3, 300, 18, 2))
            
            window_info = []

            for window_size in window_sizes:

                counter = 0
                while True:
                    start_index = counter * window_step
                    end_index = counter * window_step + window_size

                    if end_index > number_of_frames:
                        break
                    
                    window_info.append("[{} frames, {}-{}]".format(window_size,start_index,end_index))

                    network_partial_input = np.copy(network_input[:,start_index:end_index,:,:])
                    
                    network_partial_input = extend_array_to_300_frames(network_partial_input)
                    network_partial_input_zero_padded = np.copy(network_partial_input)
                    
                    network_partial_input = pad_null_frames(network_partial_input)
                    network_partial_input = np.expand_dims(network_partial_input, 0)
                    stacked_network_input_joint_frame_padded = np.append(stacked_network_input_joint_frame_padded, network_partial_input, 0)
                    
                    network_partial_input_zero_padded = np.expand_dims(network_partial_input_zero_padded, 0)
                    stacked_network_input_joint_zero_padded = np.append(stacked_network_input_joint_zero_padded, network_partial_input_zero_padded, 0)

                    counter += 1

            if stacked_network_input_joint_frame_padded.shape[0] == 0:
                return

            stacked_network_input_bone_frame_padded = transform_stacked_samples_joint_to_bone(stacked_network_input_joint_frame_padded)
            stacked_network_input_bone_zero_padded = transform_stacked_samples_joint_to_bone(stacked_network_input_joint_zero_padded)


            data = torch.tensor(stacked_network_input_joint_frame_padded)
            data = Variable(
                data.float().cuda(self.output_device), requires_grad=False)
            output1 = self.model(data)
            if isinstance(output1, tuple):
                output1, l1 = output1
                l1 = l1.mean()
            else:
                l1 = 0


            data2 = torch.tensor(stacked_network_input_bone_frame_padded)
            data2 = Variable(
                data2.float().cuda(self.output_device), requires_grad=False)
            output2 = self.model2(data2)
            if isinstance(output2, tuple):
                output2, l1 = output2
                l1 = l1.mean()
            else:
                l1 = 0


            data3 = torch.tensor(stacked_network_input_joint_zero_padded)
            data3 = Variable(
                data3.float().cuda(self.output_device), requires_grad=False)
            output3 = self.model3(data3)
            if isinstance(output3, tuple):
                output3, l1 = output3
                l1 = l1.mean()
            else:
                l1 = 0


            data4 = torch.tensor(stacked_network_input_bone_zero_padded)
            data4 = Variable(
                data4.float().cuda(self.output_device), requires_grad=False)
            output4 = self.model4(data4)
            if isinstance(output4, tuple):
                output4, l1 = output4
                l1 = l1.mean()
            else:
                l1 = 0                


            output = (1 * output1 + 0.8 * output2 + 0.5 * output3 + 0.5 * output4) / 4
            
            for j in range(output.size(0)):
                predict_labels = torch.argsort(output.data, descending=True)[j][:5]

                for i, label in enumerate(predict_labels):
                    label_item = label.item()
                    predict_text_label = text_labels[label_item]
            
            output_sum = output.sum(axis=0) / output.size(0)
            predict_labels = torch.argsort(output_sum.data, descending=True)[:5]

            prediction_strings = []

            for i, label in enumerate(predict_labels):
                label_item = label.item()
                predict_text_label = text_labels[label_item]
                prediction_strings.append("{} [{:>7.4f}] {} (A{})".format(i + 1, output_sum[label], predict_text_label, str(label_item + 1).zfill(3)))

            full_string = "\n".join(prediction_strings)

            # return full_string
            return output_sum

    def start(self, data_numpy):
        return self.predict_only(data_numpy)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
