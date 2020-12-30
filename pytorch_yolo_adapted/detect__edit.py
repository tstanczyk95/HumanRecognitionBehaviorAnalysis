from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import argparse

import torch
from torch.autograd import Variable

import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    #print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    cap = cv2.VideoCapture(0) # 0 or path to the video :)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # These 3 lines: as in datasets.py/ImageFolder's __getitem__(...)
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(frame)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, 416)#self.img_size)

        input_img = img
        input_img.unsqueeze_(0)
        input_img = Variable(input_img.type(Tensor))

        with torch.no_grad():
            detections = model(input_img)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            print(detections)

        detections = detections[0]

        if detections is not None:
            detections = rescale_boxes(detections, 416, frame.shape[:2])#img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, classes[int(cls_pred)],
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)

        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()