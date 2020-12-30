# HumanRecognitionBehaviorAnalysis
An integrated computer vision system for human recognition and behavior analysis from RGB Camera. Project realized within the master studies internship.
<p align="center"><img src="extra_materials/readme_images/registered_sequence.png" width="600"\></p>
<p align="center"><img src="extra_materials/readme_images/registered_sequence_output.png" width="600"\></p>

## Components
This system was created using the state-of-the-art computer vision components based on the following repositories:
*   object detection - [PyTorch YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3),
*   face detection - [Dlib's CNN-based face detection model](http://dlib.net/python/index.html#dlib.cnn_face_detection_model_v1),
*   face recognition - [Dlib's face recognition model](http://dlib.net/python/index.html#dlib.face_recognition_model_v1),
*   object tracking - [Dlib's correlation tracker](http://dlib.net/python/index.html#dlib.correlation_tracker),
*   facial emotion recognition - [Facial-Expression-Recognition.Pytorch](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch),
*   human pose estimation - [lightweight-human-pose-estimation.pytorch](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch),
*   skeleton-based action recognition - [2s-AGCN (MS-AAGCN)](https://github.com/lshiwjx/2s-AGCN).

The skeleton-based action recognition component was modified and expanded into practically usable component. The whole pipeline extracting skeletons human pose skeletons from RGB video was created:
<p align="center"><img src="extra_materials/readme_images/act_rec_pipelinepng.png" width="600"\></p>

The skeleton-based action recognition component was trained on the custom 2D skeleton dataset. Using the same approach with [Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) as for the action recognition pipeline, 2D skeletons were extracted from RGB videos coming from the original [NTU-RGB+D dataset](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp). First 60 classes (A001-A060) were taken into account.<br/>
<!-- The derived datasets will be released soon upon the agreement from the authors of the original dataset -->
<br/>
All the components were integrated into one system as presented below:

<p align="center"><img src="extra_materials/readme_images/system_architecture.png" width="800"\></p>

For more details about the components and the system functionality, please check [internship presentation](extra_materials/presentation.pdf) and [abstract](extra_materials/abstract.pdf).


## Installation
An RGB camera is required.<br/>
The following packages need to be installed, with the newest versions recommended:
*   pytorch (torch)
*   torchvision
*   h5py
*   sklearn
*   pycocotools
*   opencv-python
*   numpy
*   matplotlib
*   terminaltables
*   pillow
*   tqdm
*   dlib (with GPU support)
*   argparse
*   PIL
*   imutils
<br/>
As described in [PyTorch YOLOv3 installation guidelines](https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/README.md#download-pretrained-weights),

##### download pretrained weights :
    $ cd pytorch_yolo_adapted/weights/
    $ bash download_weights.sh
    
##### and download COCO:
    $ cd pytorch_yolo_adapted/data/
    $ bash get_coco_dataset.sh
<br/>

Following [the guidlines of Facial-Expression-Recognition.Pytorch](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch#visualize-for-a-test-image-by-a-pre-trained-model), create a new folder: facial_recognition_pytorch_adapted/FER2013_VGG19/, download the pre-trained model from [link1](https://drive.google.com/open?id=1Oy_9YmpkSKX1Q8jkOhJbz3Mc7qjyISzU) or [link2 (key: g2d3)](https://pan.baidu.com/s/1gCL0TlCwKctAy_5yhzHy5Q)) and place it in the FER2013_VGG19/ folder.<br/>
Further, download Dlib's human face detector model from [here](http://dlib.net/cnn_face_detector.py.html) as well as Dlib's shape predictor from [here](http://dlib.net/face_recognition.py.html), unpack both files and place them inside facial_recognition_pytorch_adapted/
<br/><br/>

As described in [lightweight-human-pose-estimation.pytorch installation guidelines](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch#pre-trained-model-), download the pre-trained model from [here](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth) and place it in lightweight_openpose_adapted/models/.
<br/><br/>

Finally, download: Dlib's human face detector model from [here](http://dlib.net/cnn_face_detector.py.html) and Dlib's shape predictor as well as Dlib's recognition resnet model from [here](http://dlib.net/face_recognition.py.html), unpack all files and place them inside data/face_recognition/

## Generating more descriptors for facial recognition
In order to generate more facial descriptors, 

##### go to data/face_recognition/ and type the following:
    $ face_descriptor_generator.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat /path/to/the/input/image "name of the person"
    
Move the generated descriptor to face_descriptors/ . Note that the input image should contain photo of only one person. In case of multiple faces detected, only the first one will be processed and saved to the file.

## Running the program
Launch command prompt or terminal and go to the project main folder (HumanRecognitionBehaviorAnalysis/).

##### Type in the console:
    python main.py
    
The camera input window will appear and the processing will start.<br/>
In order to stop the processing, set the focus on the output video window (e.g. by clicking on it) and press 'Q' from the keyboard.<br/><br/>
The generated output files will include:
*   the processed video sequence (no sound), saved in the format: "output_processed_video_[year]_[month]_[day]_[hour]_[minute]_[second].avi",
*   the activity log text file (in case of successful and sufficient registration of the frames for action recognition) with potentially identified person, the performed action and corresponding facial emotion. The activity log file will be saved in the following format: "activity_logs_[year]_[month]_[day]_[hour]_[minute]_[second].txt".
