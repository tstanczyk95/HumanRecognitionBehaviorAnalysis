# HumanRecognitionBehaviorAnalysis
An integrated computer vision system for human recognition and behavior analysis from RGB Camera. Project realized within the master studies internship.
<p align="center"><img src="readme_images/registered_sequence.png" width="600"\></p>
<p align="center"><img src="readme_images/registered_sequence_output.png" width="600"\></p>

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
<p align="center"><img src="readme_images/act_rec_pipelinepng.png" width="600"\></p>

The skeleton-based action recognition component was trained on the custom 2D skeleton dataset. Using the same approach with [Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) as for the action recognition pipeline, 2D skeletons were extracted from RGB videos coming from the original [NTU-RGB+D dataset](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp). First 60 classes (A001-A060) were taken into account.<br/>
<!-- The derived datasets will be released soon upon the agreement from the authors of the original dataset -->
<br/>
All the components were integrated into one system as presented below:

<p align="center"><img src="readme_images/system_architecture.png" width="800"\></p>

For more details about the components and the system itself, please check [internship presentation](extra/presentation.pdf) and [abstract](extra/abstract.pdff).
