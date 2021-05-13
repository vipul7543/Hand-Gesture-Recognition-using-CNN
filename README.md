# Hand Gesture Recognition Using Background Elimination and Convolution Neural Network in Python

This is a simple application of Convolution Neural Networks combined with background ellimination to detect different hand gestures. A background elimination algorithm extracts the hand image from webcam and uses it to train as well predict the type of gesture that is. More information about the algorithm can be found below.

## Requirements

* Python3
* Tensorflow
* TfLearn
* Opencv (cv2) for python3
* Numpy

## File Description

* track.py : Run this file to generate custom datasets. Change the path and name in the file accordingly
* trainer.py: This is the model trainer file. Run this file if you want to retrain the model using your custom dataset edit the number of classes and number of layer accordingly.
* predict.py: Running this file opens up your webcam and takes continuous frames of your hand image and then predicts the class of your hand gesture in realtime.

* Background elimination algorithm is use to detect the hand (https://gogul.dev/software/hand-gesture-recognition-p1).
