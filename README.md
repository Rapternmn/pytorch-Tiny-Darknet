# pytorch-Tiny-Darknet
PyTorch implementation of the Tiny Darknet Image Classification algorithm 

<!-- Original Implementation of Tiny Darknet: https://pjreddie.com/darknet/tiny-darknet/ -->

This repository contains code for a classifier based on [Tiny Darknet](https://pjreddie.com/darknet/tiny-darknet/), implementedin PyTorch. The code is based on the official code of [YOLO v3](https://github.com/pjreddie/darknet), as well as a PyTorch 
port of the original code, by [ayooshkathuria](https://github.com/ayooshkathuria/pytorch-yolo-v3). The aim of this project is to have pytorch compatible Tiny Darknet classifier. 

As of now, the code only contains the Classification module.

## Requirements
1. Python 3.5
2. OpenCV
3. PyTorch 0.4

## Running the detector

### On single or multiple images

```
python detect.py --image imgs/Malamute.jpg 
```

Output would be expected to be something similar like 
```
Class = malamute  Confidence = 62.50366973876953
Class = Norwegian elkhound  Confidence = 17.33596420288086
Class = Eskimo dog  Confidence = 9.506247520446777
Class = Siberian husky  Confidence = 5.180300235748291
Class = keeshond  Confidence = 4.281615257263184
```

### Inference Time Benchmarks

The average inference time observed on my machine (GTX 1050) is ~ 3ms