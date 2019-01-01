# ImageCaptioningAndroid
Image captioning on Android


This project has 2 parts:
1) Convert a pretrained .ckpt model to .tflite model. This tflite model can be run on android devices.
2) Android application which uses .tflite file to perform image captioning on android device.

Steps:
1) For show and tell, get the base code from tensorflow repository: https://github.com/tensorflow/models/tree/master/research/im2txt

This does not contain pretrained weights.

2) Get pretrained weights from https://github.com/KranthiGV/Pretrained-Show-and-Tell-model
Download the model trained for 2M iterations.
Use tensorflow 1.0 in Python 2 to test that the downloaded weights are used and image captioning starts working on laptop.

3) Freeze the model to convert the weights loaded from .ckpt file to .pb file.
The required changes are in this commit: https://github.com/fringedroid/ImageCaptioningAndroid/commit/4c3444cb95045e6500c42bbb940567b8f174863c

When saving to .pb succeds, make shapes of input and output tensors fixed (no None in shapes).

4) Convert .pb file to .tflite.
This is done in model_generation/im2txt/im2txt/convert_to_tflite.py
in 
Test if the .tflite model performs as exptected by using tflite interpreter in Python.
This is done in model_generation/im2txt/im2txt/cherry_pick.py

5) After getting a working .tflite model, use it in android app.
In the andorid app, Captioner.java performs image captioning by using .tflite model.


The final app borrows components from 2 apps:
a) Google's tflite demo app: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/java/demo
b) Deepsemantic image captioning app: https://github.com/deepsemantic/Captioner

The MainActivity from (b) is used in (a).
Captioner.java contains the logic for image captioning.

