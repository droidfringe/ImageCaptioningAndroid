#!/usr/bin/env bash
# Steps:
# 1) Merge checkpoint provided by https://github.com/KranthiGV/Pretrained-Show-and-Tell-model with Google's show and tell code
# 2) Use Tensorflow 1.0 in python2 to run the merged code as described in bazel build and run steps below
# 3) Freeze the model and save it to .pb . My patch does that.
# 4) Modify dimensions and input for inference part. My patch does that.
# 5) Run the bazel build and bazel run to write new .pb file with these changed dimensions and input format.
# 6) Convert the newly generated .pb model to .tflite model. Use python3 and tf-nightly 1.13 for this. convert_to_tflite.py
# 7) Use cherry_pick.py in python3 to test the tflite model



CHECKPOINT_PATH="/home/droid/show_and_tell/im2txt/im2txt/data/model.ckpt-2000000"
VOCAB_FILE="/home/droid/show_and_tell/im2txt/im2txt/data/word_counts.txt"

# JPEG image file to caption.
IMAGE_FILE='/home/droid/show_and_tell/im2txt/im2txt/data/cat_dog.jpg.346.jpg'

# Build the inference binary.
bazel build -c opt im2txt/run_inference

# Run inference to generate captions.
bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}

# This will generate output_graph_4.pb in /home/droid/show_and_tell/im2txt/im2txt/data

# TENSORFLOW 1.13 nightly to convert to tflite. TF 1.12 does not work
# Use this graph to get a tflite model
# python3 convert_to_tflite.py

# This will generate converted_model.tflite in /home/droid/show_and_tell/im2txt/im2txt/data

# To do captioning on an image using standalone tflite model, use:
# python3 cherry_pick.py