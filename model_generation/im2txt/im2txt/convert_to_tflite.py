import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import numpy as np

def main():
    sess = tf.Session()
    GRAPH_LOCATION = '/home/droid/show_and_tell/im2txt/im2txt/data/output_graph_4.pb'
    VOCAB_FILE = '/home/droid/show_and_tell/im2txt/im2txt/data/word_counts.txt'
    IMAGE_FILE = '/home/droid/show_and_tell/im2txt/im2txt/data/cat_dog.jpg.346.jpg'

    # Read model
    with gfile.FastGFile(GRAPH_LOCATION, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def)

    with tf.gfile.GFile(IMAGE_FILE, "rb") as f:
        encoded_image = f.read()

    input_names_1 = ['import/image_feed:0']
    input_names_2 = ['import/input_feed:0', 'import/lstm/state_feed:0']
    output_names_1 = ['import/lstm/initial_state:0']
    output_names_2 = ['import/softmax:0', 'import/lstm/state:0']


    g = tf.get_default_graph()
    input_tensors_1 = [g.get_tensor_by_name(x) for x in input_names_1]
    input_tensors_2 = [g.get_tensor_by_name(x) for x in input_names_2]
    output_tensors_1 = [g.get_tensor_by_name(x) for x in output_names_1]
    output_tensors_2 = [g.get_tensor_by_name(x) for x in output_names_2]

    converter = tf.contrib.lite.TFLiteConverter.from_session(sess, input_tensors_1, output_tensors_1)
    model = converter.convert()
    fid = open("/home/droid/show_and_tell/im2txt/im2txt/data/converted_model_1.tflite", "wb")
    fid.write(model)
    fid.close()

    converter = tf.contrib.lite.TFLiteConverter.from_session(sess, input_tensors_2, output_tensors_2)
    model = converter.convert()
    fid = open("/home/droid/show_and_tell/im2txt/im2txt/data/converted_model_2.tflite", "wb")
    fid.write(model)
    fid.close()

    """
    for op in g.get_operations():
        print(op.values())
    """

if __name__ == '__main__':
    main()
