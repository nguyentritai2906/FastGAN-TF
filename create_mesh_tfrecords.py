import argparse
import glob
import os

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as kutils
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--path',
                        type=str,
                        default='./datasets/ffhq/',
                        help='path of resource dataset')

    args = parser.parse_args()
    return args


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, landmarks):
    feature = {
        'image_raw': _bytes_feature(image_string),
        'landmarks': _float_feature(landmarks)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(args):
    mp_facemesh = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils

    fnames = glob.glob(os.path.join(args.path, '*.jpg'))
    assert len(fnames) == 70000

    record_file = 'ffhq-mesh.tfrecords'
    failed = 0
    failed_fnames = []

    with tf.io.TFRecordWriter(record_file) as writer:
        for filename in tqdm(fnames):
            landmarks = []
            image = cv2.imread(filename)

            with mp_facemesh.FaceMesh(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as facemesh:

                result = facemesh.process(image)
                if result.multi_face_landmarks:
                    for face_landmark in result.multi_face_landmarks:
                        for landmark in face_landmark.landmark:
                            landmarks.append(landmark.x)
                            landmarks.append(landmark.y)
                            landmarks.append(landmark.z)

            if len(landmarks) != 1404:
                failed += 1
                failed_fnames.append(filename)
                continue

            image = cv2.resize(image, (256, 256))

            success, encoded_image = cv2.imencode('.jpg', image)
            image_string = encoded_image.tobytes()
            tf_example = image_example(image_string, landmarks)
            writer.write(tf_example.SerializeToString())

    print('Number of failed images:', failed)
    print('Failed images:', failed_fnames)


if __name__ == "__main__":
    args = parse_args()
    main(args)
