import argparse
import glob
import os

import numpy as np
import tensorflow as tf
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
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, vector):
    image_shape = tf.io.decode_jpeg(image_string).shape

    feature = {
        'shape': _bytes_feature(np.array(image_shape, np.int64).tobytes()),
        'vector': _bytes_feature(vector.numpy().tobytes()),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(args):
    fnames = glob.glob(os.path.join(args.path, '*.jpg'))
    idx_vecs = tf.random.normal((len(fnames), 256))

    record_file = 'ffhq-70k.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for i, filename in tqdm(enumerate(fnames)):
            image_string = open(filename, 'rb').read()
            tf_example = image_example(image_string, idx_vecs[i])
            writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    args = parse_args()
    main(args)
