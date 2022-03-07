import argparse
import logging
import os

logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import mixed_precision, optimizers
from tensorflow.keras import utils as kutils

from diffaug import DiffAugment
from models import Discriminator, Generator
from operation import ProgressBar, get_dir, imgrid

policy = 'color'
mixed_precision.set_global_policy('mixed_float16')

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

lpips_model = None


def lpips(imgs_a, imgs_b):
    if lpips_model is None:
        init_lpips_model(args)
    return lpips_model([imgs_a, imgs_b])


def init_lpips_model(args):
    global lpips_model
    model_file = f'./lpips_lin_{args.lpips_net}.h5'
    lpips_model = tf.keras.models.load_model(model_file)


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--path',
                        type=str,
                        default='./datasets/ffhq',
                        help='path of resource dataset')
    parser.add_argument('--name', type=str, default='', help='experiment name')
    parser.add_argument('--lpips_net',
                        type=str,
                        default='vgg',
                        help='lpips backbone net')
    parser.add_argument('--iter',
                        type=int,
                        default=50000,
                        help='number of iterations')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='minibatch size')
    parser.add_argument('--im_size',
                        type=int,
                        default=256,
                        help='image resolution')
    parser.add_argument('--lr',
                        type=float,
                        default=0.00005,
                        help='learning rate')
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='resume from latest checkpoint')

    args = parser.parse_args()
    return args


def main(args):
    data_root = args.path
    total_iterations = args.iter
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = args.lr
    nbeta1 = 0.5
    current_iteration = 0
    save_interval = 10
    saved_model_folder, saved_image_folder = get_dir(args)
    AUTOTUNE = tf.data.AUTOTUNE
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    print(f"Number of GPUs in use: {num_gpus}")

    ## Dataset
    def decode_fn(record_bytes):
        return tf.io.parse_single_example(
            # Data
            record_bytes,

            # Schema
            {
                'shape': tf.io.FixedLenFeature([], tf.string),
                'vector': tf.io.FixedLenFeature([], tf.string),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
            })

    def map_fn(record_bytes):
        parsed_example = decode_fn(record_bytes)
        vector = tf.reshape(
            tf.io.decode_raw(parsed_example['vector'], tf.float32), (256, ))
        image = tf.image.decode_jpeg(parsed_example['image_raw'], channels=3)
        image = tf.image.resize(image, (int(im_size), int(im_size)))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.convert_image_dtype(image, tf.float32) / 127.5 - 1
        return image, vector

    ds = tf.data.TFRecordDataset('./pokemon.tfrecords')
    ds = ds.map(map_fn,
                num_parallel_calls=AUTOTUNE).repeat().shuffle(buffer_size=256,
                                                              seed=42)
    ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    itds = iter(ds)

    img, vec = next(itds)
    tf.keras.utils.save_img('./tmp/img.jpg', tf.squeeze(img, 0))
    print(tf.reduce_sum(vec))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
