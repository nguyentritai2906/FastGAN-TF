import argparse
import os

import tensorflow as tf
from tensorflow.keras import utils as kutils
from tqdm import tqdm

from models import Generator, NoiseInjection

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_model', type=str, help='path to h5 file')
    parser.add_argument('--save_dest',
                        type=str,
                        default='eval_results',
                        help='path to save images')
    parser.add_argument('--n_sample',
                        type=int,
                        default=8,
                        help='number of images to generate')
    parser.add_argument('--im_size',
                        type=int,
                        default=1024,
                        help='image size of trained model')
    args = parser.parse_args()

    modelG = Generator(im_size=args.im_size)
    modelG.load_weights(args.save_model)

    os.makedirs(args.save_dest, exist_ok=True)

    for i in tqdm(range(args.n_sample)):
        noise = tf.random.normal((1, 256), 0, 1, seed=None)
        img = modelG(noise, training=True)[0]

        kutils.save_img(os.path.join(args.save_dest, '%d.png' % i),
                        tf.squeeze(img, 0))
