import argparse
import os

import tensorflow as tf
from tensorflow.keras import utils as kutils
from tqdm import tqdm

from models import Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate images')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--save_dest', type=str, default='eval_results')
    parser.add_argument('--n_sample', type=int, default=8)
    parser.add_argument('--im_size', type=int, default=1024)
    args = parser.parse_args()

    noise_dim = 256

    modelG = Generator(ngf=64, im_size=args.im_size)
    modelG(tf.random.normal((1, noise_dim)))

    ## Load checkpoint
    checkpoint = tf.train.Checkpoint(modelG=modelG)
    manager = tf.train.CheckpointManager(checkpoint,
                                         args.checkpoint,
                                         max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    print('Load checkpoint from {}.'.format(manager.latest_checkpoint))
    del checkpoint, manager

    os.makedirs(args.save_dest, exist_ok=True)

    for i in tqdm(range(args.n_sample)):
        noise = tf.random.normal((1, noise_dim), 0, 1)
        img = modelG(noise)[0]

        kutils.save_img(os.path.join(args.save_dest, '%d.png' % i),
                        tf.squeeze(img, 0))
