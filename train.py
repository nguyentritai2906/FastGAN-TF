import argparse
import glob
import logging
import os
import random

import tensorflow_addons as tfa

logging.getLogger('tensorflow').disabled = True

import onnx2keras.elementwise_layers
import onnx2keras.operation_layers
import onnx2keras.utils
import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import mixed_precision, optimizers
from tensorflow.keras import utils as kutils

from diffaug import DiffAugment
from models import Discriminator, Generator
from operation import ProgressBar, crop_image_by_part, get_dir, imgrid

policy = 'color,translation'
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
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument(
        '--path',
        type=str,
        default='./datasets/pokemon/img',
        help=
        'path of resource dataset, should be a folder that has one or many sub image folders inside'
    )
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
                        default=8,
                        help='mini batch number of images')
    parser.add_argument('--im_size',
                        type=int,
                        default=1024,
                        help='image resolution')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0002,
                        help='learning rate')
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='continue training from latest checkpoint')
    parser.add_argument('--ckpt',
                        type=str,
                        default='None',
                        help='checkpoint weight path if have one')

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
    strategy = tf.distribute.MirroredStrategy(devices=None)
    num_gpus = len(tf.config.list_physical_devices('GPU'))

    ## Model
    if num_gpus > 1:
        print("Initializing distribute model")
        with strategy.scope():
            modelG = Generator(input_shape=(nz, ), ngf=ngf, im_size=im_size)
            modelD = Discriminator(ndf=ndf, im_size=im_size)
            optimizerG = mixed_precision.LossScaleOptimizer(
                optimizers.Adam(nlr / 3, nbeta1))
            optimizerD = mixed_precision.LossScaleOptimizer(
                optimizers.Adam(nlr, nbeta1))
            # optimizerG = optimizers.Adam(nlr/3, nbeta1, epsilon=1e-08)
            # optimizerD = optimizers.Adam(nlr, nbeta1, epsilon=1e-08)
    else:
        print("Initializing model")
        modelG = Generator(input_shape=(nz, ), ngf=ngf, im_size=im_size)
        modelD = Discriminator(ndf=ndf, im_size=im_size)
        # optimizerG = optimizers.Adam(nlr/3, nbeta1, epsilon=1e-08)
        # optimizerD = optimizers.Adam(nlr, nbeta1, epsilon=1e-08)
        optimizerG = mixed_precision.LossScaleOptimizer(
            optimizers.Adam(nlr / 3, nbeta1))
        optimizerD = mixed_precision.LossScaleOptimizer(
            optimizers.Adam(nlr, nbeta1))

    ## Dataset
    filenames = glob.glob(os.path.join(data_root, '*.jp*'))
    image_paths = tf.convert_to_tensor(filenames, dtype=tf.string)

    def map_fn(path):
        image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (int(im_size), int(im_size)))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.convert_image_dtype(image, tf.float32) / 127.5 - 1
        return image

    ds = tf.data.Dataset.from_tensor_slices(image_paths).cache()
    ds = ds.map(map_fn,
                num_parallel_calls=AUTOTUNE).repeat().shuffle(buffer_size=256)
    ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    if num_gpus > 1:
        ds = strategy.experimental_distribute_dataset(ds)
    itds = iter(ds)

    ## Checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizerD=optimizerD,
                                     optimizerG=optimizerG,
                                     modelG=modelG,
                                     modelD=modelD)
    manager = tf.train.CheckpointManager(checkpoint,
                                         saved_model_folder,
                                         max_to_keep=3)
    if manager.latest_checkpoint and args.resume:
        checkpoint.restore(manager.latest_checkpoint)
        current_iteration = checkpoint.step.numpy()
        print('Load ckpt from {} at step {}.'.format(manager.latest_checkpoint,
                                                     checkpoint.step.numpy()))
    else:
        print("Training from scratch.")
    avg_param_G = modelG.get_weights()
    prog_bar = ProgressBar(total_iterations, checkpoint.step.numpy())

    ## Train functions
    def train_d(real_images, fake_images):
        imgs = [
            tf.image.resize(real_images, size=[im_size, im_size]),
            tf.image.resize(real_images, size=[128, 128])
        ]

        with tf.GradientTape() as d_tape:
            ## Real images
            pred_dr, rec_all, rec_small, rec_part = modelD(imgs, training=True)

            sum_rec_all = tf.math.reduce_sum(
                lpips(rec_all, tf.image.resize(real_images,
                                               rec_all.shape[1:3])))
            sum_rec_small = tf.math.reduce_sum(
                lpips(rec_small,
                      tf.image.resize(real_images, rec_small.shape[1:3])))
            sum_rec_part = tf.math.reduce_sum(
                lpips(rec_part,
                      tf.image.resize(real_images, rec_part.shape[1:3])))
            sum_rec = sum_rec_all + sum_rec_small + sum_rec_part

            mean_pred = tf.reduce_mean(
                nn.relu(
                    tf.random.uniform(pred_dr.shape, dtype='float16') * 0.2 +
                    0.8 - pred_dr))

            err_dr = mean_pred + tf.cast(sum_rec, 'float16')

            ## Fake images
            pred_df, _, _, _ = modelD(fake_images, training=True)
            err_df = tf.reduce_mean(
                nn.relu(
                    tf.random.uniform(pred_df.shape, dtype='float16') * 0.2 +
                    0.8 + pred_df))

            err = err_dr + err_df

            scaled_err = optimizerD.get_scaled_loss(err)
        scaled_gradients = d_tape.gradient(scaled_err,
                                           modelD.trainable_variables)
        gradients = optimizerD.get_unscaled_gradients(scaled_gradients)
        optimizerD.apply_gradients(zip(gradients, modelD.trainable_variables))

        return tf.reduce_mean(tf.cast(pred_dr, 'float32')), tf.reduce_mean(
            tf.cast(pred_df, 'float32'))

    @tf.function
    def train_step(real_images):
        with tf.GradientTape() as g_tape:
            noise = tf.random.normal((batch_size, nz), 0, 1)
            fake_images = modelG(noise, training=True)
            real_images = DiffAugment(real_images, policy=policy)
            fake_images = [
                DiffAugment(fake, policy=policy) for fake in fake_images
            ]

            ## 2. train Discriminator
            err_dr, err_df = train_d(real_images, fake_images)

            ## 3. train Generator
            pred_g, _, _, _ = modelD(fake_images, training=True)
            err_g = -tf.reduce_mean(pred_g)

            scaled_err = optimizerG.get_scaled_loss(err_g)
        scaled_gradients = g_tape.gradient(scaled_err,
                                           modelG.trainable_variables)
        gradients = optimizerG.get_unscaled_gradients(scaled_gradients)
        optimizerG.apply_gradients(zip(gradients, modelG.trainable_variables))

        return -tf.reduce_mean(tf.cast(pred_g, 'float32')), err_dr, err_df

    @tf.function
    def distributed_train_step(real_images):
        err_g, err_dr, err_df = strategy.run(train_step, args=(real_images, ))
        err_g = strategy.reduce(tf.distribute.ReduceOp.SUM, err_g, axis=None)
        err_dr = strategy.reduce(tf.distribute.ReduceOp.SUM, err_dr, axis=None)
        err_df = strategy.reduce(tf.distribute.ReduceOp.SUM, err_df, axis=None)
        return err_g, err_dr, err_df

    ## Training loop
    fixed_noise = tf.random.normal((batch_size, nz), 0, 1)
    for iteration in range(current_iteration, total_iterations + 1):
        checkpoint.step.assign_add(1)
        part = tf.convert_to_tensor(random.randint(0, 3))

        if num_gpus > 1:
            err_g, err_dr, err_df = distributed_train_step(next(itds))
        else:
            err_g, err_dr, err_df = train_step(next(itds))

        prog_bar.update(
            "Pred Dr: {:>8.5f}, Pred Df: {:>8.5f}, Pred G: {:>8.5f}".format(
                tf.cast(err_dr, 'float32'), tf.cast(err_df, 'float32'),
                tf.cast(-err_g, 'float32')))

        for i, (w, avg_w) in enumerate(zip(modelG.get_weights(), avg_param_G)):
            avg_param_G[i] = avg_w * 0.999 + 0.001 * w

        if iteration % (save_interval * 10) == 0:
            real_images = next(itds)
            if num_gpus > 1:
                real_images = strategy.gather(real_images, 0)
            real_images = DiffAugment(real_images, policy=policy)
            imgs = [
                tf.image.resize(real_images, size=[im_size, im_size]),
                tf.image.resize(real_images, size=[128, 128])
            ]
            _, rec_img_all, rec_img_small, rec_img_part = modelD(imgs,
                                                                 training=True)
            model_pred = modelG(fixed_noise, training=True)[0]

            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)

            avg_model_pred = modelG(fixed_noise, training=True)[0]
            modelG.set_weights(backup_para)

            all_imgs = tf.concat([
                tf.image.resize(real_images, (128, 128)),
                tf.image.resize(rec_img_all, (128, 128)),
                tf.image.resize(rec_img_small, (128, 128)),
                tf.image.resize(rec_img_part, (128, 128)),
                tf.image.resize(model_pred, (128, 128)),
                tf.image.resize(avg_model_pred, (128, 128))
            ],
                                 axis=0)
            kutils.save_img(saved_image_folder + '/%5d_all.jpg' % iteration,
                            imgrid((all_imgs + 1) * 0.5, real_images.shape[0]))

        if (iteration + 1) % (save_interval *
                              50) == 0 or iteration == total_iterations:
            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)
            manager.save()
            modelG.set_weights(backup_para)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
