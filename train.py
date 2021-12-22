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
        init_lpips_model()
    return lpips_model([imgs_a, imgs_b])


def init_lpips_model():
    global lpips_model
    model_file = './lpips_lin_vgg.h5'
    lpips_model = tf.keras.models.load_model(model_file)


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

    def map_fn(path):
        image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (int(im_size), int(im_size)))
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, 0.5)
        # image = tf.image.random_saturation(image, 0.5, 1.5)
        # image = tf.image.random_contrast(image, 0.5, 1.5)
        # image = tfa.image.translate(image,
        #                            tf.random.uniform(
        #                                (2, ),
        #                                -int(image.shape[1] * 0.125 + 0.5),
        #                                int(image.shape[1] * 0.125 + 0.5)),
        #                            fill_value=255)
        image = tf.image.convert_image_dtype(image, tf.float32) / 127.5 - 1
        return image

    def train_d(model, optimizer, data, label="real"):
        """Train function of discriminator"""
        if label == "real":
            if type(data) is not list:
                imgs = [
                    tf.image.resize(data, size=[im_size, im_size]),
                    tf.image.resize(data, size=[128, 128])
                ]

            with tf.GradientTape() as tape:
                pred, rec_all, rec_small, rec_part = model(imgs, training=True)
                sum_rec_all = tf.math.reduce_sum(
                    lpips(rec_all, tf.image.resize(data, rec_all.shape[1:3])))
                sum_rec_small = tf.math.reduce_sum(
                    lpips(rec_small, tf.image.resize(data,
                                                     rec_small.shape[1:3])))

                sum_rec_part = tf.math.reduce_sum(
                    lpips(
                        rec_part,
                        # tf.image.resize(tf.image.central_crop(data, 0.5),
                        tf.image.resize(data, rec_part.shape[1:3])))
                mean_pred = tf.reduce_mean(
                    nn.relu(
                        tf.random.uniform(pred.shape, dtype='float32') * 0.2 +
                        0.8 - pred))

                err = tf.cast(mean_pred, 'float32') + tf.cast(
                    sum_rec_all + sum_rec_small + sum_rec_part, 'float32')
            grads = tape.gradient(err, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # scaled_loss = optimizer.get_scaled_loss(err)
            # scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
            # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return tf.reduce_mean(pred), rec_all, rec_small, rec_part
        else:
            with tf.GradientTape() as tape:
                pred, _, _, _ = model(data, training=True)
                err = tf.reduce_mean(
                    nn.relu(
                        tf.random.uniform(pred.shape, dtype='float32') * 0.2 +
                        0.8 + pred))
                # scaled_loss = optimizer.get_scaled_loss(tf.cast(err, 'float16'))
            # scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
            # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            grads = tape.gradient(err, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return tf.reduce_mean(pred)

    def train_step(real_image):
        with tf.GradientTape() as g_tape:
            ## 2. train Discriminator
            noise = tf.random.normal((batch_size, nz), 0, 1)
            fake_images = modelG(noise, training=True)
            real_image = DiffAugment(real_image, policy=policy)
            fake_images = [
                DiffAugment(fake, policy=policy) for fake in fake_images
            ]

            err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(
                modelD, optimizerD, real_image, label="real")
            train_d(modelD,
                    optimizerD, [fi for fi in fake_images],
                    label="fake")

            ## 3. train Generator
            pred_g, _, _, _ = modelD(fake_images, training=True)
            err_g = -tf.reduce_mean(pred_g)
        grads_g = g_tape.gradient(err_g, modelG.trainable_variables)
        optimizerG.apply_gradients(zip(grads_g, modelG.trainable_variables))
        # scaled_loss = optimizerG.get_scaled_loss(err_g)
        # scaled_gradients = g_tape.gradient(scaled_loss, modelG.trainable_variables)
        # gradients = optimizerG.get_unscaled_gradients(scaled_gradients)
        # optimizerG.apply_gradients(zip(gradients, modelG.trainable_variables))
        return real_image, err_g, err_dr, rec_img_all, rec_img_small, rec_img_part

    @tf.function
    def distributed_train_step(real_image):
        real_image, err_g, err_dr, rec_img_all, rec_img_small, rec_img_part = strategy.run(
            train_step, args=(real_image, ))
        err_g = strategy.reduce(tf.distribute.ReduceOp.SUM, err_g, axis=None)
        err_dr = strategy.reduce(tf.distribute.ReduceOp.SUM, err_dr, axis=None)
        rec_img_all = strategy.gather(rec_img_all, 0)
        rec_img_small = strategy.gather(rec_img_small, 0)
        rec_img_part = strategy.gather(rec_img_part, 0)
        real_image = strategy.gather(real_image, 0)
        return real_image, err_g, err_dr, rec_img_all, rec_img_small, rec_img_part

    strategy = tf.distribute.MirroredStrategy(devices=None)
    num_gpus = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = num_gpus * batch_size

    filenames = glob.glob(os.path.join(data_root, '*.jpeg'))
    image_paths = tf.convert_to_tensor(filenames, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(image_paths).cache()
    ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE).shuffle(buffer_size=256)
    ds = ds.batch(GLOBAL_BATCH_SIZE)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    with strategy.scope():
        modelG = Generator(input_shape=(nz, ), ngf=ngf, im_size=im_size)
        modelG.summary()
        modelD = Discriminator(ndf=ndf, im_size=im_size)
        modelD.summary()

        # optimizerG = mixed_precision.LossScaleOptimizer(optimizers.Adam(nlr, nbeta1))
        # optimizerD = mixed_precision.LossScaleOptimizer(optimizers.Adam(nlr, nbeta1))
        optimizerG = optimizers.Adam(nlr, nbeta1, epsilon=1e-08)
        optimizerD = optimizers.Adam(nlr, nbeta1, epsilon=1e-08)

    dist_dataset = strategy.experimental_distribute_dataset(ds)
    itds = iter(dist_dataset)

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

    fixed_noise = tf.random.normal((batch_size * num_gpus, nz), 0, 1)

    prog_bar = ProgressBar(total_iterations, checkpoint.step.numpy())

    for iteration in range(current_iteration, total_iterations + 1):
        checkpoint.step.assign_add(1)

        part = tf.convert_to_tensor(random.randint(0, 3))
        real_image = next(itds)
        real_image, err_g, err_dr, rec_img_all, rec_img_small, rec_img_part = distributed_train_step(
            real_image)

        prog_bar.update("Loss D: {:.5f}, Loss G: {:.5f}".format(err_dr, err_g))
        real_image = strategy.gather(real_image, 0)

        for i, (w, avg_w) in enumerate(zip(modelG.get_weights(), avg_param_G)):
            avg_param_G[i] = avg_w * 0.999 + 0.001 * w

        if iteration % (save_interval * 10) == 0:
            model_pred = modelG(fixed_noise, training=True)[0]

            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)

            avg_model_pred = modelG(fixed_noise, training=True)[0]

            all_imgs = tf.concat([
                tf.image.resize(real_image, (128, 128)), rec_img_all,
                rec_img_small, rec_img_part,
                tf.image.resize(model_pred, (128, 128)),
                tf.image.resize(avg_model_pred, (128, 128))
            ],
                                 axis=0)
            kutils.save_img(
                saved_image_folder + '/%5d_all.jpg' % iteration,
                imgrid((all_imgs + 1) * 0.5, batch_size * num_gpus))

            modelG.set_weights(backup_para)

        if (iteration + 1) % (save_interval *
                              50) == 0 or iteration == total_iterations:
            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)
            manager.save()
            modelG.set_weights(backup_para)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument(
        '--path',
        type=str,
        default='./datasets/pokemon/img',
        help=
        'path of resource dataset, should be a folder that has one or many sub image folders inside'
    )
    parser.add_argument('--name', type=str, default='', help='experiment name')
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
    print(args)

    main(args)
