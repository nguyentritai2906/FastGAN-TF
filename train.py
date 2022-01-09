import argparse
import glob
import logging
import os

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

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

POLICY = 'color,translation'
mixed_precision.set_global_policy('mixed_float16')


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
                        default=16,
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
    AUTOTUNE = tf.data.AUTOTUNE
    DATA_ROOT = args.path
    LPIPS_PATH = f'./lpips_lin_{args.lpips_net}.h5'
    TOTAL_ITERATIONS = args.iter
    BATCH_SIZE = args.batch_size
    IM_SIZE = args.im_size
    LR = args.lr

    NDF = 64
    NGF = 64
    NZ = 256
    BETA1 = 0.5
    MODEL_FOLDER, IMAGE_FOLDER = get_dir(args)

    print(
        f"Number of GPUs in use: {len(tf.config.list_physical_devices('GPU'))}"
    )

    ## Model
    strategy = tf.distribute.MirroredStrategy(devices=None)
    with strategy.scope():
        modelG = Generator(input_shape=(NZ, ), ngf=NGF, im_size=IM_SIZE)
        modelD = Discriminator(ndf=NDF, im_size=IM_SIZE)
        optimizerG = mixed_precision.LossScaleOptimizer(
            optimizers.Adam(LR, BETA1))
        optimizerD = mixed_precision.LossScaleOptimizer(
            optimizers.Adam(LR, BETA1))
        lpips = tf.keras.models.load_model(LPIPS_PATH)
    modelG.summary()
    modelD.summary()

    ## Dataset
    filenames = glob.glob(os.path.join(DATA_ROOT, '*.jp*'))
    image_paths = tf.convert_to_tensor(filenames, dtype=tf.string)

    def map_fn(path):
        image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (int(IM_SIZE), int(IM_SIZE)))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.convert_image_dtype(image, tf.float16) / 127.5 - 1
        return image

    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(map_fn,
                num_parallel_calls=AUTOTUNE).repeat().shuffle(buffer_size=256)
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    ds = strategy.experimental_distribute_dataset(ds)
    itds = iter(ds)

    ## Checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizerD=optimizerD,
                                     optimizerG=optimizerG,
                                     modelG=modelG,
                                     modelD=modelD)
    manager = tf.train.CheckpointManager(checkpoint,
                                         MODEL_FOLDER,
                                         max_to_keep=3)
    if manager.latest_checkpoint and args.resume:
        checkpoint.restore(manager.latest_checkpoint)
        current_iteration = checkpoint.step.numpy()
        print('Load ckpt from {} at step {}.'.format(manager.latest_checkpoint,
                                                     checkpoint.step.numpy()))
    else:
        print("Training from scratch.")
        current_iteration = 0

    avg_param_G = modelG.get_weights()
    prog_bar = ProgressBar(TOTAL_ITERATIONS, checkpoint.step.numpy())

    ## Train functions
    def train_d(real_images, fake_images):
        imgs = [
            tf.image.resize(real_images, size=[IM_SIZE, IM_SIZE]),
            tf.image.resize(real_images, size=[128, 128])
        ]

        with tf.GradientTape() as d_tape:
            ## Real images
            pred_dr, rec_all, rec_small, rec_part = modelD(imgs, training=True)

            sum_rec_all = tf.math.reduce_sum(
                lpips([
                    rec_all,
                    tf.image.resize(real_images, rec_all.shape[1:3])
                ]))
            sum_rec_small = tf.math.reduce_sum(
                lpips([
                    rec_small,
                    tf.image.resize(real_images, rec_small.shape[1:3])
                ]))
            sum_rec_part = tf.math.reduce_sum(
                lpips([
                    rec_part,
                    tf.image.resize(real_images, rec_part.shape[1:3])
                ]))
            sum_rec = sum_rec_all + sum_rec_small + sum_rec_part

            mean_pred = tf.reduce_mean(
                nn.relu(
                    tf.random.normal(pred_dr.shape, dtype=pred_dr.dtype) *
                    0.2 + 0.8 - pred_dr))

            err_dr = mean_pred + tf.cast(sum_rec, pred_dr.dtype)

            ## Fake images
            pred_df, _, _, _ = modelD(fake_images, training=True)
            err_df = tf.reduce_mean(
                nn.relu(
                    tf.random.normal(pred_df.shape, dtype=pred_df.dtype) *
                    0.2 + 0.8 + pred_df))

            err = err_dr + err_df

            scaled_err = optimizerD.get_scaled_loss(err)
        scaled_gradients = d_tape.gradient(scaled_err,
                                           modelD.trainable_variables)
        gradients = optimizerD.get_unscaled_gradients(scaled_gradients)
        optimizerD.apply_gradients(zip(gradients, modelD.trainable_variables))

        return tf.reduce_mean(pred_dr), tf.reduce_mean(pred_df)

    @tf.function()
    def train_step(real_images):
        with tf.GradientTape() as g_tape:
            noise = tf.random.normal((real_images.shape[0], NZ),
                                     0,
                                     1,
                                     dtype=real_images.dtype)
            fake_images = modelG(noise, training=True)

            real_images = DiffAugment(real_images, policy=POLICY)
            fake_images = [
                DiffAugment(fake, policy=POLICY) for fake in fake_images
            ]

            ## Train Discriminator
            err_dr, err_df = train_d(real_images, fake_images)

            ## Train Generator
            pred_g, _, _, _ = modelD(fake_images, training=True)
            err_g = -tf.reduce_mean(pred_g)

            scaled_err = optimizerG.get_scaled_loss(err_g)
        scaled_gradients = g_tape.gradient(scaled_err,
                                           modelG.trainable_variables)
        gradients = optimizerG.get_unscaled_gradients(scaled_gradients)
        optimizerG.apply_gradients(zip(gradients, modelG.trainable_variables))

        return -tf.reduce_mean(pred_g), err_dr, err_df

    def distributed_train_step(real_images):
        err_g, err_dr, err_df = strategy.run(train_step, args=(real_images, ))
        err_g = strategy.reduce(tf.distribute.ReduceOp.SUM, err_g, axis=None)
        err_dr = strategy.reduce(tf.distribute.ReduceOp.SUM, err_dr, axis=None)
        err_df = strategy.reduce(tf.distribute.ReduceOp.SUM, err_df, axis=None)
        return err_g, err_dr, err_df

    ## Training loop
    fixed_noise = tf.random.normal((BATCH_SIZE, NZ), 0, 1, seed=42)
    for iteration in range(current_iteration, TOTAL_ITERATIONS):
        checkpoint.step.assign_add(1)
        cur_step = checkpoint.step.numpy()

        real_images = next(itds)
        err_g, err_dr, err_df = distributed_train_step(real_images)

        prog_bar.update(
            "Pred Dr: {:>9.5f}, Pred Df: {:>9.5f}, Pred G: {:>9.5f}".format(
                tf.cast(err_dr, 'float32'), tf.cast(err_df, 'float32'),
                tf.cast(-err_g, 'float32')))

        for i, (w, avg_w) in enumerate(zip(modelG.get_weights(), avg_param_G)):
            avg_param_G[i] = avg_w * 0.999 + 0.001 * w

        ## Save image
        if cur_step % 1000 == 0:
            real_images = next(itds)
            real_images = strategy.gather(real_images, 0)
            real_images = DiffAugment(real_images, policy=POLICY)
            imgs = [
                tf.image.resize(real_images, size=[IM_SIZE, IM_SIZE]),
                tf.image.resize(real_images, size=[128, 128])
            ]
            _, rec_img_all, rec_img_small, rec_img_part = modelD(imgs,
                                                                 training=True)
            model_pred_fnoise = modelG(fixed_noise, training=True)[0]

            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)

            avg_model_pred_fnoise = modelG(fixed_noise, training=True)[0]
            modelG.set_weights(backup_para)

            all_imgs = tf.concat([
                tf.image.resize(real_images, (128, 128)),
                tf.image.resize(rec_img_all, (128, 128)),
                tf.image.resize(rec_img_small, (128, 128)),
                tf.image.resize(rec_img_part, (128, 128)),
                tf.image.resize(model_pred_fnoise, (128, 128)),
                tf.image.resize(avg_model_pred_fnoise, (128, 128)),
            ],
                                 axis=0)
            kutils.save_img(IMAGE_FOLDER + '/%5d_all.jpg' % cur_step,
                            imgrid((all_imgs + 1) * 0.5, real_images.shape[0]))
            kutils.save_img(IMAGE_FOLDER + '/%5d_fix.jpg' % cur_step,
                            imgrid((avg_model_pred_fnoise + 1) * 0.5, 4))

        if cur_step % 500 == 0:
            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)
            avg_model_pred_fnoise = modelG(fixed_noise, training=True)[0]
            kutils.save_img(IMAGE_FOLDER + '/%5d_fix.jpg' % cur_step,
                            imgrid((avg_model_pred_fnoise + 1) * 0.5, 4))
            modelG.set_weights(backup_para)

        ## Save model
        if cur_step % 5000 == 0 or cur_step == TOTAL_ITERATIONS:
            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)
            manager.save()
            modelG.save(os.path.join(MODEL_FOLDER, args.name + '.h5'))
            modelG.set_weights(backup_para)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
