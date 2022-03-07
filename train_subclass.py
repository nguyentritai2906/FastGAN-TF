import argparse
import datetime
import glob
import logging
import os
import random

logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import mixed_precision, optimizers
from tensorflow.keras import utils as kutils

from diffaug import DiffAugment
from models_subclass import Discriminator, Generator
from operation import ProgressBar, get_dir, imgrid

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
    BATCH_SCALER = 1.0 / float(BATCH_SIZE)
    IM_SIZE = args.im_size
    LR = args.lr

    NDF = 64
    NGF = 128
    NZ = 256
    N_CRITIC_ITER = 1
    N_GENERATOR_ITER = 1
    N_BALANCE_ITER = 2
    BETA1 = 0.5
    C_LAMBDA = 10
    N_GPU = len(tf.config.experimental.list_physical_devices('GPU'))
    MODEL_FOLDER, IMAGE_FOLDER = get_dir(args)

    print(
        f"Number of GPUs in use: {len(tf.config.list_physical_devices('GPU'))}"
    )

    log_dir = 'logs/' + args.name
    summary_writer_1 = tf.summary.create_file_writer(log_dir + '/writer_1')
    summary_writer_2 = tf.summary.create_file_writer(log_dir + '/writer_2')

    ## Model
    strategy = tf.distribute.MirroredStrategy(devices=None)
    with strategy.scope():
        modelG = Generator(w_dim=NZ, ngf=NGF, im_size=IM_SIZE)
        modelG(tf.random.normal([1, NZ]))
        modelD = Discriminator(ndf=NDF, im_size=IM_SIZE)
        modelD(tf.random.normal([1, IM_SIZE, IM_SIZE, 3]))
        optimizerG = mixed_precision.LossScaleOptimizer(optimizers.RMSprop(LR))
        optimizerD = mixed_precision.LossScaleOptimizer(optimizers.RMSprop(LR))
        # lpips = tf.keras.models.load_model(LPIPS_PATH)
    modelD.summary()
    modelG.summary()

    ## Dataset
    def decode_fn(record_bytes):
        return tf.io.parse_single_example(
            record_bytes, {'image_raw': tf.io.FixedLenFeature([], tf.string)})

    def map_fn(record_bytes):
        features = decode_fn(record_bytes)

        image_raw = features['image_raw']
        image = tf.io.decode_jpeg(image_raw)
        image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float16) / 127.5 - 1

        return image

    ds = tf.data.TFRecordDataset('ffhq-256.tfrecords')
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
    @tf.function()
    def train_d(real_images):
        noise = tf.random.normal((int(BATCH_SIZE / N_GPU), NZ),
                                 0,
                                 1,
                                 dtype=tf.float16)
        fake_images = modelG(noise, training=True)

        alpha = tf.random.uniform((int(BATCH_SIZE / N_GPU), 1, 1, 1),
                                  0,
                                  1,
                                  dtype=tf.float16)
        interpolates = (alpha * tf.cast(real_images, tf.float16)) + (
            (1 - alpha) * fake_images)

        with tf.GradientTape() as d_tape:
            ## Real images
            pred_dr = modelD(real_images, training=True)
            mean_dr = tf.reduce_mean(pred_dr)

            # Fake images
            pred_df = modelD(fake_images, training=True)
            mean_df = tf.reduce_mean(pred_df)

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolates)
                pred = modelD(interpolates, training=True)
                pred = tf.reduce_mean(pred)

            grads = gp_tape.gradient(pred, interpolates)
            norm = tf.sqrt(tf.reduce_sum(grads**2, axis=[1, 2, 3]))
            gp = tf.reduce_mean((norm - 1.0)**2)

            err = mean_df - mean_dr + C_LAMBDA * tf.cast(gp, tf.float16)

            scaled_err = optimizerD.get_scaled_loss(err)
        scaled_gradients = d_tape.gradient(scaled_err,
                                           modelD.trainable_variables)
        gradients = optimizerD.get_unscaled_gradients(scaled_gradients)
        optimizerD.apply_gradients(zip(gradients, modelD.trainable_variables))

        return mean_dr, mean_df, err, gp

    def distributed_train_d(real_images):
        err_dr, err_df, loss_d, gradient_penalty = strategy.run(
            train_d, args=(real_images, ))
        err_dr = strategy.reduce(tf.distribute.ReduceOp.SUM, err_dr, axis=None)
        err_df = strategy.reduce(tf.distribute.ReduceOp.SUM, err_df, axis=None)
        loss_d = strategy.reduce(tf.distribute.ReduceOp.SUM, loss_d, axis=None)
        gradient_penalty = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           gradient_penalty,
                                           axis=None)
        return err_dr, err_df, loss_d, gradient_penalty

    @tf.function()
    def train_g():
        with tf.GradientTape() as g_tape:
            noise = tf.random.normal((int(BATCH_SIZE / N_GPU), NZ),
                                     0,
                                     1,
                                     dtype=tf.float16)
            fake_images = modelG(noise, training=True)

            pred_g = modelD(fake_images, training=True)
            mean_g = tf.reduce_mean(pred_g)
            err_g = -mean_g

            scaled_err = optimizerG.get_scaled_loss(err_g)
        scaled_gradients = g_tape.gradient(scaled_err,
                                           modelG.trainable_variables)
        gradients = optimizerG.get_unscaled_gradients(scaled_gradients)
        optimizerG.apply_gradients(zip(gradients, modelG.trainable_variables))

        return mean_g, err_g

    def distributed_train_g():
        err_g, loss_g = strategy.run(train_g, args=())
        err_g = strategy.reduce(tf.distribute.ReduceOp.SUM, err_g, axis=None)
        loss_g = strategy.reduce(tf.distribute.ReduceOp.SUM, loss_g, axis=None)
        return err_g, loss_g

    ## Training loop
    fixed_noise = tf.random.normal((16, NZ), 0, 1, seed=42)
    for iteration in range(current_iteration, TOTAL_ITERATIONS):
        checkpoint.step.assign_add(1)
        cur_step = checkpoint.step.numpy()
        update_D, update_G = 0, 0

        # Run one step of the model.
        real_images = next(itds)
        err_dr, err_df, loss_d, gradient_penalty = distributed_train_d(
            real_images)

        if iteration == current_iteration:
            tf.summary.trace_on(graph=True)
            err_g, loss_g = distributed_train_g()
            with summary_writer_1.as_default():
                tf.summary.trace_export(name='Graph', step=0)
        else:
            err_g, loss_g = distributed_train_g()

        # Update current and previous loss
        if iteration == current_iteration:
            ldp, lgp = loss_d, loss_g
            ldc, lgc = loss_d, loss_g
        else:
            ldc, lgc = loss_d, loss_g

        for _ in range(N_BALANCE_ITER):
            # Update loss change ratio
            rg, rd = tf.abs((lgc - lgp) / lgp), tf.abs((ldc - ldp) / ldp)

            # Run another step of D or G based on loss change ratio
            if rd > 2 * rg:
                real_images = next(itds)
                err_dr, err_df, loss_d, gradient_penalty = distributed_train_d(
                    real_images)
                update_D += 1
                ldc = loss_d
            else:
                # err_g, loss_g, gradientsG = distributed_train_g()
                err_g, loss_g = distributed_train_g()
                update_G += 1
                lgc = loss_g

            # Update previous loss
            lgp, ldp = lgc, ldc

        with summary_writer_1.as_default():
            tf.summary.scalar('Pred/Pred_Dr', err_dr, step=cur_step)
            tf.summary.scalar('Pred/Pred_Df', err_df, step=cur_step)
            tf.summary.scalar('Pred/Pred_G', err_g, step=cur_step)

            tf.summary.scalar('Loss/Loss_G', loss_g, step=cur_step)
            tf.summary.scalar('Loss/Loss_D', loss_d, step=cur_step)
            tf.summary.scalar('Loss/Loss_GP', gradient_penalty, step=cur_step)

            tf.summary.scalar('Misc/Balance',
                              update_D - update_G,
                              step=cur_step)

            # Combined
            tf.summary.scalar('Loss/Loss', loss_d, step=cur_step)
            tf.summary.scalar('Pred/Pred_D', err_dr, step=cur_step)

            # for i, layer in enumerate(modelG.layers):

            #     if 'mapping' == layer.name:
            #         for l in layer.dense_layers:
            #             tf.summary.histogram(
            #                 f'Mapping/{layer.name}/{l.name}/weight',
            #                 l.w,
            #                 step=cur_step)
            #         for l in layer.bias_act_layers:
            #             tf.summary.histogram(
            #                 f'Mapping/{layer.name}/{l.name}/bias',
            #                 l.b,
            #                 step=cur_step)

            #     if 'const_layer' == layer.name:
            #         tf.summary.histogram(f'Init/{layer.name}/const',
            #                              layer.const,
            #                              step=cur_step)

            # for g, v in zip(gradientsG, modelG.layers):
            #     tf.summary.histogram("GradientG/{}/grad_histogram".format(v.name), g, step=cur_step)
            #     tf.summary.scalar("GradientG/{}/grad/sparsity".format(v.name), tf.reduce_sum(g), step=cur_step)

        with summary_writer_2.as_default():
            tf.summary.scalar('Loss/Loss', loss_g, step=cur_step)
            tf.summary.scalar('Pred/Pred_D', err_df, step=cur_step)

        prog_bar.update(
            "Pred Dr: {:>9.5f}, Pred Df: {:>9.5f}, Pred G: {:>9.5f}".format(
                tf.cast(err_dr, 'float32'), tf.cast(err_df, 'float32'),
                tf.cast(err_g, 'float32')))

        for i, (w, avg_w) in enumerate(zip(modelG.get_weights(), avg_param_G)):
            avg_param_G[i] = avg_w * 0.998 + 0.002 * w

        ## Save image
        if cur_step % 1000 == 0:
            real_images = next(itds)
            real_images = strategy.gather(real_images, 0)[:16]
            model_pred_fnoise = modelG(fixed_noise)

            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)

            avg_model_pred_fnoise = modelG(fixed_noise)
            modelG.set_weights(backup_para)

            all_imgs = tf.concat([
                tf.image.resize(real_images / tf.reduce_max(real_images),
                                (128, 128)),
                tf.image.resize(model_pred_fnoise, (128, 128)),
                tf.image.resize(avg_model_pred_fnoise, (128, 128))
            ],
                                 axis=0)

            grid_all = imgrid((all_imgs + 1) * 0.5, 8)
            grid_pred = imgrid((avg_model_pred_fnoise + 1) * 0.5, 4)

            kutils.save_img(IMAGE_FOLDER + '/%06d_all.jpg' % cur_step,
                            grid_all)
            kutils.save_img(IMAGE_FOLDER + '/%06d_fix.jpg' % cur_step,
                            grid_pred)
            with summary_writer_1.as_default():
                tf.summary.image("All images",
                                 tf.expand_dims(grid_all, 0),
                                 step=cur_step)
                tf.summary.image("Fixed noise",
                                 tf.expand_dims(grid_pred, 0),
                                 step=cur_step)
        elif cur_step % 500 == 0:
            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)
            avg_model_pred_fnoise = modelG(fixed_noise)
            grid_pred = imgrid((avg_model_pred_fnoise + 1) * 0.5, 4)
            kutils.save_img(IMAGE_FOLDER + '/%06d_fix.jpg' % cur_step,
                            grid_pred)
            modelG.set_weights(backup_para)
            with summary_writer_1.as_default():
                tf.summary.image("Fixed noise",
                                 tf.expand_dims(grid_pred, 0),
                                 step=cur_step)

        if cur_step % 5000 == 0 or cur_step == TOTAL_ITERATIONS:
            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)
            manager.save()
            tf.saved_model.save(modelG, MODEL_FOLDER + '/saved_model')
            modelG.set_weights(backup_para)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
