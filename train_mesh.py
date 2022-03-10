import argparse
import logging

logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from tensorflow.keras import mixed_precision, optimizers

from models_mesh import Discriminator, Generator
from operation import (ProgressBar, get_dir, images_to_tensorboard,
                       plot_to_tensorboard, save_weights)

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
    LPIPS_PATH = f'./lpips_lin_{args.lpips_net}.h5'
    TOTAL_ITERATIONS = args.iter
    BATCH_SIZE = args.batch_size
    BATCH_SCALER = 1.0 / float(BATCH_SIZE)
    IM_SIZE = args.im_size
    LR = args.lr

    D_FACTOR = 64
    G_FACTOR = 64
    W_DIM = 256
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
        modelG = Generator(w_dim=W_DIM, factor=G_FACTOR, im_size=IM_SIZE)
        modelG([tf.random.normal((1, W_DIM)), tf.random.normal((1, 468, 3))])
        modelD = Discriminator(w_dim=W_DIM, factor=D_FACTOR, im_size=IM_SIZE)
        modelD([
            tf.random.normal((1, IM_SIZE, IM_SIZE, 3)),
            tf.random.normal((1, 468, 3))
        ])
        optimizerG = mixed_precision.LossScaleOptimizer(optimizers.RMSprop(LR))
        optimizerD = mixed_precision.LossScaleOptimizer(optimizers.RMSprop(LR))
        # lpips = tf.keras.models.load_model(LPIPS_PATH)
    modelD.summary()
    modelG.summary()

    ## Dataset
    def decode_fn(record_bytes):
        return tf.io.parse_single_example(
            record_bytes, {
                'image_raw':
                tf.io.FixedLenFeature([], tf.string),
                'landmarks':
                tf.io.FixedLenSequenceFeature(
                    [], tf.float32, allow_missing=True),
            })

    def map_fn(record_bytes):
        features = decode_fn(record_bytes)

        image_raw = features['image_raw']
        image = tf.io.decode_jpeg(image_raw)
        image = tf.cast(image, tf.float16) / 127.5 - 1

        landmarks = features['landmarks']
        landmarks = tf.reshape(landmarks, [468, 3])
        return image, landmarks

    ds = tf.data.TFRecordDataset('ffhq-mesh.tfrecords')
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
        start_iter = checkpoint.step.numpy()
        print('Load ckpt from {} at step {}.'.format(manager.latest_checkpoint,
                                                     checkpoint.step.numpy()))
    else:
        print("Training from scratch.")
        start_iter = 0

    avg_param_G = modelG.get_weights()
    prog_bar = ProgressBar(TOTAL_ITERATIONS, checkpoint.step.numpy())

    # Train functions
    @tf.function()
    def train_d(real_images, meshes):
        noise = tf.random.normal((int(BATCH_SIZE / N_GPU), W_DIM),
                                 0,
                                 1,
                                 dtype=tf.float16)
        fake_images = modelG([noise, meshes], training=True)

        with tf.GradientTape() as d_tape:
            # Real images
            real_logits = modelD([real_images, meshes], training=True)
            real_score = tf.reduce_sum(
                tf.math.softplus(-real_logits)) * BATCH_SCALER

            # Fake images
            fake_logits = modelD([fake_images, meshes], training=True)
            fake_score = tf.reduce_sum(
                tf.math.softplus(fake_logits)) * BATCH_SCALER

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(real_images)
                real_loss = tf.reduce_sum(
                    modelD([real_images, meshes], training=True))

            grads = gp_tape.gradient(real_loss, real_images)
            gp = tf.reduce_sum(tf.math.square(grads), axis=[1, 2, 3])
            gp = tf.cast(
                tf.reduce_sum(gp) * C_LAMBDA * BATCH_SCALER, tf.float16)

            loss_d = real_score + fake_score + gp

            scaled_loss = optimizerD.get_scaled_loss(loss_d)
        scaled_gradients = d_tape.gradient(scaled_loss,
                                           modelD.trainable_variables)
        gradients = optimizerD.get_unscaled_gradients(scaled_gradients)
        optimizerD.apply_gradients(zip(gradients, modelD.trainable_variables))

        return real_score, fake_score, loss_d, gp

    def distributed_train_d(real_images, meshes):
        real_score, fake_score, loss_d, gradient_penalty = strategy.run(
            train_d, args=(real_images, meshes))
        real_score = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                     real_score,
                                     axis=None)
        fake_score = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                     fake_score,
                                     axis=None)
        loss_d = strategy.reduce(tf.distribute.ReduceOp.SUM, loss_d, axis=None)
        gradient_penalty = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           gradient_penalty,
                                           axis=None)
        return real_score, fake_score, loss_d, gradient_penalty

    @tf.function()
    def train_g(meshes):
        noise = tf.random.normal((int(BATCH_SIZE / N_GPU), W_DIM),
                                 0,
                                 1,
                                 dtype=tf.float16)

        with tf.GradientTape() as g_tape:
            fake_images = modelG([noise, meshes], training=True)

            logits = modelD([fake_images, meshes], training=True)
            loss_g = tf.reduce_sum(tf.math.softplus(-logits)) * BATCH_SCALER

            scaled_loss = optimizerG.get_scaled_loss(loss_g)
        scaled_gradients = g_tape.gradient(scaled_loss,
                                           modelG.trainable_variables)
        gradients = optimizerG.get_unscaled_gradients(scaled_gradients)
        optimizerG.apply_gradients(zip(gradients, modelG.trainable_variables))

        return loss_g

    def distributed_train_g(meshes):
        loss_g = strategy.run(train_g, args=(meshes, ))
        loss_g = strategy.reduce(tf.distribute.ReduceOp.SUM, loss_g, axis=None)
        return loss_g

    ## Training loop
    fixed_noise = tf.random.normal((8, W_DIM), 0, 1, seed=42)
    for iteration in range(start_iter, TOTAL_ITERATIONS):
        checkpoint.step.assign_add(1)
        cur_step = checkpoint.step.numpy()
        update_D, update_G = 0, 0

        # Run one step of the model.
        real_images, meshes = next(itds)
        real_score, fake_score, loss_d, gradient_penalty = distributed_train_d(
            real_images, meshes)

        if iteration == start_iter:
            tf.summary.trace_on(graph=True)
            loss_g = distributed_train_g(meshes)
            with summary_writer_1.as_default():
                tf.summary.trace_export(name='Graph', step=0)
        else:
            loss_g = distributed_train_g(meshes)

        # Update current and previous loss
        if iteration == start_iter:
            ldp, lgp = loss_d, loss_g
            ldc, lgc = loss_d, loss_g
        else:
            ldc, lgc = loss_d, loss_g

        for _ in range(N_BALANCE_ITER):
            # Update loss change ratio
            rg, rd = tf.abs((lgc - lgp) / lgp), tf.abs((ldc - ldp) / ldp)

            # Run another step of D or G based on loss change ratio
            if rd > 2 * rg:
                real_images, meshes = next(itds)
                real_score, fake_score, loss_d, gradient_penalty = distributed_train_d(
                    real_images, meshes)
                update_D += 1
                ldc = loss_d
            else:
                loss_g = distributed_train_g(meshes)
                update_G += 1
                lgc = loss_g

            # Update previous loss
            lgp, ldp = lgc, ldc

        plot_to_tensorboard(summary_writer_1, summary_writer_2, cur_step,
                            real_score, fake_score, loss_d, loss_g,
                            update_D - update_G, gradient_penalty, modelG)

        prog_bar.update(
            "Pred Dr: {:>9.5f}, Pred Df: {:>9.5f}, Pred G: {:>9.5f}".format(
                tf.cast(real_score, 'float32'), tf.cast(fake_score, 'float32'),
                tf.cast(-loss_g, 'float32')))

        for i, (w, avg_w) in enumerate(zip(modelG.get_weights(), avg_param_G)):
            avg_param_G[i] = avg_w * 0.999 + 0.001 * w

        images_to_tensorboard(summary_writer_1, cur_step, fixed_noise,
                              real_images, meshes, modelG, avg_param_G,
                              IMAGE_FOLDER)

        save_weights(modelG, avg_param_G, cur_step, TOTAL_ITERATIONS, manager,
                     MODEL_FOLDER)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
