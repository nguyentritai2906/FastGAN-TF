import argparse
import glob
import os
import random

import tensorflow as tf
from tensorflow import keras, nn
from tensorflow.keras import layers, optimizers
from tensorflow.keras import utils as kutils

from diffaug import DiffAugment
from models import Discriminator, Generator
from operation import get_dir

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
    """LPIPS: Learned Perceptual Image Patch Similarity metric

    R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang,
    “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,”
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, Jun. 2018, pp. 586–595.
    doi: 10.1109/CVPR.2018.00068.
    """
    if lpips_model is None:
        init_lpips_model()
    return lpips_model([imgs_a, imgs_b])


def init_lpips_model():
    global lpips_model
    # model_file = tf.keras.utils.get_file(
    #     "lpips_lin_alex_0.1.0",
    #     "https://github.com/HedgehogCode/lpips-tf2/releases/download/0.1.0/lpips_lin_alex.h5",
    #     file_hash="d76a9756bf43f6b731a845968e8225ad",
    #     hash_algorithm="md5",
    # )
    model_file = './lpips_lin_vgg.h5'
    lpips_model = tf.keras.models.load_model(model_file)


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :hw, :hw, :]
    if part == 1:
        return image[:, :hw, hw:, :]
    if part == 2:
        return image[:, hw:, :hw, :]
    if part == 3:
        return image[:, hw:, hw:, :]


def train_d(model, optimizer, data, label="real"):
    """Train function of discriminator"""
    with tf.GradientTape() as tape:
        if label == "real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = model(data,
                                                         label,
                                                         part=part)
            sum_rec_all = tf.math.reduce_sum(
                lpips(
                    rec_all,
                    tf.image.resize(data,
                                    [rec_all.shape[2], rec_all.shape[2]])))
            sum_rec_small = tf.math.reduce_sum(
                lpips(
                    rec_small,
                    tf.image.resize(data,
                                    [rec_small.shape[2], rec_small.shape[2]])))

            parts = crop_image_by_part(data, part)
            resized_parts = tf.image.resize(
                parts, [rec_part.shape[2], rec_part.shape[2]])

            sum_rec_part = tf.math.reduce_sum(lpips(rec_part, resized_parts))
            mean_pred = tf.reduce_mean(
                nn.relu(tf.random.uniform(pred.shape) * 0.2 + 0.8 - pred))

            err = mean_pred + sum_rec_all + sum_rec_small + sum_rec_part
            grads = tape.gradient(err, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return tf.reduce_mean(pred), rec_all, rec_small, rec_part
        else:
            pred = model(data, label)
            err = tf.reduce_mean(
                nn.relu(tf.random.uniform(pred.shape) * 0.2 + 0.8 + pred))
            grads = tape.gradient(err, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return tf.reduce_mean(pred)


def train_g(model, optimizer, data):
    with tf.GradientTape() as tape:
        pred = model(data, "fake")
        err = -tf.reduce_mean(pred)

        grads = tape.gradient(err, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return err


def train(args):
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    """
        transform_list = [
                transforms.Resize((int(im_size),int(im_size))),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        trans = transforms.Compose(transform_list)

        dataset = ImageFolder(root=data_root, transform=trans)

        dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        sampler=InfiniteSamplerWrapper(dataset),
                        num_workers=dataloader_workers, pin_memory=True))
    """
    def map_fn(path):
        image = tf.image.decode_jpeg(tf.io.read_file(path))
        image = tf.image.resize(image, (int(im_size), int(im_size)))
        image = tf.math.divide(tf.cast(image, tf.float32), 255.)
        image = tf.image.random_flip_left_right(image)
        return image

    filenames = glob.glob(os.path.join(data_root, '*.jpeg'))
    epoch_size = len(filenames)
    image_paths = tf.convert_to_tensor(filenames, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.repeat().shuffle(epoch_size)
    ds = ds.map(map_fn, num_parallel_calls=dataloader_workers)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    itds = iter(ds)

    #from model_s import Generator, Discriminator
    modelG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    # modelG.build((batch_size, nz))
    modelG(tf.random.uniform((batch_size, nz)))
    modelD = Discriminator(ndf=ndf, im_size=im_size)

    avg_param_G = modelG.get_weights()

    fixed_noise = tf.random.normal((8, nz), 0, 1)

    optimizerG = optimizers.Adam(nlr, nbeta1)
    optimizerD = optimizers.Adam(nlr, nbeta1)

    checkpoint_dir = args.checkpoint_dir
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizerD=optimizerD,
                                     optimizerG=optimizerG,
                                     avgG=avg_param_G,
                                     modelG=modelG,
                                     modelD=modelD)
    manager = tf.train.CheckpointManager(checkpoint,
                                         checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint and args.resume:
        checkpoint.restore(manager.latest_checkpoint)
        print('Load ckpt from {} at step {}.'.format(manager.latest_checkpoint,
                                                     checkpoint.step.numpy()))
    else:
        print("Training from scratch.")

    for iteration in range(current_iteration, total_iterations + 1):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        real_image = next(itds)
        current_batch_size = real_image.shape[0]
        noise = tf.random.normal((current_batch_size, nz), 0, 1)

        fake_images = modelG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [
            DiffAugment(fake, policy=policy) for fake in fake_images
        ]

        ## 2. train Discriminator
        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(
            modelD, optimizerD, real_image, label="real")
        train_d(modelD, optimizerD, [fi for fi in fake_images], label="fake")

        ## 3. train Generator
        err_g = train_g(modelD, optimizerG, fake_images)

        for i, (w, avg_w) in enumerate(zip(modelG.get_weights(), avg_param_G)):
            avg_param_G[i] = avg_w * 0.999 + 0.001 * w

        if iteration % 100 == 0:
            print("Loss D: %.5f    Loss G: %.5f" % (err_dr, -err_g))

        # if iteration % (save_interval * 10) == 0:
        #     backup_para = modelG.get_weights()
        #     modelG.set_weights(avg_param_G)

        #     kutils.save_img(
        #         saved_image_folder + '/%d.jpg' % iteration,
        #         tf.reshape(
        #             tf.add(modelG(fixed_noise)[0], 1) * 0.5,
        #             (1024 * 2, -1, 3)))
        #     kutils.save_img(
        #         saved_image_folder + '/rec_%d.jpg' % iteration,
        #         tf.add(
        #             tf.concat([
        #                 tf.image.resize(real_image, (128, 128)), rec_img_all,
        #                 rec_img_small, rec_img_part
        #             ],
        #                       axis=0), 1) * 0.5)

        #     modelG.set_weights(backup_para)

        if iteration % (save_interval *
                        50) == 0 or iteration == total_iterations:
            backup_para = modelG.get_weights()
            modelG.set_weights(avg_param_G)

            ckpt_save_path = manager.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument(
        '--path',
        type=str,
        default='./datasets/pokemon/img',
        help=
        'path of resource dataset, should be a folder that has one or many sub image folders inside'
    )
    parser.add_argument('--name',
                        type=str,
                        default='test1',
                        help='experiment name')
    parser.add_argument('--iter',
                        type=int,
                        default=50000,
                        help='number of iterations')
    parser.add_argument('--start_iter',
                        type=int,
                        default=0,
                        help='the iteration to start training')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='mini batch number of images')
    parser.add_argument('--im_size',
                        type=int,
                        default=1024,
                        help='image resolution')
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='continue training from latest checkpoint')
    parser.add_argument('--ckpt',
                        type=str,
                        default='None',
                        help='checkpoint weight path if have one')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='./logs/checkpoint',
                        help='checkpoint weight path if have one')

    args = parser.parse_args()
    print(args)

    train(args)
