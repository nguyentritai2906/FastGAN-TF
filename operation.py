import json
import logging
import os
import shutil
import sys
import time

import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as kutils


def get_dir(args):
    task_name = 'train_results/' + args.name
    saved_model_folder = os.path.join(task_name, 'models')
    saved_image_folder = os.path.join(task_name, 'images')

    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    for f in os.listdir('./'):
        if '.py' in f:
            shutil.copy(f, task_name + '/' + f)

    with open(os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder


class ProgressBar(object):
    """A progress bar which can print the progress modified from
       https://github.com/hellock/cvbase/blob/master/cvbase/progress.py"""

    def __init__(self, task_num=0, completed=0, bar_width=25):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width
                          if bar_width <= max_bar_width else max_bar_width)
        self.completed = completed
        self.first_step = completed
        self.warm_up = False

    def _get_max_bar_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            logging.info('terminal width is too small ({}), please consider '
                         'widen the terminal for better progressbar '
                         'visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def reset(self):
        """reset"""
        self.completed = 0
        self.fps = 0

    def update(self, inf_str=''):
        """update"""
        self.completed += 1

        if not self.warm_up:
            self.start_time = time.time() - 1e-1
            self.warm_up = True

        if self.completed > self.task_num:
            self.completed = self.completed % self.task_num
            self.start_time = time.time() - 1 / self.fps
            self.first_step = self.completed - 1
            sys.stdout.write('\n')

        elapsed = time.time() - self.start_time
        self.fps = (self.completed - self.first_step) / elapsed
        percentage = self.completed / float(self.task_num)
        mark_width = int(self.bar_width * percentage)
        bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
        stdout_str = '\rTraining [{}] {}/{}, {}  {:.1f} step/sec  ETA {:.1f} hrs'
        sys.stdout.write(
            stdout_str.format(
                bar_chars, self.completed, self.task_num, inf_str, self.fps,
                (self.task_num - self.completed) / self.fps / 3600))

        sys.stdout.flush()


def imgrid(imarray, cols=4):
    """Lays out a [N, H, W, C] image array as a single image grid."""
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = N // cols + int(N % cols != 0)
    grid = tf.reshape(
        tf.transpose(tf.reshape(imarray, [rows, cols, H, W, C]),
                     [0, 2, 1, 3, 4]), [rows * H, cols * W, C])
    return grid


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    h = part // 2 * hw
    w = part % 2 * hw
    return image[:, h:h + hw, w:w + hw, :]


def generate_landmarks(image):
    mp_facemesh = mp.solutions.face_mesh
    landmarks = []

    with mp_facemesh.FaceMesh(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as facemesh:

        result = facemesh.process(image)
        if result.multi_face_landmarks:
            for face_landmark in result.multi_face_landmarks:
                for landmark in face_landmark.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)
                    landmarks.append(landmark.z)

    if len(landmarks) == 0:
        return tf.zeros([468, 3], dtype=tf.float16)

    return tf.reshape(landmarks, [468, 3])


def imgs_to_landmarks(imgs):
    imgs = tf.cast(imgs, tf.uint8)
    imgs = np.array(imgs)
    landmarks_arrays = []
    for img in imgs:
        landmarks = generate_landmarks(img)
        landmarks_arrays.append(landmarks)
    return tf.convert_to_tensor(landmarks_arrays)


def plot_to_tensorboard(writer_1,
                        writer_2,
                        cur_step,
                        real_score,
                        fake_score,
                        loss_d,
                        loss_g,
                        balance,
                        gradient_penalty,
                        modelG,
                        plot_histogram=False,
                        plot_gradient=False,
                        gradientsG=None):

    with writer_1.as_default():
        tf.summary.scalar('Pred/Pred_Dr', real_score, step=cur_step)
        tf.summary.scalar('Pred/Pred_Df', fake_score, step=cur_step)
        tf.summary.scalar('Pred/Pred_G', -loss_g, step=cur_step)

        tf.summary.scalar('Loss/Loss_G', loss_g, step=cur_step)
        tf.summary.scalar('Loss/Loss_D', loss_d, step=cur_step)
        tf.summary.scalar('Loss/Loss_GP', gradient_penalty, step=cur_step)

        tf.summary.scalar('Misc/Balance', balance, step=cur_step)

        # Combined
        tf.summary.scalar('Loss/Loss', loss_d, step=cur_step)
        tf.summary.scalar('Pred/Pred_D', real_score, step=cur_step)

        if plot_histogram:
            for _, layer in enumerate(modelG.layers):
                if 'mapping' == layer.name:
                    for l in layer.dense_layers:
                        tf.summary.histogram(
                            f'Mapping/{layer.name}/{l.name}/weight',
                            l.w,
                            step=cur_step)
                    for l in layer.bias_act_layers:
                        tf.summary.histogram(
                            f'Mapping/{layer.name}/{l.name}/bias',
                            l.b,
                            step=cur_step)

                if 'const_layer' == layer.name:
                    tf.summary.histogram(f'Init/{layer.name}/const',
                                         layer.const,
                                         step=cur_step)

                if 'epilogue' in layer.name:
                    tf.summary.histogram(
                        f'Epilogue/{layer.name}/inject_noise/weight',
                        layer.inject_noise.w,
                        step=cur_step)
                    tf.summary.histogram(
                        f'Epilogue/{layer.name}/adain/scale_weight',
                        layer.adain.style_scale_transform_dense.w,
                        step=cur_step)
                    tf.summary.histogram(
                        f'Epilogue/{layer.name}/adain/scale_bias',
                        layer.adain.style_scale_transform_bias.b,
                        step=cur_step)
                    tf.summary.histogram(
                        f'Epilogue/{layer.name}/adain/shift_weight',
                        layer.adain.style_shift_transform_dense.w,
                        step=cur_step)
                    tf.summary.histogram(
                        f'Epilogue/{layer.name}/adain/shift_bias',
                        layer.adain.style_shift_transform_bias.b,
                        step=cur_step)

        if plot_gradient:
            for g, v in zip(gradientsG, modelG.layers):
                tf.summary.histogram("GradientG/{}/grad_histogram".format(
                    v.name),
                                     g,
                                     step=cur_step)
                tf.summary.scalar("GradientG/{}/grad/sparsity".format(v.name),
                                  tf.reduce_sum(g),
                                  step=cur_step)

    with writer_2.as_default():
        tf.summary.scalar('Loss/Loss', loss_g, step=cur_step)
        tf.summary.scalar('Pred/Pred_D', fake_score, step=cur_step)


def images_to_tensorboard(writer,
                          cur_step,
                          fixed_noise,
                          real_images,
                          meshes,
                          modelG,
                          avg_param_G,
                          image_folder,
                          save_image_to_harddisk=False):
    ## Save image
    if cur_step % 1000 == 0:
        model_pred_fnoise = modelG([fixed_noise, meshes])

        backup_para = modelG.get_weights()
        modelG.set_weights(avg_param_G)

        avg_model_pred_fnoise = modelG([fixed_noise, meshes])
        modelG.set_weights(backup_para)

        all_imgs = tf.concat([
            tf.image.resize(real_images, (128, 128)),
            tf.image.resize(model_pred_fnoise, (128, 128)),
            tf.image.resize(avg_model_pred_fnoise, (128, 128))
        ],
                             axis=0)

        grid_all = imgrid((all_imgs + 1) * 0.5, 8)
        grid_pred = imgrid((avg_model_pred_fnoise + 1) * 0.5, 4)

        if save_image_to_harddisk:
            kutils.save_img(image_folder + '/%06d_all.jpg' % cur_step,
                            grid_all)
            kutils.save_img(image_folder + '/%06d_fix.jpg' % cur_step,
                            grid_pred)
        with writer.as_default():
            tf.summary.image("All images",
                             tf.expand_dims(grid_all, 0),
                             step=cur_step)
            tf.summary.image("Fixed noise",
                             tf.expand_dims(grid_pred, 0),
                             step=cur_step)
    elif cur_step % 500 == 0:
        backup_para = modelG.get_weights()
        modelG.set_weights(avg_param_G)
        avg_model_pred_fnoise = modelG([fixed_noise, meshes])
        grid_pred = imgrid((avg_model_pred_fnoise + 1) * 0.5, 4)
        if save_image_to_harddisk:
            kutils.save_img(image_folder + '/%06d_fix.jpg' % cur_step,
                            grid_pred)
        modelG.set_weights(backup_para)
        with writer.as_default():
            tf.summary.image("Fixed noise",
                             tf.expand_dims(grid_pred, 0),
                             step=cur_step)


def save_weights(modelG, avg_param_G, cur_step, total_iterations, manager,
                 model_folder):

    if cur_step % 5000 == 0 or cur_step == total_iterations:

        backup_para = modelG.get_weights()

        modelG.set_weights(avg_param_G)

        manager.save()

        tf.saved_model.save(modelG, model_folder + '/saved_model/')

        modelG.set_weights(backup_para)
