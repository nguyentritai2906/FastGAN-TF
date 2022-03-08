import json
import logging
import os
import shutil
import sys
import time

import mediapipe as mp
import tensorflow as tf


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

    return tf.reshape(landmarks, [468, 3])


def imgs_to_landmarks(imgs):
    imgs = tf.cast(imgs, tf.uint8)
    imgs = imgs.numpy()
    landmarks_arrays = []
    for img in imgs:
        landmarks = generate_landmarks(img)
        landmarks_arrays.append(landmarks)
    return tf.convert_to_tensor(landmarks_arrays)


def cal_mse_landmarks(landmarks_gt, landmarks_pred):
    return tf.reduce_mean(tf.square(landmarks_gt - landmarks_pred))
