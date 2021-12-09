# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738
import tensorflow as tf
import tensorflow_addons as tfa


def DiffAugment(x, policy=''):
    if policy:
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
    return x


def rand_brightness(x):
    x = x + tf.random.uniform((x.shape[0], 1, 1, 1), dtype=x.dtype) - 0.5
    return x


def rand_saturation(x):
    x_mean = tf.reduce_mean(x, axis=1, keepdims=True)
    x = (x - x_mean) * (tf.random.uniform(
        (x.shape[0], 1, 1, 1), dtype=x.dtype) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * (tf.random.uniform(
        (x.shape[0], 1, 1, 1), dtype=x.dtype) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_factor = int(x.shape[1] * ratio + 0.5)
    trans_matrix = tf.random.uniform((x.shape[0], 2), -shift_factor,
                                     shift_factor)
    x = tfa.image.translate(x, trans_matrix)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.shape[2] * ratio + 0.5), int(x.shape[3] * ratio + 0.5)
    offset_x = tf.random.uniform([x.shape[0], 1, 1],
                                 0,
                                 x.shape[2] + (1 - cutout_size[0] % 2),
                                 dtype=tf.int64)
    offset_y = tf.random.uniform([x.shape[0], 1, 1],
                                 0,
                                 x.shape[3] + (1 - cutout_size[1] % 2),
                                 dtype=tf.int64)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(x.shape[0]),
                                             tf.range(cutout_size[0]),
                                             tf.range(cutout_size[1]))
    grid_x = tf.clip_by_value(grid_x + offset_x - cutout_size[0] // 2,
                              clip_value_min=0,
                              clip_value_max=x.shape[2] - 1)
    grid_y = tf.clip_by_value(grid_y + offset_y - cutout_size[1] // 2,
                              clip_value_min=0,
                              clip_value_max=x.shape[3] - 1)
    mask = tf.ones([x.shape[0], x.shape[2], x.shape[3]], dtype=x.dtype)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * tf.expand_dims(mask, 1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
