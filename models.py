import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow_addons.layers import SpectralNormalization

from operation import crop_image_by_part

# def conv2d(*args, **kwargs):
#     return SpectralNormalization(
#         layers.Conv2D(*args,
#                       **kwargs))


def conv2d(*args, **kwargs):
    return SpectralNormalization(
        layers.Conv2D(*args,
                      **kwargs,
                      kernel_initializer=tf.keras.initializers.RandomNormal(
                          0.0, 0.02)))


def convTranspose2d(*args, **kwargs):
    return SpectralNormalization(
        layers.Conv2DTranspose(
            *args,
            **kwargs,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))


# def batchNorm2d(*args, **kwargs):
#     return layers.BatchNormalization(
#         *args,
#         **kwargs)


def batchNorm2d(*args, **kwargs):
    return layers.BatchNormalization(
        *args,
        **kwargs,
        gamma_initializer=tf.keras.initializers.RandomNormal(1.0, 0.02))


# def linear(*args, **kwargs):
#     return SpectralNormalization(layers.Dense(*args, **kwargs))

# class PixelNorm(layers.Layer):
#     def call(self, input):
#         return input * tf.math.rsqrt(
#             tf.reduce_mean(input**2, axis=-1, keepdims=True) + 1e-8)


def GLU(x):
    nc = x.shape[-1]
    assert nc % 2 == 0, 'channels is not divisible by 2!'
    nc = int(nc / 2)
    return x[:, :, :, :nc] * tf.math.sigmoid(x[:, :, :, nc:])


class NoiseInjection(layers.Layer):
    def __init__(self):
        super().__init__()
        self.weight = tf.Variable(tf.zeros(1, dtype='float32'),
                                  dtype='float32',
                                  trainable=True)

    def call(self, feat, noise=None):
        if noise is None:
            _, height, width, _ = feat.shape
            noise = tf.random.normal((height, width, 1), dtype='float32')

        return feat + self.weight * noise


def Swish(feat):
    return feat * tf.math.sigmoid(feat)


def SEBlock(feat_small, feat_big, ch_out):
    feat_small = tfa.layers.AdaptiveAveragePooling2D(4)(feat_small)
    feat_small = conv2d(ch_out, 4, 1, use_bias=False)(feat_small)
    feat_small = Swish(feat_small)
    feat_small = conv2d(ch_out, 1, 1, use_bias=False)(feat_small)
    feat_small = layers.Activation('sigmoid')(feat_small)
    return feat_big * feat_small


def InitLayer(noise, channel):
    noise = layers.Reshape((1, 1, -1))(noise)
    noise = convTranspose2d(channel * 2, 4, 1, use_bias=False)(noise)
    noise = batchNorm2d()(noise)
    noise = GLU(noise)
    return noise


def UpBlock(x, out_planes):
    x = layers.UpSampling2D()(x)
    x = conv2d(out_planes * 2, 3, 1, 'same', use_bias=False)(x)
    x = batchNorm2d()(x)
    x = GLU(x)
    return x


def UpBlockComp(x, out_planes):
    x = layers.UpSampling2D()(x)
    x = conv2d(out_planes * 2, 3, 1, 'same', use_bias=False)(x)
    x = NoiseInjection()(x)
    x = batchNorm2d()(x)
    x = GLU(x)
    x = conv2d(out_planes * 2, 3, 1, 'same', use_bias=False)(x)
    x = NoiseInjection()(x)
    x = batchNorm2d()(x)
    x = GLU(x)
    return x


def DownBlock(x, out_planes):
    x = conv2d(out_planes, 4, 2, 'same', use_bias=False)(x)
    x = batchNorm2d()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x


def DownBlockComp(x, out_planes):
    main_path = conv2d(out_planes, 4, 2, 'same', use_bias=False)(x)
    main_path = batchNorm2d()(main_path)
    main_path = layers.LeakyReLU(0.2)(main_path)
    main_path = conv2d(out_planes, 3, 1, 'same', use_bias=False)(main_path)
    main_path = batchNorm2d()(main_path)
    main_path = layers.LeakyReLU(0.2)(main_path)

    direct_path = layers.AveragePooling2D(2, 2)(x)
    direct_path = conv2d(out_planes, 1, 1, use_bias=False)(direct_path)
    direct_path = batchNorm2d()(direct_path)
    direct_path = layers.LeakyReLU(0.2)(direct_path)
    return (main_path + direct_path) / 2


def SimpleDecoder(x, nfc_in=64, nc=3):
    nfc_multi = {
        4: 16,
        8: 8,
        16: 4,
        32: 2,
        64: 2,
        128: 1,
        256: 0.5,
        512: 0.25,
        1024: 0.125
    }
    nfc = {}
    for k, v in nfc_multi.items():
        nfc[k] = int(v * 32)

    x = tfa.layers.AdaptiveAveragePooling2D(8)(x)
    x = UpBlock(x, nfc[16])
    x = UpBlock(x, nfc[32])
    x = UpBlock(x, nfc[64])
    x = UpBlock(x, nfc[128])
    x = conv2d(nc, 3, 1, 'same', use_bias=False)(x)
    x = layers.Activation('tanh')(x)
    return x


def Discriminator(ndf=64, nc=3, im_size=512):
    imgs_big = keras.Input(shape=(im_size, im_size, 3))
    imgs_small = keras.Input(shape=(128, 128, 3))

    nfc_multi = {
        4: 16,
        8: 16,
        16: 8,
        32: 4,
        64: 2,
        128: 1,
        256: 0.5,
        512: 0.25,
        1024: 0.125
    }
    nfc = {}
    for k, v in nfc_multi.items():
        nfc[k] = int(v * ndf)

    if im_size == 1024:
        feat_2 = conv2d(nfc[1024], 4, 2, 'same', use_bias=False)(imgs_big)
        feat_2 = layers.LeakyReLU(0.2)(feat_2)
        feat_2 = conv2d(nfc[512], 4, 2, 'same', use_bias=False)(feat_2)
        feat_2 = batchNorm2d()(feat_2)
        feat_2 = layers.LeakyReLU(0.2)(feat_2)
    elif im_size == 512:
        feat_2 = conv2d(nfc[512], 4, 2, 'same', use_bias=False)(imgs_big)
        feat_2 = layers.LeakyReLU(0.2)(feat_2)
    elif im_size == 256:
        feat_2 = conv2d(nfc[512], 3, 1, 'same', use_bias=False)(imgs_big)
        feat_2 = layers.LeakyReLU(0.2)(feat_2)

    feat_4 = DownBlockComp(feat_2, nfc[256])
    feat_8 = DownBlockComp(feat_4, nfc[128])

    feat_16 = DownBlockComp(feat_8, nfc[64])
    feat_16 = SEBlock(feat_2, feat_16, nfc[64])

    feat_32 = DownBlockComp(feat_16, nfc[32])
    feat_32 = SEBlock(feat_4, feat_32, nfc[32])

    feat_last = DownBlockComp(feat_32, nfc[16])
    feat_last = SEBlock(feat_8, feat_last, nfc[16])

    rf_0 = conv2d(nfc[8], 1, 1, use_bias=False)(feat_last)
    rf_0 = batchNorm2d()(rf_0)
    rf_0 = layers.LeakyReLU(0.2)(rf_0)
    rf_0 = conv2d(1, 4, 1, use_bias=False)(rf_0)
    rf_0 = tf.reshape(rf_0, [-1])

    feat_small = conv2d(nfc[256], 4, 2, 'same', use_bias=False)(imgs_small)
    feat_small = layers.LeakyReLU(0.2)(feat_small)
    feat_small = DownBlock(feat_small, nfc[128])
    feat_small = DownBlock(feat_small, nfc[64])
    feat_small = DownBlock(feat_small, nfc[32])
    rf_1 = conv2d(1, 4, 1, use_bias=False)(feat_small)
    rf_1 = tf.reshape(rf_1, [-1])

    rec_img_big = SimpleDecoder(feat_last, nfc[16], nc)
    rec_img_small = SimpleDecoder(feat_small, nfc[32], nc)

    # croped_feat_32 = layers.CenterCrop(8, 8)(feat_32) # [N, 16, 16, 3]
    # rec_img_part = SimpleDecoder(croped_feat_32, nfc[32], nc)
    rec_img_part = SimpleDecoder(feat_32, nfc[32], nc)

    model = Model(inputs=[imgs_big, imgs_small],
                  outputs=[
                      tf.concat([rf_0, rf_1], 0), rec_img_big, rec_img_small,
                      rec_img_part
                  ])
    return model


def Generator(input_shape=(256, ), ngf=64, nc=3, im_size=1024):
    nfc_multi = {
        4: 16,
        8: 8,
        16: 4,
        32: 2,
        64: 2,
        128: 1,
        256: 0.5,
        512: 0.25,
        1024: 0.125
    }
    nfc = {}
    for k, v in nfc_multi.items():
        nfc[k] = int(v * ngf)

    x = keras.Input(shape=input_shape)
    feat_4 = InitLayer(x, channel=nfc[4])
    feat_8 = UpBlockComp(feat_4, nfc[8])
    feat_16 = UpBlock(feat_8, nfc[16])
    feat_32 = UpBlockComp(feat_16, nfc[32])

    feat_64 = SEBlock(feat_4, UpBlock(feat_32, nfc[64]), nfc[64])
    feat_128 = SEBlock(feat_8, UpBlockComp(feat_64, nfc[128]), nfc[128])
    feat_256 = SEBlock(feat_16, UpBlock(feat_128, nfc[256]), nfc[256])

    img_128 = conv2d(nc, 1, 1, use_bias=False)(feat_128)
    img_128 = layers.Activation('tanh')(img_128)

    to_big = conv2d(nc, 3, 1, 'same', use_bias=False)

    if im_size == 256:
        img_256 = to_big(feat_256)
        img_256 = layers.Activation('tanh')(img_256)
        return Model(inputs=x, outputs=[img_256, img_128])

    feat_512 = SEBlock(feat_32, UpBlockComp(feat_256, nfc[512]), nfc[512])
    if im_size == 512:
        img_512 = to_big(feat_512)
        img_512 = layers.Activation('tanh')(img_512)
        return Model(inputs=x, outputs=[img_512, img_128])

    feat_1024 = UpBlock(feat_512, nfc[1024])

    img_1024 = to_big(feat_1024)
    img_1024 = layers.Activation('tanh')(img_1024)

    model = Model(inputs=x, outputs=[img_1024, img_128])
    return model
