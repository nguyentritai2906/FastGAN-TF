import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Conv2D
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import get_custom_objects
from tensorflow.nn import depth_to_space
from tensorflow_addons.layers import SpectralNormalization


def conv2d(*args, **kwargs):
    return SpectralNormalization(
        layers.Conv2D(*args,
                      **kwargs,
                      kernel_initializer=tf.keras.initializers.RandomNormal(
                          0.0, 0.02)))


def dense(*args, **kwargs):
    return SpectralNormalization(
        layers.Dense(*args,
                     **kwargs,
                     kernel_initializer=tf.keras.initializers.RandomNormal(
                         0.0, 0.02)))


def convTranspose2d(*args, **kwargs):
    return SpectralNormalization(
        layers.Conv2DTranspose(
            *args,
            **kwargs,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))


def batchNorm2d(*args, **kwargs):
    return layers.BatchNormalization(
        *args,
        **kwargs,
        gamma_initializer=tf.keras.initializers.RandomNormal(1.0, 0.02))


def glu(x):
    nc = x.shape[-1]
    assert nc % 2 == 0, 'channels is not divisible by 2!'
    nc = int(nc / 2)
    return x[:, :, :, :nc] * tf.math.sigmoid(x[:, :, :, nc:])


class NoiseInjection(layers.Layer):

    def __init__(self, **kwargs):
        super(NoiseInjection, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = tf.Variable(tf.zeros(1, dtype='float32'),
                             dtype='float32',
                             trainable=True)

    def call(self, feat, noise=None):
        if noise is None:
            _, height, width, _ = feat.shape
            noise = tf.random.normal((height, width, 1),
                                     dtype=self.compute_dtype)

        return feat + tf.cast(self.w, self.compute_dtype) * noise


class SubpixelConv2D(Conv2D):

    def __init__(self,
                 upsampling_factor,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SubpixelConv2D, self).__init__(
            filters=filters * upsampling_factor * upsampling_factor,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        super(SubpixelConv2D, self).build(input_shape)
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim % (factor) != 0:
            raise ValueError('Channel ' + str(last_dim) + ' should be of '
                             'integer times of upsampling_factor^2: ' +
                             str(factor) + '.')

    def call(self, inputs, **kwargs):
        return depth_to_space(
            super(SubpixelConv2D, self).call(inputs), self.upsampling_factor)

    def get_config(self):
        base_config = super(SubpixelConv2D, self).get_config()
        base_config[
            'filters'] /= self.upsampling_factor * self.upsampling_factor
        config = {'upsampling_factor': self.upsampling_factor}
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({
    'SubpixelConv2D': SubpixelConv2D,
    'NoiseInjection': NoiseInjection,
})


def swish(feat):
    return feat * tf.math.sigmoid(feat)


def init_layer(noise, channel):
    noise = layers.Reshape((1, 1, -1))(noise)
    noise = convTranspose2d(channel * 2, 4, 1, use_bias=False)(noise)
    noise = batchNorm2d()(noise)
    noise = glu(noise)
    return noise


def se_block(feat_small, feat_big, ch_out):
    feat_small = tfa.layers.AdaptiveAveragePooling2D(4)(feat_small)
    feat_small = conv2d(ch_out, 4, 1, use_bias=False)(feat_small)
    feat_small = swish(feat_small)
    feat_small = conv2d(ch_out, 1, 1, use_bias=False)(feat_small)
    feat_small = layers.Activation('sigmoid')(feat_small)
    return feat_big * feat_small


def up_block(x, out_planes):
    x = layers.UpSampling2D()(x)
    x = conv2d(out_planes * 2, 3, 1, 'same', use_bias=False)(x)
    x = batchNorm2d()(x)
    x = glu(x)
    return x


def up_block_comp(x, out_planes):
    x = layers.UpSampling2D()(x)
    x = conv2d(out_planes * 2, 3, 1, 'same', use_bias=False)(x)
    x = NoiseInjection()(x)
    x = batchNorm2d()(x)
    x = glu(x)
    x = conv2d(out_planes * 2, 3, 1, 'same', use_bias=False)(x)
    x = NoiseInjection()(x)
    x = batchNorm2d()(x)
    x = glu(x)
    return x


def down_block(x, out_planes):
    x = conv2d(out_planes, 4, 2, 'same', use_bias=False)(x)
    x = batchNorm2d()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x


def down_block_comp(x, out_planes):
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


def SimpleDecoder(x, nfc_in=32, nc=3):
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
        nfc[k] = int(v * nfc_in)

    x = up_block(x, nfc[16])
    x = up_block(x, nfc[32])
    x = up_block(x, nfc[64])
    x = up_block(x, nfc[128])
    x = conv2d(nc, 3, 1, 'same', use_bias=False)(x)
    x = layers.Activation('tanh')(x)
    return x


def Discriminator(ndf=64, nc=3, im_size=256):
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
        feat_2 = conv2d(nfc[512], 4, 1, 'same', use_bias=False)(imgs_big)
        feat_2 = layers.LeakyReLU(0.2)(feat_2)

    feat_4 = down_block_comp(feat_2, nfc[256])
    feat_8 = down_block_comp(feat_4, nfc[128])

    feat_16 = down_block_comp(feat_8, nfc[64])
    feat_16 = se_block(feat_2, feat_16, nfc[64])

    feat_32 = down_block_comp(feat_16, nfc[32])
    feat_32 = se_block(feat_4, feat_32, nfc[32])

    feat_last = down_block_comp(feat_32, nfc[16])
    feat_last = se_block(feat_8, feat_last, nfc[16])

    rf_0 = down_block_comp(feat_last, nfc[8])
    rf_0 = conv2d(1, 4, 1, use_bias=False)(rf_0)
    rf_0 = tf.reshape(rf_0, [-1])

    feat_small = down_block(imgs_small, nfc[256])
    feat_small = down_block(feat_small, nfc[128])
    feat_small = down_block(feat_small, nfc[64])
    feat_small = down_block(feat_small, nfc[32])

    rf_1 = down_block(feat_small, nfc[16])
    rf_1 = conv2d(1, 4, 1, use_bias=False)(rf_1)
    rf_1 = tf.reshape(rf_1, [-1])

    rec_img_big = SimpleDecoder(feat_last, nfc[128], nc)
    rec_img_small = SimpleDecoder(feat_small, nfc[256], nc)
    rec_img_part = SimpleDecoder(feat_32, nfc[256], nc)

    model = Model(inputs=[imgs_big, imgs_small],
                  outputs=[
                      tf.concat([rf_0, rf_1], 0), rec_img_big, rec_img_small,
                      rec_img_part
                  ])
    return model


def Generator(input_shape=(256, ), ngf=64, nc=3, im_size=256):
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
    feat_4 = init_layer(x, channel=nfc[4])
    feat_8 = up_block_comp(feat_4, nfc[8])
    feat_16 = up_block(feat_8, nfc[16])
    feat_32 = up_block_comp(feat_16, nfc[32])

    feat_64 = se_block(feat_4, up_block(feat_32, nfc[64]), nfc[64])
    feat_128 = se_block(feat_8, up_block_comp(feat_64, nfc[128]), nfc[128])
    feat_256 = se_block(feat_16, up_block(feat_128, nfc[256]), nfc[256])

    img_128 = conv2d(nc, 1, 1, use_bias=False)(feat_128)
    img_128 = layers.Activation('tanh')(img_128)

    to_big = conv2d(nc, 3, 1, 'same', use_bias=False)

    if im_size == 256:
        img_256 = to_big(feat_256)
        img_256 = layers.Activation('tanh')(img_256)
        return Model(inputs=x, outputs=[img_256, img_128])

    feat_512 = se_block(feat_32, up_block_comp(feat_256, nfc[512]), nfc[512])
    if im_size == 512:
        img_512 = to_big(feat_512)
        img_512 = layers.Activation('tanh')(img_512)
        return Model(inputs=x, outputs=[img_512, img_128])

    feat_1024 = up_block(feat_512, nfc[1024])

    img_1024 = to_big(feat_1024)
    img_1024 = layers.Activation('tanh')(img_1024)

    model = Model(inputs=x, outputs=[img_1024, img_128])
    return model


if __name__ == "__main__":
    disc = Discriminator()
    gen = Generator()
    gen.summary()
    disc.summary()
