import math

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Conv2D
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.utils import get_custom_objects
from tensorflow.nn import depth_to_space
from tensorflow_addons.layers import SpectralNormalization


def conv2d(*args, **kwargs):
    return SpectralNormalization(
        layers.Conv2D(*args,
                      **kwargs,
                      kernel_initializer=tf.keras.initializers.RandomNormal(
                          0.0, 0.02)))


# def dense(*args, **kwargs):
#     return layers.Dense(*args, **kwargs,
#                         activation=tf.nn.leaky_relu)


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


def compute_runtime_coef(weight_shape, gain, lrmul):
    fan_in = tf.reduce_prod(
        weight_shape[:-1]
    )  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    fan_in = tf.cast(fan_in, dtype=tf.float32)
    he_std = gain / tf.sqrt(fan_in)
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
    return init_std, runtime_coef


class Dense(layers.Layer):

    def __init__(self, fmaps, gain=np.sqrt(2), lrmul=0.01, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        fan_in = tf.reduce_prod(input_shape[1:])
        weight_shape = [fan_in, self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(
            weight_shape, self.gain, self.lrmul)

        w_init = tf.random.normal(shape=weight_shape,
                                  mean=0.0,
                                  stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True, dtype='float32')

    def call(self, inputs, training=None, mask=None):
        weight = tf.cast(self.runtime_coef * self.w, self.compute_dtype)
        c = tf.reduce_prod(tf.shape(inputs)[1:])
        x = tf.reshape(inputs, shape=[-1, c])
        x = tf.matmul(x, weight)
        return x

    def get_config(self):
        base_config = super(Dense, self).get_config()
        config = {
            'fmaps': self.fmaps,
        }
        return dict(list(base_config.items()) + list(config.items()))


class BiasAct(layers.Layer):

    def __init__(self, lrmul, act, **kwargs):
        super(BiasAct, self).__init__(**kwargs)
        assert act in ['linear', 'lrelu']
        self.lrmul = lrmul

        if act == 'linear':
            self.act = tf.keras.layers.Lambda(lambda x: tf.identity(x))
            self.gain = 1.0
        else:
            self.act = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.gain = np.sqrt(2)

    def build(self, input_shape):
        self.len2 = True if len(input_shape) == 2 else False
        b_init = tf.zeros(shape=(input_shape[1], ), dtype=tf.dtypes.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.lrmul * self.b

        if self.len2:
            x = inputs + tf.cast(b, self.compute_dtype)
        else:
            x = inputs + tf.reshape(b, shape=[1, 1, 1, -1])
        x = self.act(x)
        x = self.gain * x
        return x

    def get_config(self):
        config = super(BiasAct, self).get_config()
        config.update({
            'lrmul': self.lrmul,
            'gain': self.gain,
            'len2': self.len2,
        })
        return config


class MappingTFDense(layers.Layer):

    def __init__(self, n_mapping, w_dim=256, **kwargs):
        super(MappingTFDense, self).__init__(**kwargs)
        self.w_dim = w_dim
        self.n_mapping = n_mapping

        self.normalize = tf.keras.layers.Lambda(lambda x: x * tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8))

        self.dense_layers = list()
        for _ in range(self.n_mapping):
            self.dense_layers.append(
                layers.Dense(w_dim, activation=tf.nn.leaky_relu))

    def call(self, inputs):
        x = inputs

        # normalize inputs
        x = self.normalize(x)

        # apply mapping blocks
        for dense in self.dense_layers:
            x = dense(x)

            # x = InstanceNormalization(-1)(x)

        return x

    def get_config(self):
        config = super(MappingTFDense, self).get_config()
        config.update({
            'w_dim': self.w_dim,
            'n_mapping': self.n_mapping,
        })
        return config


class Mapping(layers.Layer):

    def __init__(self, n_mapping, w_dim=256, **kwargs):
        super(Mapping, self).__init__(**kwargs)
        self.w_dim = w_dim
        self.n_mapping = n_mapping
        self.gain = np.sqrt(2)
        self.lrmul = 0.01

        self.normalize = tf.keras.layers.Lambda(lambda x: x * tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8))

        self.dense_layers = list()
        self.bias_act_layers = list()
        for ii in range(self.n_mapping):
            self.dense_layers.append(
                Dense(w_dim,
                      gain=self.gain,
                      lrmul=self.lrmul,
                      name='dense_{:d}'.format(ii)))
            self.bias_act_layers.append(
                BiasAct(lrmul=self.lrmul,
                        act='lrelu',
                        name='bias_{:d}'.format(ii)))

    def call(self, inputs, training=None, mask=None):
        latents = inputs
        x = latents

        # normalize inputs
        x = self.normalize(x)

        # apply mapping blocks
        for dense, apply_bias_act in zip(self.dense_layers,
                                         self.bias_act_layers):
            x = dense(x)
            x = apply_bias_act(x)

        x = InstanceNormalization(axis=-1)(x)

        return x

    def get_config(self):
        config = super(Mapping, self).get_config()
        config.update({
            'w_dim': self.w_dim,
            'n_mapping': self.n_mapping,
        })
        return config


class AdaIN(layers.Layer):

    def __init__(self, channels, w_dim=256, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.channels = channels
        self.w_dim = w_dim

        self.instance_norm = InstanceNormalization(-1)

        # self.style_scale_transform = layers.Dense(channels)
        # self.style_shift_transform = layers.Dense(channels)
        self.style_scale_transform_dense = Dense(channels,
                                                 gain=np.sqrt(2),
                                                 lrmul=0.01)
        self.style_scale_transform_bias = BiasAct(lrmul=0.01, act='linear')
        self.style_shift_transform_dense = Dense(channels,
                                                 gain=np.sqrt(2),
                                                 lrmul=0.01)
        self.style_shift_transform_bias = BiasAct(lrmul=0.01, act='linear')

    def build(self, input_shape):
        pass

    def call(self, x, w):
        normalized_x = self.instance_norm(x)
        # style_scale = self.style_scale_transform(w)[:, None, None, :]
        # style_shift = self.style_shift_transform(w)[:, None, None, :]
        style_scale = self.style_scale_transform_bias(
            self.style_scale_transform_dense(w))[:, None, None, :]
        style_shift = self.style_shift_transform_bias(
            self.style_shift_transform_dense(w))[:, None, None, :]

        transformed_x = style_scale * normalized_x + style_shift

        return transformed_x


class LayerEpilogue(layers.Layer):

    def __init__(self, **kwargs):
        super(LayerEpilogue, self).__init__(**kwargs)

    def build(self, input_shape):
        self.inject_noise = InjectNoise()
        self.adain = AdaIN(input_shape[-1])
        self.activation = layers.LeakyReLU(alpha=0.2)

    def call(self, x, w):
        x = self.inject_noise(x)
        x = self.adain(x, w)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(LayerEpilogue, self).get_config()
        config.update({})
        return config


class ConstLayer(layers.Layer):

    def __init__(self, n_const_fmap, **kwargs):
        super(ConstLayer, self).__init__(**kwargs)
        self.n_const_fmap = n_const_fmap

    def build(self, input_shape):
        self.const = tf.Variable(tf.ones((1, 4, 4, self.n_const_fmap)),
                                 dtype='float32',
                                 trainable=True)
        self.conv = layers.Conv2D(input_shape[-1],
                                  3,
                                  padding='same',
                                  use_bias=False)
        self.layer_epilogue = LayerEpilogue()

    def call(self, w):
        const = tf.tile(self.const, [tf.shape(w)[0], 1, 1, 1])
        const = self.conv(const)
        x = self.layer_epilogue(const, w)
        return x

    def get_config(self):
        config = super(ConstLayer, self).get_config()
        config.update({
            'n_const_fmap': self.n_const_fmap,
        })
        return config


class StyleModulation(layers.Layer):

    def __init__(self, **kwargs):
        super(StyleModulation, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(input_shape[-1] * 2)
        self.bias_act = BiasAct(lrmul=0.01, act='lrelu')
        # self.dense = dense(input_shape[-1] * 2)

    def call(self, x, latten):
        style = self.bias_act(self.dense(latten))
        # style = self.dense(latten)
        style = tf.reshape(style,
                           [-1, 2] + [1] * (len(x.shape) - 2) + [x.shape[-1]])
        return x * (style[:, 0] + 1) + style[:, 1]


class InstanceNormalization(layers.Layer):

    def __init__(self, axis=[1, 2], **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        pass

    def call(self, x, epsilon=1e-8):
        x -= tf.reduce_mean(x, axis=self.axis, keepdims=True)
        x *= tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=self.axis, keepdims=True) +
            epsilon)
        return x

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({
            'axis': self.axis,
        })
        return config


class PixelNormalization(layers.Layer):

    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, epsilon=1e-8):
        return x * tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)


class NoiseInjection(layers.Layer):

    def __init__(self, **kwargs):
        super(NoiseInjection, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = tf.Variable(tf.zeros(input_shape[-1], dtype='float32'),
                             dtype='float32',
                             trainable=True)

    def call(self, feat, noise=None):
        if noise is None:
            _, height, width, _ = feat.shape
            noise = tf.random.normal((height, width, 1),
                                     dtype=self.compute_dtype)
        return feat + tf.cast(self.w, self.compute_dtype) * noise


class InjectNoise(layers.Layer):

    def __init__(self, **kwargs):
        super(InjectNoise, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.random.normal(shape=(1, 1, 1, input_shape[-1]),
                                  dtype=tf.dtypes.float32)
        self.w = tf.Variable(w_init, trainable=True, name='w')

    def call(self, inputs, noise=None):
        x_shape = tf.shape(inputs)

        # noise: [1, x_shape[1], x_shape[2], 1] or None
        if noise is None:
            noise = tf.random.normal(shape=(x_shape[0], x_shape[1], x_shape[2],
                                            1),
                                     dtype=tf.dtypes.float32)

        x = inputs + tf.cast(noise * self.w, dtype=inputs.dtype)
        return x

    def get_config(self):
        config = super(InjectNoise, self).get_config()
        config.update({})
        return config


class Noise(layers.Layer):

    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.zeros(shape=(), dtype=tf.dtypes.float32)
        self.noise_strength = tf.Variable(w_init, trainable=True, name='w')

    def call(self, inputs, noise=None, training=None, mask=None):
        x_shape = tf.shape(inputs)

        # noise: [1, x_shape[1], x_shape[2], 1] or None
        if noise is None:
            noise = tf.random.normal(shape=(x_shape[0], x_shape[1], x_shape[2],
                                            1),
                                     dtype=tf.dtypes.float32)

        x = inputs + tf.cast(noise * self.noise_strength, dtype=inputs.dtype)
        return x

    def get_config(self):
        config = super(Noise, self).get_config()
        config.update({})
        return config


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
    'InstanceNormalization': InstanceNormalization,
    'StyleModulation': StyleModulation,
    'PixelNormalization': PixelNormalization,
    'ConstLayer': ConstLayer,
    'Dense': Dense,
    'Mapping': Mapping,
    'MappingTFDense': MappingTFDense,
    'BiasAct': BiasAct,
    'Noise': Noise,
    'AdaIN': AdaIN,
    'LayerEpilogue': LayerEpilogue,
    'InjectNoise': InjectNoise,
})


def swish(feat):
    return feat * tf.math.sigmoid(feat)


def glu(x):
    nc = x.shape[-1]
    assert nc % 2 == 0, 'channels is not divisible by 2!'
    nc = int(nc / 2)
    return x[:, :, :, :nc] * tf.math.sigmoid(x[:, :, :, nc:])


def gelu(x, approximate=False):
    return tf.nn.gelu(x, approximate=approximate)


def init_layer(noise, channel):
    noise = layers.Reshape((1, 1, -1))(noise)
    noise = convTranspose2d(channel * 2, 4, 1, use_bias=False)(noise)
    noise = batchNorm2d()(noise)
    # noise = InstanceNormalization()(noise)
    noise = glu(noise)
    return noise


def leaky_relu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=0.2)


def latten_mapping(x, n_map_layers=8):
    x = PixelNormalization()(x)
    for _ in range(n_map_layers):
        # x = dense(x.shape[-1], gain=np.sqrt(2))(x)
        x = dense(x.shape[-1])(x)
        x = batchNorm2d()(x)
        # x = leaky_relu(x)
    return x


def layer_epilogue(x,
                   latten,
                   use_noise=False,
                   use_instance_norm=True,
                   use_pixel_norm=False):
    if use_noise:
        x = Noise()(x)
    x = leaky_relu(x)
    if use_pixel_norm:
        x = PixelNormalization()(x)
    if use_instance_norm:
        x = InstanceNormalization()(x)
    x = StyleModulation()(x, latten)
    return x


def se_block(feat_small, feat_big, ch_out):
    feat_small = tfa.layers.AdaptiveAveragePooling2D(4)(feat_small)
    feat_small = conv2d(ch_out, 4, 1, use_bias=False)(feat_small)
    feat_small = swish(feat_small)
    feat_small = conv2d(ch_out, 1, 1, use_bias=False)(feat_small)
    feat_small = layers.Activation('sigmoid')(feat_small)
    return feat_big * feat_small


def up_block(x, out_planes, latten=None):
    x = layers.UpSampling2D()(x)
    x = conv2d(out_planes * 2, 3, 1, 'same', use_bias=False)(x)
    x = batchNorm2d()(x)
    x = glu(x)
    if latten is not None:
        x = LayerEpilogue()(x, latten)
    return x


def up_block_comp(x, out_planes, latten=None):
    x = layers.UpSampling2D()(x)
    x = conv2d(out_planes * 2, 3, 1, 'same', use_bias=False)(x)
    x = batchNorm2d()(x)
    x = glu(x)
    if latten is not None:
        x = LayerEpilogue()(x, latten)
    else:
        x = NoiseInjection()(x)
    x = conv2d(out_planes * 2, 3, 1, 'same', use_bias=False)(x)
    x = batchNorm2d()(x)
    x = glu(x)
    if latten is not None:
        x = LayerEpilogue()(x, latten)
    else:
        x = NoiseInjection()(x)
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
    rf_0 = layers.Flatten()(rf_0)
    rf_0 = layers.Dense(1)(rf_0)

    model = Model(inputs=imgs_big, outputs=rf_0)
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
    # latten = x

    # latten = latten_mapping(x, 4)
    latten = Mapping(8)(x)

    # feat_4 = init_layer(latten, channel=nfc[4])

    feat_4 = ConstLayer(n_const_fmap=256)(x)

    feat_8 = up_block_comp(feat_4, nfc[8], latten)
    feat_16 = up_block(feat_8, nfc[16], latten)
    feat_32 = up_block_comp(feat_16, nfc[32], latten)

    feat_64 = se_block(feat_4, up_block(feat_32, nfc[64], latten), nfc[64])
    feat_128 = se_block(feat_8, up_block_comp(feat_64, nfc[128], latten),
                        nfc[128])
    feat_256 = se_block(feat_16, up_block(feat_128, nfc[256], latten),
                        nfc[256])

    to_big = conv2d(nc, 3, 1, 'same', use_bias=False)

    if im_size == 256:
        img_256 = to_big(feat_256)
        img_256 = layers.Activation('tanh')(img_256)
        return Model(inputs=x, outputs=img_256)

    feat_512 = se_block(feat_32, up_block_comp(feat_256, nfc[512], latten),
                        nfc[512])
    if im_size == 512:
        img_512 = to_big(feat_512)
        img_512 = layers.Activation('tanh')(img_512)
        return Model(inputs=x, outputs=img_512)

    feat_1024 = up_block(feat_512, nfc[1024], latten)

    img_1024 = to_big(feat_1024)
    img_1024 = layers.Activation('tanh')(img_1024)

    model = Model(inputs=x, outputs=img_1024)
    return model


if __name__ == "__main__":
    # disc = Discriminator()
    gen = Generator()
    for layer in gen.layers:
        print(layer.name)
    # gen.summary()
    # disc.summary()
    # gen.save('gen.h5')
    # tf.keras.models.load_model('gen.h5').summary()
