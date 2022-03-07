import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Conv2D
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import get_custom_objects
from tensorflow.nn import depth_to_space
from tensorflow_addons.layers import SpectralNormalization


class Dense(layers.Layer):

    def __init__(self, fmaps, gain=np.sqrt(2), lrmul=0.01, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        fan_in = tf.reduce_prod(input_shape[1:])
        weight_shape = [fan_in, self.fmaps]
        init_std, self.runtime_coef = self.compute_runtime_coef(
            weight_shape, self.gain, self.lrmul)

        w_init = tf.random.normal(shape=weight_shape,
                                  mean=0.0,
                                  stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True, dtype='float32')

    def call(self, inputs):
        weight = tf.cast(self.runtime_coef * self.w, self.compute_dtype)
        c = tf.reduce_prod(tf.shape(inputs)[1:])
        x = tf.reshape(inputs, shape=[-1, c])
        x = tf.matmul(x, weight)
        return x

    def compute_runtime_coef(self, weight_shape, gain, lrmul):
        fan_in = tf.reduce_prod(
            weight_shape[:-1]
        )  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
        fan_in = tf.cast(fan_in, dtype=tf.float32)
        he_std = gain / tf.sqrt(fan_in)
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
        return init_std, runtime_coef

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

        self.mesh_dense = Dense(w_dim,
                                gain=self.gain,
                                lrmul=self.lrmul,
                                name='mesh_dense')
        self.mesh_bias = BiasAct(lrmul=self.lrmul,
                                 act='linear',
                                 name='mesh_bias')
        self.labels_embedding = LabelEmbedding(embed_dim=self.w_dim,
                                               name='labels_embedding')
        self.y_dense = Dense(w_dim,
                             gain=self.gain,
                             lrmul=self.lrmul,
                             name='y_dense')
        self.y_bias = BiasAct(lrmul=self.lrmul, act='linear', name='y_bias')

    def call(self, x, y, training=None, mask=None):
        y = self.y_dense(y)
        y = self.y_bias(y)

        # normalize inputs
        x = self.normalize(x)

        # apply mapping blocks
        for dense, apply_bias_act in zip(self.dense_layers,
                                         self.bias_act_layers):
            x = dense(x)
            x = apply_bias_act(x)

        # x = InstanceNormalization(axis=-1)(x)
        x = tf.concat([x, y], axis=-1)

        return x

    def get_config(self):
        config = super(Mapping, self).get_config()
        config.update({
            'w_dim': self.w_dim,
            'n_mapping': self.n_mapping,
        })
        return config


class LabelEmbedding(layers.Layer):

    def __init__(self, embed_dim, **kwargs):
        super(LabelEmbedding, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        weight_shape = [input_shape[1] * input_shape[2], self.embed_dim]
        # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        inputs = layers.Flatten()(inputs)
        x = tf.matmul(inputs, tf.cast(self.w, self.compute_dtype))
        return x

    def get_config(self):
        config = super(LabelEmbedding, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config


class AdaIN(layers.Layer):

    def __init__(self, channels, w_dim=256, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.channels = channels
        self.w_dim = w_dim

    def build(self, input_shape):
        self.instance_norm = InstanceNormalization(-1)

        self.style_scale_transform_dense = Dense(self.channels,
                                                 gain=np.sqrt(2),
                                                 lrmul=0.01)
        self.style_scale_transform_bias = BiasAct(lrmul=0.01, act='linear')
        self.style_shift_transform_dense = Dense(self.channels,
                                                 gain=np.sqrt(2),
                                                 lrmul=0.01)
        self.style_shift_transform_bias = BiasAct(lrmul=0.01, act='linear')

    def call(self, x, w):
        normalized_x = self.instance_norm(x)
        style_scale = self.style_scale_transform_dense(w)
        style_scale = self.style_scale_transform_bias(style_scale)[:, None,
                                                                   None, :]
        style_shift = self.style_shift_transform_dense(w)
        style_shift = self.style_shift_transform_bias(style_shift)[:, None,
                                                                   None, :]

        transformed_x = style_scale * normalized_x + style_shift

        return transformed_x


class LayerEpilogue(layers.Layer):

    def __init__(self, **kwargs):
        super(LayerEpilogue, self).__init__(**kwargs)

    def build(self, input_shape):
        self.inject_noise = PerPixelNoiseInjection()
        self.adain = AdaIN(input_shape[-1])
        self.activation = layers.LeakyReLU(alpha=0.2)

    def call(self, x, w):
        # x = self.inject_noise(x)
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


class PerChannelNoiseInjection(layers.Layer):

    def __init__(self, **kwargs):
        super(PerChannelNoiseInjection, self).__init__(**kwargs)

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


class PerPixelNoiseInjection(layers.Layer):

    def __init__(self, **kwargs):
        super(PerPixelNoiseInjection, self).__init__(**kwargs)

    def build(self, input_shape):
        # w_init = tf.random.normal(shape=(1, 1, 1, input_shape[-1]),
        #                           dtype=tf.dtypes.float32)
        w_init = tf.zeros(shape=(1, 1, 1, input_shape[-1]),
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
        config = super(PerPixelNoiseInjection, self).get_config()
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


class Swish(layers.Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, x):
        return x * tf.nn.sigmoid(x)


class GLU(layers.Layer):

    def __init__(self, **kwargs):
        super(GLU, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[
            -1] % 2 == 0, 'Last dimension of input shape must be even.'

    def call(self, inputs):
        return inputs[:, :, :, :inputs.shape[-1] // 2] * tf.math.sigmoid(
            inputs[:, :, :, inputs.shape[-1] // 2:])


class InitLayer(layers.Layer):

    def __init__(self, channel, **kwargs):
        super(InitLayer, self).__init__(**kwargs)
        self.channel = channel

        self.init = tf.keras.Sequential([
            layers.Reshape((1, 1, -1)),
            SpectralNormalization(
                layers.Conv2DTranspose(channel * 2, 4, 1, use_bias=False)),
            layers.BatchNormalization(),
            # InstanceNormalization(),
            GLU(),
        ])

    def call(self, inputs):
        return self.init(inputs)

    def get_config(self):
        config = super(InitLayer, self).get_config()
        config.update({'channel': self.channel})
        return config


class SEBlock(layers.Layer):

    def __init__(self, channel_out, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channel_out = channel_out

        self.block = tf.keras.Sequential([
            tfa.layers.AdaptiveAveragePooling2D(4),
            SpectralNormalization(
                layers.Conv2D(channel_out,
                              4,
                              1,
                              padding='valid',
                              use_bias=False)),
            Swish(),
            SpectralNormalization(
                layers.Conv2D(channel_out,
                              1,
                              1,
                              padding='valid',
                              use_bias=False)),
            layers.Activation('sigmoid')
        ])

    def call(self, x_small, x_big):
        return x_big * self.block(x_small)

    def get_config(self):
        base_config = super(SEBlock, self).get_config()
        config = {'channel_out': self.channel_out}
        return dict(list(base_config.items()) + list(config.items()))


class UpBlock(layers.Layer):

    def __init__(self, out_planes, **kwargs):
        super(UpBlock, self).__init__(**kwargs)
        self.out_planes = out_planes

        self.upsample = layers.UpSampling2D()
        self.block = tf.keras.Sequential([
            layers.Conv2D(out_planes * 2, 3, 1, 'same', use_bias=False),
            layers.BatchNormalization(),
            GLU(),
        ])
        self.epilogue = LayerEpilogue()

    def call(self, x, w=None):
        x = self.upsample(x)
        x = self.block(x)
        if w is not None:
            x = self.epilogue(x, w)
        return x


class UpBlockComp(layers.Layer):

    def __init__(self, out_planes, **kwargs):
        super(UpBlockComp, self).__init__(**kwargs)
        self.out_planes = out_planes

        self.upsample = layers.UpSampling2D()
        self.block_1 = tf.keras.Sequential([
            SpectralNormalization(
                layers.Conv2D(out_planes * 2, 3, 1, 'same', use_bias=False)),
            layers.BatchNormalization(),
            GLU(),
        ])
        self.block_2 = tf.keras.Sequential([
            SpectralNormalization(
                layers.Conv2D(out_planes * 2, 3, 1, 'same', use_bias=False)),
            layers.BatchNormalization(),
            GLU(),
        ])
        self.epilogue_1 = LayerEpilogue()
        self.epilogue_2 = LayerEpilogue()

    def call(self, x, w=None):
        x = self.upsample(x)
        x = self.block_1(x)
        if w is not None:
            x = self.epilogue_1(x, w)
        x = self.block_2(x)
        if w is not None:
            x = self.epilogue_2(x, w)
        return x

    def get_config(self):
        base_config = super(UpBlockComp, self).get_config()
        config = {'out_planes': self.out_planes}
        return dict(list(base_config.items()) + list(config.items()))


class DownBlock(layers.Layer):

    def __init__(self, out_planes, **kwargs):
        super(DownBlock, self).__init__(**kwargs)
        self.out_planes = out_planes
        self.down = tf.keras.Sequential([
            SpectralNormalization(
                layers.Conv2D(out_planes, 4, 2, 'same', use_bias=False)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])

    def call(self, inputs):
        x = self.down(inputs)
        return x


class DownBlockComp(layers.Layer):

    def __init__(self, out_planes, **kwargs):
        super(DownBlockComp, self).__init__(**kwargs)
        self.out_planes = out_planes

        self.main_path = tf.keras.Sequential([
            SpectralNormalization(
                layers.Conv2D(out_planes, 4, 2, 'same', use_bias=False)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            SpectralNormalization(
                layers.Conv2D(out_planes, 3, 1, 'same', use_bias=False)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])

        self.direct_path = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2),
            SpectralNormalization(
                layers.Conv2D(out_planes, 1, 1, 'valid', use_bias=False)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])

    def call(self, inputs, **kwargs):
        main_path = self.main_path(inputs)
        direct_path = self.direct_path(inputs)
        return (main_path + direct_path) / 2

    def get_config(self):
        config = super(DownBlockComp, self).get_config()
        config.update({
            'out_planes': self.out_planes,
        })
        return config


class SimpleDecoder(layers.Layer):

    def __init__(self, nfc_in=32, nc=3, **kwargs):
        super(SimpleDecoder, self).__init__(**kwargs)
        self.nfc_multi = {
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
        self.nfc = {}
        for k, v in self.nfc_multi.items():
            self.nfc[k] = int(nfc_in * v)

        self.conv = SpectralNormalization(
            layers.Conv2D(nc, 3, 1, 'same', use_bias=False))
        self.up_block_16 = UpBlock(self.nfc[16])
        self.up_block_32 = UpBlock(self.nfc[32])
        self.up_block_64 = UpBlock(self.nfc[64])
        self.up_block_128 = UpBlock(self.nfc[128])
        self.act = layers.Activation('tanh')

    def call(self, x):
        x = self.up_block_16(x)
        x = self.up_block_32(x)
        x = self.up_block_64(x)
        x = self.up_block_128(x)
        x = self.conv(x)
        x = self.act(x)
        return x

    def get_config(self):
        base_config = super(SimpleDecoder, self).get_config()
        config = {'nfc_in': self.nfc_in, 'nc': self.nc}
        return dict(list(base_config.items()) + list(config.items()))


class Discriminator(Model):

    def __init__(self, ndf=64, im_size=256):
        super(Discriminator, self).__init__()
        self.im_size = im_size
        self.ndf = ndf
        self.nfc_multi = {
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
        self.nfc = {}
        for k, v in self.nfc_multi.items():
            self.nfc[k] = int(v * ndf)

        if im_size == 1024:
            self.feat_2 = tf.keras.Sequential([
                SpectralNormalization(
                    layers.Conv2D(self.nfc[1024], 4, 2, 'same',
                                  use_bias=False)),
                layers.LeakyReLU(0.2),
                SpectralNormalization(
                    layers.Conv2D(self.nfc[512], 4, 2, 'same',
                                  use_bias=False)),
                layers.BatchNormalization(),
                layers.LeakyReLU(0.2)
            ])
        elif im_size == 512:
            self.feat_2 = tf.keras.Sequential([
                SpectralNormalization(
                    layers.Conv2D(self.nfc[512], 4, 2, 'same',
                                  use_bias=False)),
                layers.LeakyReLU(0.2)
            ])
        elif im_size == 256:
            self.feat_2 = tf.keras.Sequential([
                SpectralNormalization(
                    layers.Conv2D(self.nfc[512], 4, 1, 'same',
                                  use_bias=False)),
                layers.LeakyReLU(0.2)
            ])

        self.feat_4 = DownBlockComp(self.nfc[256])
        self.feat_8 = DownBlockComp(self.nfc[128])

        self.feat_16 = DownBlockComp(self.nfc[64])
        self.se_block_16 = SEBlock(self.nfc[64])
        self.feat_32 = DownBlockComp(self.nfc[32])
        self.se_block_32 = SEBlock(self.nfc[32])
        self.feat_last = DownBlockComp(self.nfc[16])
        self.se_block_last = SEBlock(self.nfc[16])

        self.out = tf.keras.Sequential([
            SpectralNormalization(layers.Conv2D(1, 4, 1, use_bias=False)),
            layers.Flatten(),
            layers.Dense(1)
        ])

    def call(self, inputs):
        feat_2 = self.feat_2(inputs)
        feat_4 = self.feat_4(feat_2)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        se_block_16 = self.se_block_16(feat_2, feat_16)
        feat_32 = self.feat_32(se_block_16)
        se_block_32 = self.se_block_32(feat_4, feat_32)
        feat_last = self.feat_last(se_block_32)
        se_block_last = self.se_block_last(feat_8, feat_last)
        output = self.out(se_block_last)
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = super(Discriminator, self).get_config()
        config.update({
            'im_size': self.im_size,
            'ndf': self.ndf,
        })
        return config


class Generator(Model):

    def __init__(self, ngf=64, im_size=256, nc=3, **kwargs):
        super(Generator, self).__init__(**kwargs)
        assert im_size in [256, 512,
                           1024], 'im_size must be in [256, 512, 1024]'
        self.im_size = im_size
        self.nc = nc
        self.ngf = ngf
        self.nfc_multi = {
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
        self.nfc = {}
        for k, v in self.nfc_multi.items():
            self.nfc[k] = int(v * ngf)

    def build(self, input_shape):
        self.mapping = Mapping(8, input_shape[-1])
        self.const = ConstLayer(input_shape[-1])

        self.feat_8 = UpBlockComp(self.nfc[8])
        self.feat_16 = UpBlock(self.nfc[16])
        self.feat_32 = UpBlockComp(self.nfc[32])
        self.feat_64 = UpBlock(self.nfc[64])
        self.feat_128 = UpBlockComp(self.nfc[128])
        self.feat_256 = UpBlock(self.nfc[256])

        if self.im_size == 512:
            self.feat_512 = UpBlockComp(self.nfc[512])
            self.se_block_512 = SEBlock(self.nfc[512])
        elif self.im_size == 1024:
            self.feat_512 = UpBlockComp(self.nfc[512])
            self.se_block_512 = SEBlock(self.nfc[512])
            self.feat_1024 = UpBlock(self.nfc[1024])

        self.se_block_64 = SEBlock(self.nfc[64])
        self.se_block_128 = SEBlock(self.nfc[128])
        self.se_block_256 = SEBlock(self.nfc[256])

        self.to_big = SpectralNormalization(
            layers.Conv2D(self.nc, 3, 1, 'same', use_bias=False))
        self.act = layers.Activation('tanh')

    def call(self, x, y):
        w = self.mapping(x, y)
        const = self.const(w)
        feat_8 = self.feat_8(const, w)
        feat_16 = self.feat_16(feat_8, w)
        feat_32 = self.feat_32(feat_16, w)
        feat_64 = self.feat_64(feat_32, w)
        se_block_64 = self.se_block_64(const, feat_64)
        feat_128 = self.feat_128(se_block_64, w)
        se_block_128 = self.se_block_128(feat_8, feat_128)
        feat_256 = self.feat_256(se_block_128, w)
        se_block_256 = self.se_block_256(feat_16, feat_256)

        if self.im_size == 256:
            to_big = self.to_big(se_block_256)
            output = self.act(to_big)
        elif self.im_size == 512:
            feat_512 = self.feat_512(se_block_256, w)
            se_block_512 = self.se_block_512(feat_32, feat_512)
            to_big = self.to_big(se_block_512)
            output = self.act(to_big)
        else:
            feat_512 = self.feat_512(se_block_256, w)
            se_block_512 = self.se_block_512(feat_32, feat_512)
            feat_1024 = self.feat_1024(se_block_512, w)
            to_big = self.to_big(feat_1024)
            output = self.act(to_big)

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.im_size, self.im_size, self.nc)

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({
            'nc': self.nc,
            'ngf': self.ngf,
            'im_size': self.im_size
        })
        return config


if __name__ == "__main__":
    disc = Discriminator()
    disc.build((None, 256, 256, 3))
    disc.summary()

    gen = Generator()
    gen.build((None, 256))
    gen.summary()

    gen.save_weights('./checkpoints/')
    gen.load_weights('./checkpoints/')
    print('Success')
