import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow_addons.layers import SpectralNormalization


def conv2d(*args, **kwargs):
    return SpectralNormalization(layers.Conv2D(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return SpectralNormalization(layers.Conv2DTranspose(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return layers.BatchNormalization(*args, **kwargs)


def linear(*args, **kwargs):
    return SpectralNormalization(layers.Dense(*args, **kwargs))


class PixelNorm(tf.Module):
    def __call__(self, input):
        return input * tf.math.rsqrt(
            tf.reduce_mean(input**2, axis=1, keepdims=True) + 1e-8)


class GLU(tf.Module):
    def __call__(self, x):
        nc = x.shape[-1]
        assert nc % 2 == 0, 'channels is not divisible by 2!'
        nc = int(nc / 2)
        return x[:, :, :, :nc] * tf.math.sigmoid(x[:, :, :, nc:])


class NoiseInjection(tf.Module):
    def __init__(self):
        super().__init__()
        self.weight = tf.Variable(tf.zeros(1), trainable=True)

    def __call__(self, feat, noise=None):
        if noise is None:
            batch, height, width, _ = feat.shape
            noise = tf.random.normal((batch, height, width, 1))

        return feat + self.weight * noise


class Swish(tf.Module):
    def __call__(self, feat):
        return feat * tf.math.sigmoid(feat)


class SEBlock(tf.Module):
    def __init__(self, ch_out):
        super().__init__()
        self.main = keras.Sequential([
            tfa.layers.AdaptiveAveragePooling2D(4),
            conv2d(ch_out, 4, 1, use_bias=False),
            Swish(),
            conv2d(ch_out, 1, 1, use_bias=False),
            layers.Activation('sigmoid')
        ])

    def __call__(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(tf.Module):
    def __init__(self, chatfel):
        super().__init__()
        self.init = keras.Sequential([
            # f, k, s, 'valid'
            convTranspose2d(chatfel * 2, 4, 1, use_bias=False),
            batchNorm2d(),
            GLU()
        ])

    def __call__(self, noise):
        noise = tf.reshape(noise, (noise.shape[0], 1, 1, -1))
        return self.init(noise)


def UpBlock(out_planes):
    block = keras.Sequential([
        layers.UpSampling2D(),
        conv2d(out_planes * 2, 3, 1, 'same', use_bias=False),
        #convTranspose2d(out_planes*2, 4, 2, 1, use_bias=False),
        batchNorm2d(),
        GLU()
    ])
    return block


def UpBlockComp(out_planes):
    block = keras.Sequential([
        layers.UpSampling2D(),
        conv2d(out_planes * 2, 3, 1, 'same', use_bias=False),
        #convTranspose2d(out_planes*2, 4, 2, 1, use_bias=False),
        NoiseInjection(),
        batchNorm2d(),
        GLU(),
        conv2d(out_planes * 2, 3, 1, 'same', use_bias=False),
        NoiseInjection(),
        batchNorm2d(),
        GLU()
    ])
    return block


class Generator(Model):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(Generator, self).__init__()

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

        self.im_size = im_size

        self.init = InitLayer(chatfel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[8])
        self.feat_16 = UpBlock(nfc[16])
        self.feat_32 = UpBlockComp(nfc[32])
        self.feat_64 = UpBlock(nfc[64])
        self.feat_128 = UpBlockComp(nfc[128])
        self.feat_256 = UpBlock(nfc[256])

        self.se_64 = SEBlock(nfc[64])
        self.se_128 = SEBlock(nfc[128])
        self.se_256 = SEBlock(nfc[256])

        self.to_128 = conv2d(nc, 1, 1, use_bias=False)
        self.to_big = conv2d(nc, 3, 1, 'same', use_bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[512])
            self.se_512 = SEBlock(nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[1024])

    def call(self, input):

        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = tf.tanh(self.to_128(feat_128))
        im_1024 = tf.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


class DownBlock(tf.Module):
    def __init__(self, out_planes):
        super(DownBlock, self).__init__()

        self.main = keras.Sequential([
            conv2d(out_planes, 4, 2, 'same', use_bias=False),
            batchNorm2d(),
            layers.LeakyReLU(0.2),
        ])

    def __call__(self, feat):
        return self.main(feat)


class DownBlockComp(tf.Module):
    def __init__(self, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = keras.Sequential([
            conv2d(out_planes, 4, 2, 'same', use_bias=False),
            batchNorm2d(),
            layers.LeakyReLU(0.2),
            conv2d(out_planes, 3, 1, 'same', use_bias=False),
            batchNorm2d(),
            layers.LeakyReLU(0.2)
        ])

        self.direct = keras.Sequential([
            layers.AveragePooling2D(2, 2),
            conv2d(out_planes, 1, 1, use_bias=False),
            batchNorm2d(),
            layers.LeakyReLU(0.2)
        ])

    def __call__(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class SimpleDecoder(tf.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

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

        def upBlock(out_planes):
            block = keras.Sequential([
                layers.UpSampling2D(),
                conv2d(out_planes * 2, 3, 1, 'same', use_bias=False),
                batchNorm2d(),
                GLU()
            ])
            return block

        self.main = keras.Sequential([
            tfa.layers.AdaptiveAveragePooling2D(8),
            upBlock(nfc[16]),
            upBlock(nfc[32]),
            upBlock(nfc[64]),
            upBlock(nfc[128]),
            conv2d(nc, 3, 1, 'same', use_bias=False),
            layers.Activation('tanh')
        ])

    def __call__(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


class Discriminator(tf.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

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
            self.down_from_big = keras.Sequential([
                conv2d(nfc[1024], 4, 2, 'same', use_bias=False),
                layers.LeakyReLU(0.2),
                conv2d(nfc[512], 4, 2, 'same', use_bias=False),
                batchNorm2d(),
                layers.LeakyReLU(0.2)
            ])
        elif im_size == 512:
            self.down_from_big = keras.Sequential([
                conv2d(nfc[512], 4, 2, 'same', use_bias=False),
                layers.LeakyReLU(0.2)
            ])
        elif im_size == 256:
            self.down_from_big = keras.Sequential([
                conv2d(nfc[512], 3, 1, 'same', use_bias=False),
                layers.LeakyReLU(0.2)
            ])

        self.down_4 = DownBlockComp(nfc[256])
        self.down_8 = DownBlockComp(nfc[128])
        self.down_16 = DownBlockComp(nfc[64])
        self.down_32 = DownBlockComp(nfc[32])
        self.down_64 = DownBlockComp(nfc[16])

        self.rf_big = keras.Sequential([
            conv2d(nfc[8], 1, 1, use_bias=False),
            batchNorm2d(),
            layers.LeakyReLU(0.2),
            conv2d(1, 4, 1, use_bias=False)
        ])

        self.se_2_16 = SEBlock(nfc[64])
        self.se_4_32 = SEBlock(nfc[32])
        self.se_8_64 = SEBlock(nfc[16])

        self.down_from_small = keras.Sequential([
            conv2d(nfc[256], 4, 2, 'same', use_bias=False),
            layers.LeakyReLU(0.2),
            DownBlock(nfc[128]),
            DownBlock(nfc[64]),
            DownBlock(nfc[32]),
        ])

        self.rf_small = conv2d(1, 4, 1, use_bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def __call__(self, imgs, label, part=None):
        #NOTE: Could be 'bilinear' or 'nearest'
        if type(imgs) is not list:
            imgs = [
                tf.image.resize(imgs, size=[self.im_size, self.im_size]),
                tf.image.resize(imgs, size=[128, 128])
            ]

        feat_2 = self.down_from_big(imgs[0])
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)

        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        #rf_0 = torch.cat([self.rf_big_1(feat_last).view(-1),self.rf_big_2(feat_last).view(-1)])
        #rff_big = torch.sigmoid(self.rf_factor_big)
        rf_0 = tf.reshape(self.rf_big(feat_last), -1)

        feat_small = self.down_from_small(imgs[1])
        #rf_1 = torch.cat([self.rf_small_1(feat_small).view(-1),self.rf_small_2(feat_small).view(-1)])
        rf_1 = tf.reshape(self.rf_small(feat_small), -1)

        if label == 'real':
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :8, :8, :])
            if part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :8, 8:, :])
            if part == 2:
                rec_img_part = self.decoder_part(feat_32[:, 8:, :8, :])
            if part == 3:
                rec_img_part = self.decoder_part(feat_32[:, 8:, 8:, :])

            return tf.concat([rf_0, rf_1], axis=0), [
                rec_img_big, rec_img_small, rec_img_part
            ]

        return tf.concat([rf_0, rf_1], axis=0)


class TextureDiscriminator(tf.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {
            4: 16,
            8: 8,
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

        self.down_from_small = keras.Sequential([
            conv2d(nc, nfc[256], 4, 2, 1, use_bias=False),
            layers.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]),
        ])
        self.rf_small = keras.Sequential(
            [conv2d(nfc[16], 1, 4, 1, 0, use_bias=False)])

        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, img, label):
        img = tf.image.random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = tf.reshape(self.rf_small(feat_small), -1)

        if label == 'real':
            rec_img_small = self.decoder_small(feat_small)
            return rf, rec_img_small, img

        return rf
