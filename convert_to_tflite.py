import argparse
import os

import tensorflow as tf
from tensorflow.keras import utils as kutils
from tqdm import tqdm

from models import Generator, NoiseInjection

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_model', type=str, help='path to h5 file')
    parser.add_argument('--save_dest',
                        type=str,
                        default='./',
                        help='path to save tflite model')
    parser.add_argument('--name',
                        type=str,
                        default='model',
                        help='name of model')
    parser.add_argument('--im_size',
                        type=int,
                        default=256,
                        help='image size of trained model')
    args = parser.parse_args()

    modelG = Generator(im_size=args.im_size)
    modelG.load_weights(args.save_model)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(modelG)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Save the model.
    with open(os.path.join(args.save_dest, args.name + '.tflite'), 'wb') as f:
        f.write(tflite_model)
