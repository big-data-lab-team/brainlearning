import nibabel as nib

from keras.engine import Layer
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math


def encoder_model():
    model = Sequential()
    # Layer E1
    model.add(Layer(input_shape=(320, 320, 1)))
    model.add(Conv2D(64, (11, 11), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('tanh'))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result 82x82x64

    # Layer E2
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result 41x41x128

    # Layer E3
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result 20x20x512

    # Layer E4
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation("sigmoid"))

    return model


def decoder_model():
    model = Sequential()

    # Layer D1
    model.add(Layer(input_shape=(2048,)))
    model.add(Dense(204800))
    model.add(Activation("sigmoid"))
    model.add(Reshape((20, 20, -1)))

    # Layer D2
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))
    model.add(UpSampling2D(size=(2, 2)))

    # Layer D3
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))
    model.add(UpSampling2D(size=(2, 2)))

    # Layer D4
    model.add(Conv2D(64, (11, 11), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(UpSampling2D(size=(2, 2)))

    # Layer D5
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(UpSampling2D(size=(2, 2)))

    return model


def combine_models(e, d):
    model = Sequential()
    model.add(e)
    model.add(d)
    return model


def train(BATCH_SIZE):
    e = encoder_model()
    d = decoder_model()
    e_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    #
    e.compile(loss='binary_crossentropy', optimizer=e_optim)
    print("Encoder Model")
    print(e.summary())
    #
    # full_model.compile(loss='binary_crossentropy', optimizer=g_optim)
    #
    # # d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    print("Decoder Model")
    print(d.summary())

    full_model = combine_models(e, d)
    print("Full Model")
    print(full_model.summary())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    return parser.parse_args()


def recreate_mask(array_element):
    if array_element > 0:
        return 1
    else:
        return 0


def pre_process():
    full_image_file_name = "../ml-bet/103414_3T_T1w_MPR1.nii.gz"
    brain_image_file_name = "../ml-bet/103414_3T_T1w_MPR1_brain.nii.gz"

    full_image_file = nib.load(full_image_file_name)
    brain_image_file = nib.load(brain_image_file_name)

    full_image = full_image_file.get_data()
    padded_full_image = np.pad(full_image, ((32, 32), (0, 0), (0, 0)), mode='constant')
    brain_image = brain_image_file.get_data()
    padded_brain_image = np.pad(brain_image, ((32, 32), (0, 0), (0, 0)), mode='constant')

    print('Full Image Shape', full_image.shape)
    print('Padded Full Image Shape', padded_full_image.shape)
    print('Brain Image Shape', brain_image.shape)
    print('Padded brain Image Shape', padded_brain_image.shape)

    # numpy.set_printoptions(threshold=numpy.nan)
    print('1 slice Full Image Shape')
    print(full_image[125, :, :])
    print('1 slice Padded Full Image Shape')
    print(padded_full_image[125, :, :])
    print('1 slice Image Shape')
    print(brain_image[125, :, :])

    z_full_stack = np.copy(padded_full_image)
    y_full_stack = np.rot90(padded_full_image, axes=(0, 1))
    x_full_stack = np.rot90(padded_full_image, axes=(0, 2))
    full_image_stack = np.concatenate((z_full_stack, y_full_stack, x_full_stack))
    full_image_stack_max = np.amax(full_image_stack, axis=(1, 2))
    print(full_image_stack_max.shape)

    full_image_stack_normallized = np.divide(full_image_stack, full_image_stack_max[:, None, None],
                                             where=full_image_stack_max[:, None, None] != 0)
    print(full_image_stack_normallized.shape)

    z_brain_stack = np.copy(padded_brain_image)
    y_brain_stack = np.rot90(padded_brain_image, axes=(0, 1))
    x_brain_stack = np.rot90(padded_brain_image, axes=(0, 2))
    brain_image_stack = np.concatenate((z_brain_stack, y_brain_stack, x_brain_stack))
    recreate_mask_vectorized = np.vectorize(recreate_mask)
    # brain_image_stack = recreate_mask_vectorized(brain_image_stack)


def experiments():
    m = np.array([[[1, 2], [3, 4]], [[21, 22], [23, 24]]], int)
    print(m.shape)
    print(m[0, :, :])

    n = np.rot90(m, axes=(0, 1))
    print(n.shape)
    print(n[0, :, :])

    x = np.rot90(m, axes=(0, 2))
    print(x.shape)
    print(x[0, :, :])

    new_array = np.concatenate((m, n, x))
    recreate_mask_vectorized = np.vectorize(recreate_mask)
    print(new_array.shape)
    print(new_array[0, :, :])

    mask = recreate_mask_vectorized(new_array)
    print(mask.shape)
    print(mask[0, :, :])

    maxs = np.amax(new_array, axis=(1, 2))
    print(maxs.shape)

    normallized = np.divide(new_array, maxs[:, None, None])
    print(normallized.shape)


if __name__ == "__main__":
    args = get_args()
    # experiments()

    train(BATCH_SIZE=args.batch_size)
    # pre_process()

    # if args.mode == "train":
    #     train(BATCH_SIZE=args.batch_size)
    # elif args.mode == "generate":
    # generate(BATCH_SIZE=args.batch_size, nice=args.nice)
