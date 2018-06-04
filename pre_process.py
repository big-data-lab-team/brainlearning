import datetime as dt
import importlib

import nibabel as nib
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.engine import Layer
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU, Dropout
from keras.layers import Reshape, Softmax
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import os
import time
from os import listdir
from os.path import isfile, join
import fnmatch


def pre_process():
    full_image_file_name = "../ml-bet/103414_3T_T1w_MPR1.nii.gz"
    full_image_file = nib.load(full_image_file_name)
    full_image = full_image_file.get_data()
    padded_full_image = np.pad(full_image, ((32, 32), (0, 0), (0, 0)), mode='constant')

    brain_image_file_name = "../ml-bet/103414_3T_T1w_MPR1_brain.nii.gz"
    brain_image_file = nib.load(brain_image_file_name)
    brain_image = brain_image_file.get_data()
    brain_image_mask = (brain_image != 0).astype(int)
    # padded_brain_image = np.pad(brain_image, ((32, 32), (0, 0), (0, 0)), mode='constant')
    padded_brain_image_mask = np.pad(brain_image_mask, ((32, 32), (0, 0), (0, 0)), mode='constant')

    print('Full Image Shape', full_image.shape)
    print('Padded Full Image Shape', padded_full_image.shape)

    print('Brain Image Shape', brain_image.shape)
    # print('Padded Brain Image Shape', padded_brain_image.shape)
    print('Brain Mask Image Shape', brain_image_mask.shape)
    print('Padded Brain Mask Image Shape', padded_brain_image_mask.shape)

    z_full_stack = np.copy(padded_full_image)
    y_full_stack = np.rot90(padded_full_image, axes=(0, 1))
    x_full_stack = np.rot90(padded_full_image, axes=(0, 2))
    full_image_stack = np.concatenate((z_full_stack, y_full_stack, x_full_stack))

    full_image_stack_max = np.amax(full_image_stack, axis=(1, 2))
    print(full_image_stack_max.shape)

    full_image_stack_normalized = np.divide(full_image_stack, full_image_stack_max[:, None, None],
                                            where=full_image_stack_max[:, None, None] != 0)

    print(full_image_stack_normalized.shape)

    # z_brain_stack = np.copy(padded_brain_image)
    # y_brain_stack = np.rot90(padded_brain_image, axes=(0, 1))
    # x_brain_stack = np.rot90(padded_brain_image, axes=(0, 2))
    # brain_image_stack = np.concatenate((z_brain_stack, y_brain_stack, x_brain_stack))
    # print(brain_image_stack.shape)

    z_brain_mask_stack = np.copy(padded_brain_image_mask)
    y_brain_mask_stack = np.rot90(padded_brain_image_mask, axes=(0, 1))
    x_brain_mask_stack = np.rot90(padded_brain_image_mask, axes=(0, 2))
    brain_image_mask_stack = np.concatenate((z_brain_mask_stack, y_brain_mask_stack, x_brain_mask_stack))
    print(brain_image_mask_stack.shape)

    # print('same ', dice_loss(brain_image_mask_stack[125], brain_image_mask_stack[125]))
    # print('different ', dice_loss(brain_image_mask_stack[125], brain_image_mask_stack[126]))
    # print('middle and zero ', dice_loss(brain_image_mask_stack[125], brain_image_mask_stack[0]))

    return full_image_stack_normalized, brain_image_mask_stack


def process_pair(files):
    full_image_file_name, brain_image_file_name = files

    full_image_file = nib.load(full_image_file_name)
    full_image = full_image_file.get_data()
    padded_full_image = np.pad(full_image, ((32, 32), (0, 0), (0, 0)), mode='constant')

    brain_image_file = nib.load(brain_image_file_name)
    brain_image = brain_image_file.get_data()
    brain_image_mask = (brain_image != 0).astype(int)
    padded_brain_image_mask = np.pad(brain_image_mask, ((32, 32), (0, 0), (0, 0)), mode='constant')

    z_full_stack = np.copy(padded_full_image)
    y_full_stack = np.rot90(padded_full_image, axes=(0, 1))
    x_full_stack = np.rot90(padded_full_image, axes=(0, 2))
    full_image_stack = np.concatenate((z_full_stack, y_full_stack, x_full_stack))

    full_image_stack_max = np.amax(full_image_stack, axis=(1, 2))
    # print(full_image_stack_max.shape)

    full_image_stack_normalized = np.divide(full_image_stack, full_image_stack_max[:, None, None],
                                            where=full_image_stack_max[:, None, None] != 0)
    # print(full_image_stack_normalized.shape)

    z_brain_mask_stack = np.copy(padded_brain_image_mask)
    y_brain_mask_stack = np.rot90(padded_brain_image_mask, axes=(0, 1))
    x_brain_mask_stack = np.rot90(padded_brain_image_mask, axes=(0, 2))
    brain_image_mask_stack = np.concatenate((z_brain_mask_stack, y_brain_mask_stack, x_brain_mask_stack))

    return full_image_stack_normalized, brain_image_mask_stack


def get_file_pairs(dir_path, file_pattern='*.nii.gz', distinguish_pattern='_brain'):
    only_files = [f for f in listdir(dir_path) if (isfile(join(dir_path, f)) and fnmatch.fnmatch(f, file_pattern))]
    # not_brain_list = [f for f in only_files if '_brain' not in f]
    brain_list = [f for f in only_files if distinguish_pattern in f]
    brain_list.sort()

    file_pairs = []
    for f in brain_list:
        if f.replace(distinguish_pattern, '') in only_files:
            file_pairs.append((dir_path + f.replace(distinguish_pattern, ''), dir_path + f))
    return file_pairs
