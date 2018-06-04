import fnmatch
from os import listdir
from os.path import isfile, join

import keras
import nibabel as nib
import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dir_path, file_pattern='*.nii.gz', distinguish_pattern='_brain', batch_size=10, dim=320,
                 n_channels=1, shuffle=True):
        self.dir_path = dir_path
        self.file_pattern = file_pattern
        self.distinguish_pattern = distinguish_pattern
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.file_pairs = self.get_file_pairs(self.dir_path, file_pattern, distinguish_pattern)
        self.indexes = np.arange(len(self.file_pairs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_pairs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.file_pairs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        x = np.empty(shape=(0, self.dim, self.dim, self.n_channels))
        y = np.empty(shape=(0, self.dim, self.dim, 1))

        for i in list_ids_temp:
            x_temp, y_temp = self.process_pair(i)
            x_temp = x_temp.reshape((self.dim * 3, self.dim, self.dim, self.n_channels))
            y_temp = y_temp.reshape((self.dim * 3, self.dim, self.dim, 1))
            x = np.append(x, x_temp, axis=0)
            y = np.append(y, y_temp, axis=0)
        return x, y

    @staticmethod
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

    @staticmethod
    def get_file_pairs(dir_path, file_pattern, distinguish_pattern):
        only_files = [f for f in listdir(dir_path) if (isfile(join(dir_path, f)) and fnmatch.fnmatch(f, file_pattern))]
        brain_list = [f for f in only_files if distinguish_pattern in f]
        brain_list.sort()

        file_pairs = []
        for f in brain_list:
            if f.replace(distinguish_pattern, '') in only_files:
                file_pairs.append((dir_path + f.replace(distinguish_pattern, ''), dir_path + f))
        return file_pairs
