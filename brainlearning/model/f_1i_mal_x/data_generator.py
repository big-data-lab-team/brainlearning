import fnmatch
from os import listdir
from os.path import isfile, join

import keras
import nibabel as nib
import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 dir_path,
                 file_pattern='*.nii.gz',
                 distinguish_pattern='_brain',
                 batch_size=10,
                 dim=320,
                 n_channels=1,
                 shuffle=True):

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
        # print('Get Item index ', index)

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.file_pairs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_of_pairs):
        x = np.empty(shape=(0, self.dim, self.dim, 1))
        y = np.empty(shape=(0, self.dim, self.dim, 1))
        # print('list_of_pairs: ', list_of_pairs)

        for pair in list_of_pairs:
            x_file, y_file = self.process_pair(pair)
            # random_idx = np.random.randint(self.dim * 3, size=self.batch_size)
            random_idx = np.random.randint(self.dim, size=self.batch_size)

            x_temp = x_file[random_idx, :]
            y_temp = y_file[random_idx, :]

            x_temp = x_temp.reshape((self.batch_size, self.dim, self.dim, 1))
            y_temp = y_temp.reshape((self.batch_size, self.dim, self.dim, 1))

            x = np.append(x, x_temp, axis=0)
            y = np.append(y, y_temp, axis=0)

        return x, y

    def get_file(self, file_name=None):
        if file_name is None:
            file_name = self.file_pairs[0][0]

        full_image_file = nib.load(file_name)
        full_image = full_image_file.get_data()
        padded_full_image = np.pad(full_image, ((32, 32), (0, 0), (0, 0)), mode='constant')

        z_full_stack = np.copy(padded_full_image)
        z_full_stack_max = np.amax(z_full_stack, axis=(1, 2))
        z_full_stack_normalized = np.divide(z_full_stack, z_full_stack_max[:, None, None],
                                            where=z_full_stack_max[:, None, None] != 0)
        z_full_stack_normalized = np.reshape(z_full_stack_normalized, newshape=(320, 320, 320, 1))

        y_full_stack = np.rot90(padded_full_image, axes=(0, 1))
        y_full_stack_max = np.amax(y_full_stack, axis=(1, 2))
        y_full_stack_normalized = np.divide(y_full_stack, y_full_stack_max[:, None, None],
                                            where=y_full_stack_max[:, None, None] != 0)
        y_full_stack_normalized = np.reshape(y_full_stack_normalized, newshape=(320, 320, 320, 1))

        x_full_stack = np.rot90(padded_full_image, axes=(0, 2))
        x_full_stack_max = np.amax(x_full_stack, axis=(1, 2))
        x_full_stack_normalized = np.divide(x_full_stack, x_full_stack_max[:, None, None],
                                            where=z_full_stack_max[:, None, None] != 0)
        x_full_stack_normalized = np.reshape(x_full_stack_normalized, newshape=(320, 320, 320, 1))

        # full_image_stack = np.concatenate((z_full_stack, y_full_stack, x_full_stack))
        #
        # full_image_stack_max = np.amax(full_image_stack, axis=(1, 2))
        # full_image_stack_normalized = np.divide(full_image_stack, full_image_stack_max[:, None, None],
        #                                         where=full_image_stack_max[:, None, None] != 0)
        return x_full_stack_normalized, y_full_stack_normalized, z_full_stack_normalized, file_name

    @staticmethod
    def group_into_layers(array, dim, n_channels):
        output = np.empty(shape=(0, dim, dim, n_channels))
        margin = (n_channels - 1) // 2
        padding = np.zeros(shape=(margin, dim, dim))
        for layer in range(dim):
            chunk = np.empty(shape=(0, dim, dim))
            if layer == 0:
                chunk = np.concatenate((chunk, padding), axis=0)
                chunk = np.concatenate((chunk, array[layer: layer + margin + 1]), axis=0)
            elif layer == dim - 1:
                chunk = np.concatenate((chunk, array[layer - margin: layer + 1]), axis=0)
                chunk = np.concatenate((chunk, padding), axis=0)
            else:
                chunk = np.concatenate((chunk, array[layer - margin: layer + margin + 1]), axis=0)
            chunk = np.moveaxis(chunk, 0, -1)
            output = np.concatenate((output, [chunk]), axis=0)
        return output

    @staticmethod
    def get_layer(pile, layer, l, r, channels, dim):
        if l - (layer - channels) > 0:
            return np.zeros(shape=(dim, dim))
        if r - (layer + channels) < 0:
            return np.zeros(shape=(dim, dim))
        return pile[layer]

    @staticmethod
    def edges(layer, dim):
        l = None
        r = None
        if layer < dim:
            l = 0
            r = dim
        elif layer < dim * 2:
            l = dim
            r = dim * 2
        elif layer < dim * 3:
            l = dim * 2
            r = dim * 3
        return l, r

    @staticmethod
    def save_to_file(image, file_name, new_file_name):
        full_image_file = nib.load(file_name)
        new_image = nib.Nifti1Image(image, full_image_file.affine, full_image_file.header)
        nib.save(new_image, new_file_name)

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

        # z_full_stack = np.copy(padded_full_image)
        # y_full_stack = np.rot90(padded_full_image, axes=(0, 1))
        x_full_stack = np.rot90(padded_full_image, axes=(0, 2))
        # full_image_stack = np.concatenate((z_full_stack, y_full_stack, x_full_stack))
        full_image_stack = x_full_stack

        full_image_stack_max = np.amax(full_image_stack, axis=(1, 2))
        # print(full_image_stack_max.shape)

        full_image_stack_normalized = np.divide(full_image_stack, full_image_stack_max[:, None, None],
                                                where=full_image_stack_max[:, None, None] != 0)
        # print(full_image_stack_normalized.shape)

        # z_brain_mask_stack = np.copy(padded_brain_image_mask)
        # y_brain_mask_stack = np.rot90(padded_brain_image_mask, axes=(0, 1))
        x_brain_mask_stack = np.rot90(padded_brain_image_mask, axes=(0, 2))
        # brain_image_mask_stack = np.concatenate((z_brain_mask_stack, y_brain_mask_stack, x_brain_mask_stack))
        brain_image_mask_stack = x_brain_mask_stack

        return full_image_stack_normalized, brain_image_mask_stack

    @staticmethod
    def calculate_padding(shape):
        padding = 320 - shape[0]
        if shape[0] % 2 == 0:
            return padding / 2, padding / 2
        else:
            return padding / 2, padding / 2 + 1

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
