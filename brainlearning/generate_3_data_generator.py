import os.path

import nibabel as nib
import numpy as np


class DataGenerator:
    def __init__(self,
                 file_pattern='*.nii.gz',
                 distinguish_pattern='_brain',
                 dim=320):
        self.file_pattern = file_pattern
        self.distinguish_pattern = distinguish_pattern
        self.dim = dim

    def get_file(self, file_name):

        file_name = os.path.realpath(file_name)
        full_image_file = nib.load(file_name)
        full_image = full_image_file.get_data()
        first_padding = self.calculate_padding(full_image.shape[0])
        second_padding = self.calculate_padding(full_image.shape[1])
        third_padding = self.calculate_padding(full_image.shape[2])

        padded_full_image = np.pad(full_image, (first_padding, second_padding, third_padding), mode='constant')

        z_full_stack = np.copy(padded_full_image)
        z_full_stack_max = np.amax(z_full_stack, axis=(1, 2))
        z_full_stack_normalized = np.divide(z_full_stack, z_full_stack_max[:, None, None],
                                            where=z_full_stack_max[:, None, None] != 0)
        z_full_stack_normalized = np.reshape(z_full_stack_normalized, newshape=(self.dim, self.dim, self.dim, 1))

        y_full_stack = np.rot90(padded_full_image, axes=(0, 1))
        y_full_stack_max = np.amax(y_full_stack, axis=(1, 2))
        y_full_stack_normalized = np.divide(y_full_stack, y_full_stack_max[:, None, None],
                                            where=y_full_stack_max[:, None, None] != 0)
        y_full_stack_normalized = np.reshape(y_full_stack_normalized, newshape=(self.dim, self.dim, self.dim, 1))

        x_full_stack = np.rot90(padded_full_image, axes=(0, 2))
        x_full_stack_max = np.amax(x_full_stack, axis=(1, 2))
        x_full_stack_normalized = np.divide(x_full_stack, x_full_stack_max[:, None, None],
                                            where=z_full_stack_max[:, None, None] != 0)
        x_full_stack_normalized = np.reshape(x_full_stack_normalized, newshape=(self.dim, self.dim, self.dim, 1))

        return x_full_stack_normalized, y_full_stack_normalized, z_full_stack_normalized, file_name, (
            first_padding, second_padding, third_padding)

    @staticmethod
    def save_to_file(image, file_name, new_file_name):
        full_image_file = nib.load(file_name)
        new_image = nib.Nifti1Image(image, full_image_file.affine, full_image_file.header)
        nib.save(new_image, new_file_name)

    def calculate_padding(self, shape):
        padding = self.dim - shape
        if shape % 2 == 0:
            return padding // 2, padding // 2
        else:
            return padding // 2, padding // 2 + 1

    def process_x(self, pred_x, padding):
        pred_x = np.reshape(pred_x, (self.dim, self.dim, self.dim))
        pred_x = np.rot90(pred_x, k=-1, axes=(0, 2))
        return self.trim_array(pred_x, padding)

    def process_y(self, pred_y, padding):
        pred_y = np.reshape(pred_y, (self.dim, self.dim, self.dim))
        pred_y = np.rot90(pred_y, k=-1, axes=(0, 1))
        return self.trim_array(pred_y, padding)

    def process_z(self, pred_z, padding):
        pred_z = np.reshape(pred_z, (self.dim, self.dim, self.dim))
        return self.trim_array(pred_z, padding)

    @staticmethod
    def trim_array(array, padding):
        first_padding = padding[0]
        if first_padding[1] == 0:
            first_padding_end = array.shape[0]
        else:
            first_padding_end = -first_padding[1]

        second_padding = padding[1]
        if second_padding[1] == 0:
            second_padding_end = array.shape[1]
        else:
            second_padding_end = -second_padding[1]

        third_padding = padding[2]
        if third_padding[1] == 0:
            third_padding_end = array.shape[2]
        else:
            third_padding_end = -third_padding[1]

        print('==== trim_array ==== ')
        print('input shape = ', array.shape)
        print('first_padding = ', first_padding)
        print('second_padding = ', second_padding)
        print('third_padding = ', third_padding)

        return array[first_padding[0]:first_padding_end,
               second_padding[0]:second_padding_end,
               third_padding[0]:third_padding_end]
