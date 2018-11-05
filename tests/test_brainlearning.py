#!/usr/bin/env python

from unittest import TestCase

import numpy as np

import brainlearning.generate_3_data_generator as generate_3_data_generator


class TestBrainLearning(TestCase):

    def test_calculate_padding(self):
        data_generator_inst = generate_3_data_generator.DataGenerator()
        assert (10, 10) == data_generator_inst.calculate_padding(300)

    def test_process_x(self):
        data_generator_inst = generate_3_data_generator.DataGenerator()
        array = np.ones((320, 320, 320, 1))
        array[10][10][10][0] = 100
        padding = ((10, 10), (5, 5), (0, 0))
        result = data_generator_inst.process_x(array, padding)
        assert (300, 310, 320) == result.shape
        assert result[0][5][309] == 100

    def test_process_y(self):
        data_generator_inst = generate_3_data_generator.DataGenerator()
        array = np.ones((320, 320, 320, 1))
        array[10][10][10][0] = 100
        padding = ((10, 10), (5, 5), (0, 0))
        result = data_generator_inst.process_y(array, padding)
        assert (300, 310, 320) == result.shape
        assert result[0][304][10] == 100

    def test_process_z(self):
        data_generator_inst = generate_3_data_generator.DataGenerator()
        array = np.ones((320, 320, 320, 1))
        array[10][10][10][0] = 100
        padding = ((10, 10), (5, 5), (0, 0))
        result = data_generator_inst.process_z(array, padding)
        assert (300, 310, 320) == result.shape
        assert result[0][5][10] == 100

    def test_trim_array_1_to_trim(self):
        data_generator_inst = generate_3_data_generator.DataGenerator()
        array = np.array([[[11, 12, 13],
                           [14, 15, 16],
                           [17, 18, 19]],
                          [[21, 22, 23],
                           [24, 25, 26],
                           [27, 28, 29]],
                          [[31, 32, 33],
                           [34, 35, 36],
                           [37, 38, 39]]])
        padding = ((1, 1), (1, 1), (1, 1))
        assert (1, 1, 1) == data_generator_inst.trim_array(array, padding).shape

    def test_trim_array_0_to_trim(self):
        data_generator_inst = generate_3_data_generator.DataGenerator()
        array = np.array([[[11, 12, 13],
                           [14, 15, 16],
                           [17, 18, 19]],
                          [[21, 22, 23],
                           [24, 25, 26],
                           [27, 28, 29]],
                          [[31, 32, 33],
                           [34, 35, 36],
                           [37, 38, 39]]])
        padding = ((0, 0), (0, 0), (0, 0))
        assert (3, 3, 3) == data_generator_inst.trim_array(array, padding).shape
