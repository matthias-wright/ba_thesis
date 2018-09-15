# -*- coding: utf-8 -*-

"""
This module handles the input the data.
"""

__author__ = 'Matthias Wright'


import numpy as np
import math


class Data:
    X = None

    def __init__(self, path):
        self.X = np.loadtxt(fname=path, delimiter=',')

    def get_mini_batches(self, mb_size):
        """
        This function returns a list containing the mini-batches.
        :param mb_size: size of the mini batch
        :return: mini-batch list
        """
        np.random.shuffle(self.X)
        m = self.X.shape[0]
        num_batches = int(math.floor(m/mb_size))
        mini_batches = []
        for k in range(0, num_batches):
            mini_batch = self.X[k * mb_size:(k+1) * mb_size, :]
            mini_batches.append(mini_batch)

        if m % mb_size != 0:
            mini_batch = self.X[-(m % mb_size):, :]
            mini_batches.append(mini_batch)
        return mini_batches

    def get_single_batch(self, mb_size):
        return [self.X[:mb_size, :]]

    def get_all_data(self):
        np.random.shuffle(self.X)
        return self.X
