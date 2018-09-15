# -*- coding: utf-8 -*-

"""
This module processes the records.
It uses the native Python WFDB package: https://github.com/MIT-LCP/wfdb-python
"""

__author__ = 'Matthias Wright'

import numpy as np
import wfdb as wf


def process_records(db_path, records_path, dataset_normal, dataset_abnormal, signal_name, sample_size):
    """
    Processes the specified records from the specified database. A signal is divided in multiple intervals,
    where one interval represents the time frame between heart beats. The values of the interval are interpolated,
    sp that every interval contains 100 values. The length of the interval is then added to the front of the interval.
    If the interval is annotated as normal ('N'), it will be written in the dataset file for normal examples, otherwise
    it will be written in the dataset file for abnormal examples.
    Only one record is processed at a time and then directly written to the corresponding output file because all of
    the records are unlikely to fit into the RAM at once.
    :param db_path: path of the database
    :param records_path: path of the file containing the record names that should be selected
    :param dataset_normal: output file path for the dataset containing the normal examples
    :param dataset_abnormal: output file path for the dataset containing the abnormal examples
    :param signal_name: name of the signal that should be selected
    :param sample_size: amount of values per example
    """
    file = open(records_path, 'r')
    names = file.read().splitlines()
    f_normal = open(dataset_normal, 'a+')
    f_abnormal = open(dataset_abnormal, 'a+')
    for name in names:
        record = wf.rdrecord(db_path + '/' + name)
        annotation = wf.rdann(db_path + '/' + name, 'atr')
        signal_index = __get_signal_index(record, signal_name)
        if signal_index == -1:
            continue
        intervals_normal, intervals_abnormal = __get_intervals(record.p_signal.T[signal_index],
                                                               annotation.sample, annotation.symbol)
        intervals_normal_inter = __interpolate(intervals_normal, sample_size)
        intervals_abnormal_inter = __interpolate(intervals_abnormal, sample_size)
        for interval in intervals_normal_inter:
            f_normal.write(__array_to_string(interval) + '\n')
        for interval in intervals_abnormal_inter:
            f_abnormal.write(__array_to_string(interval) + '\n')
    f_normal.close()
    f_abnormal.close()


def __array_to_string(array):
    """
    Converts an array/list into a comma-separated string. The values are rounded up to 2 decimal places.
    :param array: the array.
    :return: string.
    """
    s = ''
    for i in range(len(array)):
        s += str(round(array[i], 2)) + ','
    return s[:-1]


def __interpolate(intervals, size):
    """
    Interpolates the values in the interval to a specified amount (size).
    :param intervals: original interval
    :param size: size of the interpolated interval
    :return: interpolated interval
    """
    intervals_new = []
    for interval in intervals:
        x = np.arange(len(interval))
        x_new = np.linspace(start=0, stop=len(interval), num=size)
        # add length to the beginning of the interval
        interval_new = np.concatenate(([len(interval)], np.interp(x_new, x, interval)), axis=0)
        intervals_new.append(interval_new)
    return intervals_new


def __get_intervals(signal, indexes, symbols):
    """
    Divides the normal intervals from the abnormal one and returns them in two different lists.
    :param signal: signal containing the intervals
    :param indexes: indexes of the annotations
    :param symbols: annotation symbol
    :return: list containing the normal intervals, list containing the abnormal intervals
    """
    assert len(indexes) == len(symbols)
    intervals_normal = []
    intervals_abnormal = []
    for i in range(len(indexes)-1):
        if symbols[i] == 'N' and symbols[i+1] == 'N':
            intervals_normal.append(signal[indexes[i]:indexes[i+1]])
        else:
            intervals_abnormal.append(signal[indexes[i]:indexes[i+1]])
    return intervals_normal, intervals_abnormal


def __get_signal_index(record, signal_name):
    """
    Returns the index of the specified signal.
    :param record: record containing the signal
    :param signal_name: name of the signal
    :return: index of the signal within the record (-1 if the signal is not contained in the record)
    """
    signal_index = -1
    for i in range(len(record.sig_name)):
        if record.sig_name[i] == signal_name:
            signal_index = i
            break
    return signal_index
