# -*- coding: utf-8 -*-

__author__ = 'Matthias Wright'

import numpy as np
from sklearn.metrics import roc_curve, auc


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_auc(loss_normal, loss_abnormal):
    """
    Computes the area under the ROC curve (AUC), a means for evaluating medical tests.
    :param loss_normal: the losses produced by the test set containing normal data.
    :param loss_abnormal: the losses produced by the test set containing abnormal data.
    :return: (float) AUC
    """
    actual_normal = np.zeros(loss_normal.shape[0])
    actual_abnormal = np.ones(loss_abnormal.shape[0])
    prediction = np.append(loss_normal, loss_abnormal)
    prediction = _softmax(prediction - np.max(prediction))
    actual = np.append(actual_normal, actual_abnormal)
    index = np.argsort(prediction)
    prediction = np.flip(prediction[index], axis=0)
    actual = np.flip(actual[index], axis=0)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, prediction, pos_label=1)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return roc_auc