from math import log
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Imported for legends
from matplotlib import collections as mc
import numpy as np
import pandas as pd
import seaborn as sns


def xlogx(x):
    """

    :param x:
    :type x:
    :return:
    :rtype:
    """
    """Returns xlogx for entropy calculation purposes"""
    return x * log(x) if x != 0 else 0


def truncate(x):
    """

    :param x:
    :type x:
    :return:
    :rtype:
    """
    """Truncates X to between 0 and 1"""
    return max(min(x, 1), 0)


def compute_lower_bound(
        primary_index, protected_class_prediction, outcome_prob, protected_class_status, outcome
):  # eq 35 of "Fairness using Data Combination"
    """

    :param primary_index:
    :type primary_index:
    :param protected_class_prediction:
    :type protected_class_prediction:
    :param outcome_prob:
    :type outcome_prob:
    :param protected_class_status:
    :type protected_class_status:
    :param outcome:
    :type outcome:
    :return:
    :rtype:
    """
    """Estimates the lower bound of the expected value of an outcome for a given protected class.
    Implements eq35 of https://arxiv.org/abs/1906.00285

    Args:
        primary_index (Series:bool): Boolean indicating if a row is in the primary dataset or auxiliary dataset
        protected_class_prediction (Series:bool): Boolean indicating if a row is estimated to belong to the protected class or not
        outcome_prob (Series:bool): Boolean indicating the model's expected prediction
        protected_class (Series:bool): Boolean indicating if a row is part of the protected class. This will be null for rows in the primary set.
        outcome (Series:bool): Boolean indicating if a row is part of the protected class. This will be null for rows in the primary set.

    Returns:
         float: Lower bound of the expected value of an outcome for a given protected class"""

    ind = outcome_prob + protected_class_prediction >= 1
    lbd = ind * (outcome_prob + protected_class_prediction - 1)
    xi = ind * (protected_class_status - protected_class_prediction)
    gamma = ind * (outcome - outcome_prob)
    return truncate((lbd.mean() + xi[~primary_index].mean() + gamma[primary_index].mean()) * (
                1 / protected_class_prediction.mean()))


def compute_upper_bound(primary_index, race_prob, outcome_prob, race, outcome):
    """

    :param primary_index:
    :type primary_index:
    :param race_prob:
    :type race_prob:
    :param outcome_prob:
    :type outcome_prob:
    :param race:
    :type race:
    :param outcome:
    :type outcome:
    :return:
    :rtype:
    """
    """Estimates the upper bound of the expected value of an outcome for a given protected class.
    Implements eq36 of https://arxiv.org/abs/1906.00285

    Args:
        primary_index (Series:bool): Boolean indicating if a row is in the primary dataset or auxiliary dataset
        race_prob (Series:bool): Boolean indicating if a row is estimated to belong to the protected class or not
        outcome_prob (Series:bool): Boolean indicating the model's expected prediction
        protected_class (Series:bool): Boolean indicating if a row is part of the protected class. This will be null for rows in the primary set.
        outcome (Series:bool): Boolean indicating if a row is part of the protected class. This will be null for rows in the primary set.

    Returns:
         float: Upper bound of the expected value of an outcome for a given protected class"""

    ind = outcome_prob - race_prob <= 0
    lbd = (ind * (outcome_prob - race_prob)) + race_prob  #
    xi = (1 - ind) * (race - race_prob)
    gamma = ind * (outcome - outcome_prob)
    return truncate((lbd.mean() + xi[~primary_index].mean() + gamma[primary_index].mean()) * (1 / race_prob.mean()))


def compute_expected_outcome(
        primary_index, protected_class_prediction, outcome_prob, protected_class_status, outcome
):
    """

    :param primary_index:
    :type primary_index:
    :param protected_class_prediction:
    :type protected_class_prediction:
    :param outcome_prob:
    :type outcome_prob:
    :param protected_class_status:
    :type protected_class_status:
    :param outcome:
    :type outcome:
    :return:
    :rtype:
    """
    """
    :rtype: object
    """
    return ((protected_class_prediction[primary_index] + outcome[primary_index]) == 2).mean() / \
           protected_class_prediction[primary_index].mean()


def make_proxy(col, scalar_mappable, **kwargs):
    """

    :rtype: Line2D
    """
    return Line2D([0, 1], [0, 1], linestyle='--', color=col, label='possible values', **kwargs)


def plot_intervals_0_1(title, protected_class_names, intervals):
    """

    :param title:
    :type title:
    :param protected_class_names:
    :type protected_class_names:
    :param intervals:
    :type intervals:
    :return:
    :rtype:
    """
    num_intervals = len(intervals)
    blues = plt.cm.get_cmap('Blues', num_intervals)
    colors = np.array([blues((idx + 30) / (num_intervals + 25)) for idx in range(len(intervals))])

    # Prepare the input data in correct format for LineCollection
    lines = [[(i[0], j), (i[1], j)] for i, j in zip(intervals, range(len(intervals)))]

    lc = mc.LineCollection(lines, colors=colors, linestyle='--', linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.margins(0.1)
    plt.yticks([], [])
    proxies = [make_proxy(c, lc) for c in colors]

    # Adding annotations
    for i, (lower, expected, upper) in enumerate(intervals):
        plt.text(lower, i + 0.1, protected_class_names[i], color=colors[i])
        plt.plot(lower, i, 'go')
        if expected:
            if i == 0:
                plt.plot(expected, i, 'x', markersize=10, color='black', label='expected value')
            else:
                plt.plot(expected, i, 'x', markersize=10, color='black')
        plt.plot(upper, i, 'go')
    plt.title(title, pad=20)
    plt.xlim(-0.05, 1.05)
    plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    handles.append(proxies[0])
    plt.legend(handles=handles)
    return plt.gcf()
