from math import log
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Imported for legends
from matplotlib import collections as mc
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('fivethirtyeight')


def compute_lower_bound(
        primary_index, protected_class_prediction, outcome_prob, protected_class_status, outcome
):
    """
    Implements eq35 of https://arxiv.org/abs/1906.00285.

    Estimates the lower bound of the expected value of an outcome for a given protected class.
    This lower bound is calculated by assuming that all classification errors push the expected value lower
    and ignoring any classification errors which could push the expected value higher.

    Args:
        primary_index (pd.Series): Boolean int indicating if a row is in the primary dataset or auxiliary dataset
        protected_class_prediction (pd.Series): Boolean indicating if a row is predicted to fall within the predicted class based on proxies
        outcome_prob (pd.Series): Boolean int indicating the proxy model's estimate of the original model's prediction
        protected_class_status (pd.Series): Boolean int indicating if a row is part of the protected class. This will be null for rows in the primary set.
        outcome (pd.Series): Boolean indicating the original model's prediction

    Returns:
        float: Estimates the lower bound of the expected value of an outcome for a given protected class.
    """
    ind = outcome_prob + protected_class_prediction >= 1
    lbd = ind * (outcome_prob + protected_class_prediction - 1)
    xi = ind * (protected_class_status - protected_class_prediction)
    gamma = ind * (outcome - outcome_prob)
    return truncate((lbd.mean() + xi[~primary_index].mean() + gamma[primary_index].mean()) * (
                1 / protected_class_prediction.mean()))


def compute_upper_bound(primary_index, race_prob, outcome_prob, race, outcome):
    """
    Implements eq36 of https://arxiv.org/abs/1906.00285.

    Estimates the upper bound of the expected value of an outcome for a given protected class.
    This upper bound is calculated by assuming that all classification errors push the expected value higher
    and ignoring any classification errors which could push the expected value lower.

    Args:
        primary_index (pd.Series): Boolean int indicating if a row is in the primary dataset or auxiliary dataset
        protected_class_prediction (pd.Series): Boolean indicating if a row is predicted to fall within the predicted class based on proxies
        outcome_prob (pd.Series): Boolean int indicating the proxy model's estimate of the original model's prediction
        protected_class_status (pd.Series): Boolean int indicating if a row is part of the protected class. This will be null for rows in the primary set.
        outcome (pd.Series): Boolean indicating the original model's prediction

    Returns:
        float: Estimates the upper bound of the expected value of an outcome for a given protected class.
    """

    ind = outcome_prob - race_prob <= 0
    lbd = (ind * (outcome_prob - race_prob)) + race_prob  #
    xi = (1 - ind) * (race - race_prob)
    gamma = ind * (outcome - outcome_prob)
    return truncate((lbd.mean() + xi[~primary_index].mean() + gamma[primary_index].mean()) * (1 / race_prob.mean()))


def compute_expected_outcome(
        primary_index, protected_class_prediction, outcome_prob, protected_class_status, outcome
):
    """
    Returns the expected value of the outcome for the protected class.
    This expected value assumes that the proxy models capture both the underlying model and the unseen protected class information perfectly,
    providing a spurious point estimate.

    Args:
        primary_index (pd.Series): Boolean int indicating if a row is in the primary dataset or auxiliary dataset
        protected_class_prediction (pd.Series): Boolean indicating if a row is predicted to fall within the predicted class based on proxies
        outcome_prob (pd.Series): Boolean int indicating the proxy model's estimate of the original model's prediction
        protected_class_status (pd.Series): Boolean int indicating if a row is part of the protected class. This will be null for rows in the primary set.
        outcome (pd.Series): Boolean indicating the original model's prediction

    Returns:
        float: Estimates the lower bound of the expected value of an outcome for a given protected class.
    """
    return ((protected_class_prediction[primary_index] + outcome[primary_index]) == 2).mean() / \
           protected_class_prediction[primary_index].mean()


def make_proxy(col, **kwargs):

    """
    Helper function for graph legends
    Args:
        col (str): desired colour

    Returns:
        Line2D object
    """
    return Line2D([0, 1], [0, 1], linestyle='--', color=col, label='possible values', **kwargs)


def plot_intervals(title, protected_class_names, intervals, lower_limit=0):
    """
    Interface to plot partial identification sets
    Args:
        title (str): The title of the returned plot
        protected_class_names (list): Protected class names to be plotted
        intervals (tuple):
            A tuple corresponding to each protected class name.
            Should be ordered as follows: (lower_bound, expected_value, upper_bound)

    Returns:
        plt.figure: Plot of partial identification sets
    """

    if type(intervals[0])==dict:
        intervals = [list(i.values()) for i in intervals]
    num_intervals = len(intervals)
    blues = plt.cm.get_cmap('Blues', num_intervals)
    colors = np.array([blues((idx + 30) / (num_intervals + 25)) for idx in range(len(intervals))])

    # Prepare the input data in correct format for LineCollection
    lines = [[(i[0], j), (i[2], j)] for i, j in zip(intervals, range(len(intervals)))]

    lc = mc.LineCollection(lines, colors=colors, linestyle='--', linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.margins(0.1)
    plt.yticks([], [])
    proxies = [make_proxy(c) for c in colors]

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
    plt.xlim(lower_limit-0.05, 1.05)
    plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    handles.append(proxies[0])
    plt.legend(handles=handles)
    fig = plt.gcf()
    return fig


def xlogx(x):
    """
    helper function for entropy calculations
    Args:
        float: x
    Returns:
        float: x
    """
    return x * log(x) if x != 0 else 0

def truncate(x):
    """
    Truncates x to between 0 and 1
    Args:
        float: x
    Returns:
        float: x
    """
    return max(min(x, 1.0), 0.0)