from math import log
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # Imported for legends
from matplotlib import collections as mc
import numpy as np
import pandas as pd
import seaborn as sns


def xlogx(x):
    """Returns xlogx for entropy calculation purposes"""
    return x * log(x) if x != 0 else 0


def truncate(x):
    """Truncates X to between 0 and 1"""
    return max(min(x, 1), 0)


def compute_lower_bound(
    primary_index, race_prob, outcome_prob, race, outcome
):  # eq 35 of "Fairness using Data Combination"
    """Estimates the lower bound of the expected value of an outcome for a given protected class.
    Implements eq35 of https://arxiv.org/abs/1906.00285

    Args:
        primary_index (Series:bool): Boolean indicating if a row is in the primary dataset or auxiliary dataset
        race_prob (Series:bool): Boolean indicating if a row is estimated to belong to the protected class or not
        outcome_prob (Series:bool): Boolean indicating the model's expected prediction
        protected_class (Series:bool): Boolean indicating if a row is part of the protected class. This will be null for rows in the primary set.
        outcome (Series:bool): Boolean indicating if a row is part of the protected class. This will be null for rows in the primary set.

    Returns:
         float: Lower bound of the expected value of an outcome for a given protected class"""

    ind = outcome_prob + race_prob >= 1
    lbd = ind * (outcome_prob + race_prob - 1)
    xi = ind * (race - race_prob)
    gamma = ind * (outcome - outcome_prob)
    return truncate((lbd.mean() + xi[~primary_index].mean() + gamma[primary_index].mean()) * (1/race_prob.mean()))


def compute_upper_bound(primary_index, race_prob, outcome_prob, race, outcome):
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
    return truncate((lbd.mean() + xi[~primary_index].mean() + gamma[primary_index].mean()) * (1/race_prob.mean()))

def compute_expected_outcome(
    primary_index, race_prob, outcome_prob, race, outcome
):
    return ((race_prob[primary_index] + outcome[primary_index])==2).mean() / \
           race_prob[primary_index].mean()

def make_proxy(col, scalar_mappable, **kwargs):
    color = col
    return Line2D([0, 1], [0, 1], linestyle='--', color=color, label='possible values', **kwargs)

def plot_intervals_0_1(title, protected_class_names, intervals, emphasis_points):

    num_intervals = len(intervals)
    blues = plt.cm.get_cmap('Blues', num_intervals)
    colors = np.array([blues((idx+30) / (num_intervals+25)) for idx in range(len(intervals))])

    # Prepare the input data in correct format for LineCollection
    lines = [[(i[0], j), (i[1], j)] for i, j in zip(intervals, range(len(intervals)))]

    lc = mc.LineCollection(lines, colors= colors, linestyle='--', linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.margins(0.1)
    plt.yticks([], [])
    proxies = [make_proxy(c, lc) for c in colors]

    # Adding annotations
    for i, x in enumerate(intervals):
        plt.text(x[0], i+0.1, protected_class_names[i], color=colors[i])
        plt.plot(intervals[i][0], i,'go')
        if emphasis_points:
            if i==0:
                plt.plot(emphasis_points[i], i,'x',markersize=10, color='black', label='expected value')
            else:
                plt.plot(emphasis_points[i], i,'x',markersize=10, color='black')
        plt.plot(intervals[i][1], i,'go')
    plt.title(title, pad=20)
    plt.xlim(-0.05,1.05)
    plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    handles.append(proxies[0])
    plt.legend(handles=handles)
    return plt.gcf()