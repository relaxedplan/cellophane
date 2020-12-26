from math import log
import pandas as pd
import seaborn as sns


def xlogx(x):
    """Returns xlogx for entropy calculation purposes"""
    return x * log(x) if x != 0 else 0


def truncate(x):
    """Truncates X to between 0 and 1"""
    return max(min(x, 1), 0)


def compute_Wbar_L_hyy(
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
    print(race_prob.mean())
    return truncate((lbd.mean() + xi[~primary_index].mean() + gamma[primary_index].mean()) * (1/race_prob.mean()))


def compute_Wbar_U_hyy(primary_index, race_prob, outcome_prob, race, outcome):
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