import itertools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from . import metrics
from .constants import *
from .utils import *


class Cellophane:
    def __init__(
            self,
            primary_dataset,
            auxiliary_dataset,
            primary_feature_cols,
            proxy_cols,
            primary_ground_truth_col,
            protected_class_col,
            prediction_col,

            estimator=RandomForestClassifier
    ):
        """
    Args:
        primary_dataset: (DataFrame)
            The primary dataset containing features, ground truth, and prediction
        auxiliary_dataset: (DataFrame):
            The secondary dataset containing proxy variables and protected class data
        primary_feature_cols: (list)
            Feature names, which should all exist in the primary dataset
        proxy_cols: (list)
            Column names indicating proxy variables shared between primary and secondary dataset.
        primary_ground_truth_col: (str)
            Column indicating ground truth in the primary dataset
        protected_class_col: (str)
            Column name indicating protected class status in the secondary dataset
        prediction_col: (str)
            Column name indicating model's prediction in the primary dataset
    Notes:
        Assessments of bias along protected class lines are usually challenged by the fact that granular protected class
        information is not available within the data.

        One quantitative method to address the absence of protected class data is to combine the primary dataset (
        which contains features, ground truth, and model predictions) with
        another auxiliary dataset which contains protected class information, features which overlap with the primary set
        (called proxies) but not ground truth or model predictions

        Naively combining the these two sets will lead to a fundamentally spurious point estimate for any bias metric.

        The algorithms in this package provide partial identification sets for common bias metrics in a binary
        classification context. These partial identification sets contain *all possible* values for a bias metric,
        subject to the weak assumptions made at https://arxiv.org/pdf/1906.00285.pdf.

        Typically the stronger your proxies, the smaller the partial identification sets will be, reflecting
        increased certainty.


    The Cellophane object contains methods for estimating bias metrics along protected class lines,
    when the protected class information can't be directly observed in the available data.

    The object is based around a primary dataset and an auxiliary dataset.
    The primary set contains features, a target, and/or the estimates for that target based on a
    machine learning model.

    The auxiliary set contains neither the target nor an estimate for that target, but it contains
    protected class data. It should contain at least some columns which overlap with the primary set (these are
    termed "proxy" columns)


    References:
        [1] BISG proxy methodology (https://github.com/cfpb/proxy-methodology)

        [2] Fairness Under Unawareness: Assessing Disparity When Protected Class Is Unobserved (
        https://arxiv.org/abs/1811.11154)

        [3] Assessing Algorithmic Fairness with Unobserved Protected Class Using Data Combination (
        https://arxiv.org/abs/1906.00285)

        """
        assert primary_ground_truth_col in primary_dataset.columns
        assert all(f in primary_dataset.columns for f in primary_feature_cols)
        assert all(p in primary_dataset.columns for p in proxy_cols)
        assert all(p in auxiliary_dataset.columns for p in proxy_cols)
        assert protected_class_col in auxiliary_dataset.columns

        self.primary_dataset = primary_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.primary_ground_truth_col = primary_ground_truth_col
        self.primary_features_col = primary_feature_cols
        self.protected_class_col = protected_class_col
        self.proxies = proxy_cols
        self.prediction_col = prediction_col
        self.protected_class_names = self.primary_dataset[protected_class_col].unique()
        self.protected_class_names.sort()
        self.combined = pd.concat([self.primary_dataset, self.auxiliary_dataset]).reset_index()
        self.estimator = estimator

        self.Y = self.combined[self.primary_ground_truth_col]
        self.Yhat = self.combined[self.prediction_col]
        self.primary_index = pd.Series([True] * len(self.primary_dataset) + [False] * len(self.auxiliary_dataset))
        (self.Y_from_proxies, self.model_prediction_from_proxies,
         self.protected_class_membership, self.protected_class_prob) = self.build_estimators_from_proxies()


        self.quadrant_dict = self.generate_binary_classification_quadrants()
        self.hemisphere_readings = self.generate_hemisphere_statistics_by_protected_class()
        self.quadrant_readings = self.generate_quadrant_statistics_by_protected_class()
        self.tpr_tnr, self.ppv_npv = self.generate_tpr_tnr_ppv_npv_by_protected_class()

    def generate_binary_classification_quadrants(self):
        """
        Precalculates binary classification quadrants (true positive rates, true negative rates, false positive
        rates, false negative rate)
        Args:
            self (Cellophane object): base object
        Returns dictionary with true positive rates, true negative rates, false positive
        rates, false negative rates
        """
        TT = ((self.Y_from_proxies).astype(bool) & (self.model_prediction_from_proxies).astype(bool)
              ).astype(int)
        TF = ((self.Y_from_proxies).astype(bool) & (1 - self.model_prediction_from_proxies).astype(bool)
              ).astype(int)
        FT = ((1 - self.Y_from_proxies).astype(bool) & (self.model_prediction_from_proxies).astype(bool)
              ).astype(int)
        FF = ((1 - self.Y_from_proxies).astype(bool) & (1 - self.model_prediction_from_proxies).astype(bool)
              ).astype(int)
        return {
            (NEGATIVE, NEGATIVE): FF,
            (NEGATIVE, POSITIVE): FT,
            (POSITIVE, NEGATIVE): TF,
            (POSITIVE, POSITIVE): TT,
        }

    def generate_hemisphere_statistics_by_protected_class(self):
        """
        Precalculates confusion matrix hemispheres (positive rates, predicted positive rates, negative rates,
        predicted negative rates)

        Args:
            self: Cellophane

        Returns:
            Dictionary with positive rate, negative rate, predicted positive rate, predicted negative rate
        """
        bounds = [LOWER_BOUND, UPPER_BOUND, EXPECTATION]
        hemisphere_component_tuples= list(
            itertools.product(
                self.protected_class_names,
                bounds,
                [NEGATIVE, POSITIVE],
                [IN_PROTECTED_CLASS, NOT_IN_PROTECTED_CLASS],
                [GROUND_TRUTH, MODEL_PREDICTION]
            )
        )
        return {c: self.partial_identification_hemisphere_bounds_for_protected_class(*c)
                                 for c in hemisphere_component_tuples}

    def generate_quadrant_statistics_by_protected_class(self):
        """
        Precalculates confusion matrix quadrants (true positives, true negatives,
        false positives and false negatives)

        Args:
            self: Cellophane

        Returns:
            Dictionary with true positives, true negatives, false positives, and false negatives {}
        """
        bounds = [LOWER_BOUND, UPPER_BOUND, EXPECTATION]
        quadrant_component_tuples = list(
            itertools.product(
                self.protected_class_names, bounds, [NEGATIVE, POSITIVE], [NEGATIVE, POSITIVE],
                [IN_PROTECTED_CLASS, NOT_IN_PROTECTED_CLASS]
            )
        )

        return {c: self.partial_identification_quadrant_bounds_for_protected_class(*c)
                               for c in quadrant_component_tuples}


    def generate_tpr_tnr_ppv_npv_by_protected_class(self):
        """
        Calculates all partial identification sets for TPR, TNR, PPV and NPV by protected class

        Args:
            self: Cellophane
        """
        tpr_tnr = {}
        ppv_npv = {}
        component_tuples = list(
            itertools.product(
                self.protected_class_names, [LOWER_BOUND, UPPER_BOUND, EXPECTATION], [NEGATIVE, POSITIVE],
                [NEGATIVE, POSITIVE], [IN_PROTECTED_CLASS, NOT_IN_PROTECTED_CLASS]
            )
        )
        for component_tuple in component_tuples:
            selected_protected_class, bound, truth, pred, in_class = component_tuple
            if bound == LOWER_BOUND:  # see (38) of https://arxiv.org/pdf/1906.00285.pdf
                num_start = LOWER_BOUND
                denom_start = LOWER_BOUND
                denom_end = UPPER_BOUND

            elif bound == UPPER_BOUND:
                num_start = UPPER_BOUND
                denom_start = UPPER_BOUND
                denom_end = LOWER_BOUND

            elif bound == EXPECTATION:
                num_start = EXPECTATION
                denom_start = EXPECTATION
                denom_end = EXPECTATION

            ############TPRD_TPND_CALCULATION#############

            numerator = truncate(
                self.quadrant_readings[selected_protected_class, num_start, truth, pred, in_class])
            denominator = (truncate(self.quadrant_readings[selected_protected_class, denom_start, truth, pred, in_class])
                           + truncate(
                        self.quadrant_readings[selected_protected_class, denom_end, truth, 1 - pred, in_class]))

            if denominator == 0:
                tpr_tnr[component_tuple] = 1
            else:
                tpr_tnr[component_tuple] = numerator / denominator

            ############PPVD_NPVD_CALCULATION#############

            numerator = truncate(
                self.quadrant_readings[selected_protected_class, num_start, truth, pred, in_class])
            denominator = (
                    truncate(self.quadrant_readings[selected_protected_class, denom_start, truth, pred, in_class])
                    + truncate(self.quadrant_readings[selected_protected_class, denom_end, 1 - truth, pred, in_class]))

            if denominator == 0:
                ppv_npv[component_tuple] = 1
            else:
                ppv_npv[component_tuple] = numerator / denominator

        return tpr_tnr, ppv_npv

    def partial_identification_quadrant_bounds_for_protected_class(self, protected_class_name, bound, truth, predicted, in_class=IN_PROTECTED_CLASS):
        """
        Calculates a specified bound for a specified confusion matrix quadrant for a specified protected class

        Parameters
        ----------
        self: Cellophane
            Base object containing primary and auxiliary datasets
        protected_class_name: str
            Name of protected class
        bound: str
            One of "lower" or "upper"
        truth: pd.Series
            Series which represents the ground truth, or model's prediction, for the protected class
        predicted: pd.Series
            Series which represents the estimate of ground truth, or estimate of model's prediction, for the protected
            class, based on proxy variables only
        in_class: bool
            If True, provide estimate only for the protected class. If False, provide estimate for all samples which
            are not in the protected class.
        Notes
        -----

        """
        if bound == LOWER_BOUND:
            f = compute_lower_bound
        elif bound == UPPER_BOUND:
            f = compute_upper_bound
        elif bound == EXPECTATION:
            f = compute_expected_outcome
        if in_class:
            partial_identification_input = {
                'primary_index': self.primary_index,
                'protected_class_prediction_from_proxies':self.protected_class_prob[protected_class_name],
                'outcome_prediction_from_proxies':self.quadrant_dict[(truth, predicted)],
                'protected_class_status':self.protected_class_membership[protected_class_name],
                'predicted_outcome':(self.Yhat == predicted) & (self.Y == truth)
            }
        else:
            partial_identification_input = {
                'primary_index': self.primary_index,
                'protected_class_prediction_from_proxies':1  - self.protected_class_prob[protected_class_name],
                'outcome_prediction_from_proxies':self.quadrant_dict[(truth, predicted)],
                'protected_class_status':1 - self.protected_class_membership[protected_class_name],
                'predicted_outcome':(self.Yhat == predicted) & (self.Y == truth)
            }
        return f(partial_identification_input['primary_index'],
                 partial_identification_input['protected_class_prediction_from_proxies'],
                 partial_identification_input['outcome_prediction_from_proxies'],
                 partial_identification_input['protected_class_status'],
                 partial_identification_input['predicted_outcome'],
                 )

    def partial_identification_hemisphere_bounds_for_protected_class(self, protected_class_name, bound, truth_or_prediction, in_class=IN_PROTECTED_CLASS,
                                 target='prediction'):
        """
        Calculates a specified bound for a specified hemisphere for a specified protected class

        Parameters
        ----------
        self: Cellophane
            Base object containing primary and auxiliary datasets
        protected_class_name: str
            Name of protected class
        bound: str
            One of "lower" or "upper"
        truth_or_prediction: pd.Series
            Series which represents the ground truth, or model's prediction, for the protected class
        in_class: bool
            If True, provide estimate only for the protected class. If False, provide estimate for all samples which
            are not in the protected class.
        Notes
        -----

        """
        if bound == LOWER_BOUND:
            f = compute_lower_bound
        elif bound == UPPER_BOUND:
            f = compute_upper_bound
        elif bound == EXPECTATION:
            f = compute_expected_outcome

        partial_identification_input = {'primary_index': self.primary_index,
                                        'protected_class_prediction_from_proxies': None,
                                        'outcome_prediction_from_proxies': None,
                                        'protected_class_status': None,
                                        'predicted_outcome': None}
        if target == 'prediction':
            partial_identification_input['outcome_prediction_from_proxies'] = (
                        self.model_prediction_from_proxies == truth_or_prediction).astype(int)
            partial_identification_input['predicted_outcome'] = (self.Yhat == truth_or_prediction).astype(int)
        else:
            partial_identification_input['outcome_prediction_from_proxies'] = (
                        self.Y_from_proxies == truth_or_prediction).astype(int)
            partial_identification_input['predicted_outcome'] = (self.Y == truth_or_prediction).astype(int)

        if in_class:
            partial_identification_input['protected_class_prediction_from_proxies'] = (
                        self.protected_class_prob[protected_class_name]).astype(int)
            partial_identification_input['protected_class_status'] = (
                        self.protected_class_membership[protected_class_name]).astype(int)
        else:
            partial_identification_input['protected_class_prediction_from_proxies'] = (
                        1 - self.protected_class_prob[protected_class_name]).astype(int)
            partial_identification_input['protected_class_status'] = (
                        1 - self.protected_class_membership[protected_class_name]).astype(int)

        return f(partial_identification_input['primary_index'],
                 partial_identification_input['protected_class_prediction_from_proxies'],
                 partial_identification_input['outcome_prediction_from_proxies'],
                 partial_identification_input['protected_class_status'],
                 partial_identification_input['predicted_outcome'],
                 )

    def build_estimators_from_proxies(
            self, n_k=5):
        """
        Parameters
        ----------
        self: Cellophane
            Base object containing primary and auxiliary datasets
        n_k: int
            The number of splits used for cross-validation.
        estimator: sklearn.base.BaseEstimator
            Base estimator class. Must implement .fit and .predict_proba
        Notes
        -----
        This function implements Steps 1-8 of Algorithm 1 of https://arxiv.org/pdf/1906.00285.pdf.
        To create partial estimation sets, new estimators need to be made to estimate the target,
        the models' prediction of the target, and the protected class, for each row.
        These estimators are fitted on the proxy variables as described in steps 6 and 7 of Algorithm 1.
        For each estimation target (target, prediction, and protected class), predicted values are added directly
        to the class.
        """

        """1. Divide Primary and Auxiliary datasets into K subsets"""

        K_pri, K_aux = np.array_split(self.primary_dataset, n_k), \
                       np.array_split(self.auxiliary_dataset, n_k)

        """2. Define models to learn the ground truth, model's prediction, and protected class status
         from proxies.
        Recall that the ground truth and model's predicion are only available for the primary set,
        and protected class status is only available for the auxiliary set"""
        rfcs_y_pri = [self.estimator() for idx in range(n_k)]
        rfcs_yhat_pri = [self.estimator() for idx in range(n_k)]
        rfcs_protected_class_sec = [self.estimator() for idx in range(n_k)]

        """3. Train K models for each of the K subsets"""
        for idx in range(n_k):  # train k models on each of k sets
            rfcs_y_pri[idx].fit(K_pri[idx][self.proxies], K_pri[idx][self.primary_ground_truth_col])
            rfcs_yhat_pri[idx].fit(K_pri[idx][self.proxies], K_pri[idx][self.prediction_col])
            rfcs_protected_class_sec[idx].fit(K_aux[idx][self.proxies], K_aux[idx][self.protected_class_col])

        """4. Use all ~k models to infer ground truth, model's prediction, and protected class status on the kth set."""
        for idx in range(n_k):
            range_excluding_idx = [n for n in range(n_k) if n != idx]
            K_pri[idx][GROUND_TRUTH_FROM_PROXIES] = np.asarray(
                [rfcs_y_pri[notk_idx].predict_proba(K_pri[idx][self.proxies])[:, 1] for notk_idx in
                 range_excluding_idx]).mean(axis=0)
            K_aux[idx][GROUND_TRUTH_FROM_PROXIES] = np.asarray(
                [rfcs_y_pri[notk_idx].predict_proba(K_aux[idx][self.proxies])[:, 1] for notk_idx in
                 range_excluding_idx]).mean(axis=0)
            K_pri[idx][MODEL_PREDICTION_FROM_PROXIES] = np.asarray(
                [rfcs_yhat_pri[notk_idx].predict_proba(K_pri[idx][self.proxies])[:, 1] for notk_idx in
                 range_excluding_idx]).mean(axis=0)
            K_aux[idx][MODEL_PREDICTION_FROM_PROXIES] = np.asarray(
                [rfcs_yhat_pri[notk_idx].predict_proba(K_aux[idx][self.proxies])[:, 1] for notk_idx in
                 range_excluding_idx]).mean(axis=0)
            K_pri[idx][PROTECTED_CLASS_FROM_PROXIES] = np.asarray(
                [rfcs_protected_class_sec[notk_idx].predict_proba(K_pri[idx][self.proxies]) for notk_idx in
                 range_excluding_idx]).mean(axis=0).tolist()
            K_aux[idx][PROTECTED_CLASS_FROM_PROXIES] = np.asarray(
                [rfcs_protected_class_sec[notk_idx].predict_proba(K_aux[idx][self.proxies]) for notk_idx in
                 range_excluding_idx]).mean(axis=0).tolist()

        """5. recombine k subsets into a combined set"""
        combined = pd.concat(K_pri + K_aux).reset_index()

        """6a. record proxy model's prediction of ground truth."""
        Y_from_proxies = combined[GROUND_TRUTH_FROM_PROXIES].round(0)

        """6b. record proxy model's prediction of model's prediction."""
        model_prediction_from_proxies = combined[MODEL_PREDICTION_FROM_PROXIES].round(0)


        """6c. record model-based prediction of protected class membership for each protected class"""
        protected_class_prob = {}
        for idx, protected_class in enumerate(rfcs_protected_class_sec[0].classes_):
            protected_class_prob[protected_class] = \
                combined[PROTECTED_CLASS_FROM_PROXIES].apply(lambda x: x[idx]).reset_index()[
                    PROTECTED_CLASS_FROM_PROXIES].round(0)

        """6d. record actual protected class membership for each protected class."""
        protected_class_membership = {}
        for idx, protected_class in enumerate(rfcs_protected_class_sec[0].classes_):
            protected_class_membership[protected_class] = np.hstack([
                np.empty(shape=len(self.primary_dataset),),
                (self.auxiliary_dataset[self.protected_class_col] == protected_class).values])

        return Y_from_proxies, model_prediction_from_proxies, protected_class_membership, protected_class_prob


    def positive_or_negative_hemisphere_reading(self, protected_class, result=POSITIVE, variable=MODEL_PREDICTION,
                          within_protected_class=True):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
                        metrics.positive_or_negative_hemisphere_reading(self.hemisphere_readings, protected_class, result, variable,
                                                               within_protected_class)))

    def confusion_matrix_quadrant_reading(self, protected_class, truth=POSITIVE, prediction=POSITIVE,
                        within_protected_class=True):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
            metrics.confusion_matrix_quadrant_reading(self.hemisphere_readings, protected_class, truth, prediction,
                                                         within_protected_class)))

    def tpr_reading(self, protected_class):
        """For documentation see metrics.py"""
        return dict((zip(('lower_bound','expected_value','upper_bound'),
                        metrics.tpr_reading(self.tpr_tnr, protected_class))))

    def tnr_reading(self, protected_class):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
                        metrics.tnr_reading(self.tpr_tnr, protected_class)))

    def npv_reading(self, protected_class):
        """For documentation see metrics.py"""
        return dict(zip('lower_bound','expected_value','upper_bound',
                        metrics.npv_reading(self.ppv_npv, protected_class)))

    def ppv_reading(self, protected_class):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
                        metrics.ppv_reading(self.ppv_npv, protected_class)))

    def positive_or_negative_hemisphere_disparity(self, protected_class, comparison_class=OTHERS, variable=MODEL_PREDICTION, value=POSITIVE):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
                        metrics.positive_or_negative_hemisphere_disparity(self.hemisphere_readings,protected_class, comparison_class, variable, value)))

    def confusion_matrix_quadrant_disparity(self, protected_class, truth=POSITIVE, prediction=POSITIVE, comparison_class=OTHERS):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
                        metrics.confusion_matrix_quadrant_disparity(self.quadrant_readings,protected_class, truth, prediction, comparison_class)))

    def tpr_disparity(self, protected_class, comparison_class=OTHERS):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
                        metrics.tpr_disparity(self.tpr_tnr, protected_class, comparison_class)))

    def tnr_disparity(self, protected_class, comparison_class=OTHERS):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
                        metrics.tnr_disparity(self.tpr_tnr, protected_class, comparison_class)))

    def ppv_disparity(self, protected_class, comparison_class=OTHERS):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
                        metrics.ppv_disparity(self.ppv_npv, protected_class, comparison_class)))

    def npv_disparity(self, protected_class, comparison_class=OTHERS):
        """For documentation see metrics.py"""
        return dict(zip(('lower_bound','expected_value','upper_bound'),
                   metrics.npv_disparity(self.ppv_npv, protected_class, comparison_class)))