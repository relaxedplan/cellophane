import itertools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from utils import compute_Wbar_L_hyy, compute_Wbar_U_hyy, truncate


class PartialIdentification:
    """The use of a PartialIdentification object is to determine bias metrics
    in a model along protected class lines, where the protected class data is not available.

    The object is based around a primary dataset and an auxiliary dataset.
    The primary set contains both a target and the estimates for that target based on a
    machine learning model.

    The secondary set contains neither the target nor an estimate for that target, but it contains
    protected class data.

    args:
        primary_dataset (DataFrame): The primary dataset containing features, ground truth, and prediction
        auxiliary_dataset (DataFrame): The secondary dataset containing proxy variables and protected class data
        primary_ground_truth_col (str, int): Column indicating ground truth in the primary dataset
        primary_features_col (list(str, int)): Feature names, which should all exist in the primary dataset
        protected_class_col (str,int): Column name indicating protected class status in the secondary dataset
        prediction_col (str,int): Column name indicating model's prediction in the primary dataset
        proxies_col list(str, int): Column names indicating proxy variables shared between primary and secondary dataset
    """

    def __init__(
            self,
            primary_dataset,
            auxiliary_dataset,
            primary_ground_truth_col,
            primary_features_col,
            protected_class_col,
            prediction_col,
            proxies_col,
    ):
        assert primary_ground_truth_col in primary_dataset.columns
        assert all(f in primary_dataset.columns for f in primary_features_col)
        assert all(p in primary_dataset.columns for p in proxies_col)
        assert all(p in auxiliary_dataset.columns for p in proxies_col)
        assert protected_class_col in auxiliary_dataset.columns

        self.primary_dataset = primary_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.primary_ground_truth_col = primary_ground_truth_col
        self.primary_features_col = primary_features_col
        self.protected_class_col = protected_class_col
        self.proxies = proxies_col
        self.prediction_col = prediction_col
        self.protected_class_names = self.primary_dataset[protected_class_col].unique()
        self.protected_class_names.sort()
        self.combined = pd.concat([self.primary_dataset, self.auxiliary_dataset]).reset_index()
        self.Y = None
        self.Yhat = None
        self.primary_index = None
        self.tprd_wbar_quadrants = None
        self.tprd_tpnd_components = None

        self.Y = self.combined[self.primary_ground_truth_col]
        self.Yhat = self.combined[self.prediction_col]
        self.primary_index = pd.Series(
            [True] * len(self.primary_dataset) + [False] * len(self.auxiliary_dataset)
        )

        self.build_estimators_from_proxies()
        (self.Y_from_proxies, self.Yhat_from_proxies,
         self.protected_class_membership, self.protected_class_prob) = self.build_estimators_from_proxies()
        self.Y = self.combined[self.primary_ground_truth_col]
        self.Yhat = self.combined[self.prediction_col]
        self.primary_index = pd.Series([True] * len(self.primary_dataset) + [False] * len(self.auxiliary_dataset))
        self.quadrant_dict = self.generate_binary_classification_quadrants()
        self.generate_wbars()
        self.create_plugin_estimator_components()

    def build_estimators_from_proxies(
            self, n_k=5, estimator=RandomForestClassifier
    ):
        """Implements Steps 1-8 of Algorithm 1 of https://arxiv.org/pdf/1906.00285.pdf.
        To create partial estimation sets, new estimators need to be made to estimate the target,
        the models' prediction of the target, and the protected class, for each row.
        These estimators are fitted on the proxy variables as described in steps 6 and 7 of Algorithm 1.
        For each estimation target (target, prediction, and protected class), predicted values are added directly
        to the class.

        Args:
            PartialIdentification objects
            n_k: The number of splits used for cross-validation.
            Estimator will be trained on each individual split and ensembled for each unseen split
            estimator: The base estimator used for each split as in steps 6 and 7 of Algorithm 1

        Returns: None"""

        primary_shuffled = self.primary_dataset  # .sample(frac=1)
        secondary_shuffled = self.auxiliary_dataset  # .sample(frac=1)
        K_pri, K_sec = np.array_split(primary_shuffled, n_k), np.array_split(secondary_shuffled, n_k)
        rfcs_y_pri = [estimator() for idx in range(n_k)]
        rfcs_yhat_pri = [estimator() for idx in range(n_k)]
        rfcs_protected_class_sec = [estimator() for idx in range(n_k)]

        for idx in range(n_k):  # train k models on each of k sets
            rfcs_y_pri[idx].fit(K_pri[idx][self.proxies], K_pri[idx][self.primary_ground_truth_col])
            rfcs_yhat_pri[idx].fit(K_pri[idx][self.proxies], K_pri[idx][self.prediction_col])
            rfcs_protected_class_sec[idx].fit(K_sec[idx][self.proxies], K_sec[idx][self.protected_class_col])

        for idx in range(n_k):  # infer on each of k sets using ensemble of ~k models
            range_excluding_idx = [n for n in range(n_k) if n != idx]
            K_pri[idx]['y_from_proxies'] = np.asarray(
                [rfcs_y_pri[notk_idx].predict_proba(K_pri[idx][self.proxies])[:, 1] for notk_idx in
                 range_excluding_idx]).mean(axis=0)
            K_sec[idx]['y_from_proxies'] = np.asarray(
                [rfcs_y_pri[notk_idx].predict_proba(K_sec[idx][self.proxies])[:, 1] for notk_idx in
                 range_excluding_idx]).mean(axis=0)
            K_pri[idx]['yhat_from_proxies'] = np.asarray(
                [rfcs_yhat_pri[notk_idx].predict_proba(K_pri[idx][self.proxies])[:, 1] for notk_idx in
                 range_excluding_idx]).mean(axis=0)
            K_sec[idx]['yhat_from_proxies'] = np.asarray(
                [rfcs_yhat_pri[notk_idx].predict_proba(K_sec[idx][self.proxies])[:, 1] for notk_idx in
                 range_excluding_idx]).mean(axis=0)
            K_pri[idx]['protected_class_from_proxies'] = np.asarray(
                [rfcs_protected_class_sec[notk_idx].predict_proba(K_pri[idx][self.proxies]) for notk_idx in
                 range_excluding_idx]).mean(axis=0).tolist()
            K_sec[idx]['protected_class_from_proxies'] = np.asarray(
                [rfcs_protected_class_sec[notk_idx].predict_proba(K_sec[idx][self.proxies]) for notk_idx in
                 range_excluding_idx]).mean(axis=0).tolist()
        combined = pd.concat(K_pri + K_sec).reset_index()

        protected_class_prob = {}
        for idx, protected_class in enumerate(rfcs_protected_class_sec[0].classes_):
            protected_class_prob[protected_class] = \
                combined['protected_class_from_proxies'].apply(lambda x: x[idx]).reset_index()[
                    'protected_class_from_proxies'].round(0)
        protected_class_membership = {}  # true protected classes
        for idx, protected_class in enumerate(rfcs_protected_class_sec[0].classes_):
            protected_class_membership[protected_class] = pd.concat([
                (self.primary_dataset[self.protected_class_col] == protected_class),
                (self.auxiliary_dataset[self.protected_class_col] == protected_class)]) \
                .reset_index()[self.protected_class_col]

        Y_from_proxies = combined['y_from_proxies'].round(0)
        Yhat_from_proxies = combined['yhat_from_proxies'].round(0)
        protected_class_membership = protected_class_membership
        protected_class_prob = protected_class_prob

        return Y_from_proxies, Yhat_from_proxies, protected_class_membership, protected_class_prob

    def generate_binary_classification_quadrants(self):
        """Precalculates confusion matrix quadrants (true positives, true negatives,
        false positives and false negatives)

        Args:
            PartialIdentification object

        Returns:
            Dictionary with true positives, true negatives, false positives, and false negatives {}"""
        TT = ((self.Y_from_proxies).astype(bool) & (self.Yhat_from_proxies).astype(bool)
        ).astype(int)
        TF = ((self.Y_from_proxies).astype(bool) & (1 - self.Yhat_from_proxies).astype(bool)
        ).astype(int)
        FT = ((1 - self.Y_from_proxies).astype(bool) & (self.Yhat_from_proxies).astype(bool)
        ).astype(int)
        FF = ((1 - self.Y_from_proxies).astype(bool) & (1 - self.Yhat_from_proxies).astype(bool)
        ).astype(int)
        return {
            (0, 0): FF,
            (0, 1): FT,
            (1, 0): TF,
            (1, 1): TT,
        }

    def dd_intervals_manual(self, protected_class_name):
        """For a given protected class, calculate the partial identification set of
         Demographic Disparity / Disparate Impact difference for the one selected protected class vs the rest.

         For example, if the partial identification set is (0.2,0.4), then the selected protected class has an average
         rate of positive predictions between 20% and 40% more than the rest."""
        if protected_class_name == 'Black':
            print('hi')
        one_input_tuple = (
            self.primary_index,
            self.protected_class_prob[protected_class_name],
            self.Yhat_from_proxies,
            self.protected_class_membership[protected_class_name],
            (self.Yhat == 1),
        )
        lower_positive, upper_positive = (
            compute_Wbar_L_hyy(*one_input_tuple),
            compute_Wbar_U_hyy(*one_input_tuple),
        )
        print('one', protected_class_name, lower_positive, upper_positive)
        rest_input_tuple = (
            self.primary_index,
            1 - self.protected_class_prob[protected_class_name],
            self.Yhat_from_proxies,
            1 - self.protected_class_membership[protected_class_name],
            (self.Yhat == 1),
        )
        lower_negative, upper_negative = (
            compute_Wbar_L_hyy(*rest_input_tuple),
            compute_Wbar_U_hyy(*rest_input_tuple),
        )
        print('rest', protected_class_name, lower_negative, upper_negative)
        return lower_positive - upper_negative, upper_positive - lower_negative

    def generate_wbars(self):
        """For each """
        bounds = ["lower", "upper"]
        quadrant_combos = list(
            itertools.product(
                self.protected_class_names, bounds, [0, 1], [0, 1], [True, False]
            )
        )
        hemisphere_combos = list(
            itertools.product(
                self.protected_class_names, bounds, [0, 1], [True, False], ['truth','prediction']
            )
        )
        self.tprd_wbar_quadrants = {c: self.compute_wbar_quadrants(*c) for c in quadrant_combos}
        self.tprd_wbar_hemispheres = {c: self.compute_wbar_hemispheres(*c) for c in hemisphere_combos}


    def tprd_intervals(self, protected_class):
        """For a given protected class, calculate the partial identification set of
         True Positive Rate Disparity for the one selected protected class vs the rest.

         For example, if the partial identification set is (0.2,0.4), then the selected protected class has an average
         true positive rate between 0.2 and 0.4 more than the rest."""
        return (
            self.tprd_tpnd_components[protected_class, "LU", 1, 1, False]
            - self.tprd_tpnd_components[protected_class, "UL", 1, 1, True],
            self.tprd_tpnd_components[protected_class, "UL", 1, 1, False]
            - self.tprd_tpnd_components[protected_class, "LU", 1, 1, True],
        )

    def tnrd_intervals(self, protected_class):
        """For a given protected class, calculate the partial identification set of
         True Negative Rate Disparity for the one selected protected class vs the rest.

         For example, if the partial identification set is (0.2,0.4), then the selected protected class has an average
         true negative rate between 0.2 and 0.4 more than the rest."""
        return (
            self.tprd_tpnd_components[protected_class, "LU", 0, 0, False]
            - self.tprd_tpnd_components[protected_class, "UL", 0, 0, True],
            self.tprd_tpnd_components[protected_class, "UL", 0, 0, False]
            - self.tprd_tpnd_components[protected_class, "LU", 0, 0, True],
        )

    def create_plugin_estimator_components(self):
        """
        Combines precalculated components of TPRD and TPND partial identification sets
         into the upper and lower bounds of the partial identification sets
        :return:
        """
        self.tprd_tpnd_components = {}
        """Each combination specifies bound, truth, prediction, and inverse. 
        A bound of "LU" is the lower bound for TPND or TPRD.
        A bound of "UL" is the lower bound for TPND or TPRD.
        If inverse is set to False, the bound is calculated for the one selected protected class.
        If inverse is set to True, the bound is calculated for the rest of the selected protected classes.
        See equations (14), (15), (37), and (38) of https://arxiv.org/pdf/1906.00285.pdf for the derivation 
        of these bounds"""
        combos = list(
            itertools.product(
                self.protected_class_names, ["LU", "UL"], [0, 1], [0, 1], [True, False]
            )
        )
        for c in combos:
            selected_protected_class, bound, truth, pred, inverse = c
            if bound == "LU":  # see (38) of https://arxiv.org/pdf/1906.00285.pdf
                num_start = "lower"
                denom_start = "lower"
                denom_end = "upper"
                if (truncate(self.tprd_wbar_quadrants[selected_protected_class, denom_start, truth, pred, inverse])
                        + truncate(self.tprd_wbar_quadrants[selected_protected_class, denom_end, truth, 1 - pred, inverse])) == 0:
                    self.tprd_tpnd_components[c] = 1
                else:
                    self.tprd_tpnd_components[c] = truncate(
                        self.tprd_wbar_quadrants[selected_protected_class, num_start, truth, pred, inverse]) / \
                                                   (truncate(self.tprd_wbar_quadrants[
                                                     selected_protected_class, denom_start, truth, pred, inverse])
                                        + truncate(self.tprd_wbar_quadrants[
                                                       selected_protected_class, denom_end, truth, 1 - pred, inverse]))
            elif bound == "UL":  # see (38) of https://arxiv.org/pdf/1906.00285.pdf
                num_start = "upper"
                denom_start = "upper"
                denom_end = "lower"
                if (truncate(self.tprd_wbar_quadrants[selected_protected_class, denom_start, truth, pred, inverse]) +
                    truncate(self.tprd_wbar_quadrants[selected_protected_class, denom_end, truth, 1 - pred, inverse])) == 0:
                    self.tprd_tpnd_components[c] = 1
                else:
                    self.tprd_tpnd_components[c] = truncate(
                        self.tprd_wbar_quadrants[selected_protected_class, num_start, truth, pred, inverse]) / \
                                                   (truncate(self.tprd_wbar_quadrants[
                                                 selected_protected_class, denom_start, truth, pred, inverse])
                                    + truncate(self.tprd_wbar_quadrants[
                                                   selected_protected_class, denom_end, truth, 1 - pred, inverse])
                                    )

    def compute_wbar_quadrants(self, protected_class_name, bound, truth, predicted, invert=True):
        """
        Calculates lower or upper bounds for confusion matrix quadrants by protected class
        
        :param protected_class_name: str Selected protected class name
        :param bound: 'lower' or 'upper' bound
        :param truth: binary int
        :param predicted: binary int
        :param invert: boolean
        :return: float
        """
        if bound == "lower":
            f = compute_Wbar_L_hyy
        else:
            f = compute_Wbar_U_hyy
        if not invert:
            input_tuple = (
                self.primary_index,
                self.protected_class_prob[protected_class_name],
                self.quadrant_dict[(truth, predicted)],
                self.protected_class_membership[protected_class_name],
                (self.Yhat == predicted) & (self.Y == truth),
            )
        else:
            input_tuple = (
                self.primary_index,
                1 - self.protected_class_prob[protected_class_name],
                self.quadrant_dict[(truth, predicted)],
                1 - self.protected_class_membership[protected_class_name],
                (self.Yhat == predicted) & (self.Y == truth),
            )
        return f(*input_tuple)

    def compute_wbar_hemispheres(self, protected_class_name, bound, truth_or_prediction, invert=True, target='prediction'):
        """
        Calculates lower or upper bounds for confusion matrix quadrants by protected class

        :param protected_class_name: str Selected protected class name
        :param bound: 'lower' or 'upper' bound
        :param truth: binary int
        :param predicted: binary int
        :param invert: boolean
        :return: float
        """
        if bound == "lower":
            f = compute_Wbar_L_hyy
        else:
            f = compute_Wbar_U_hyy
        input_list = [self.primary_index,
                       None,
                       None,
                       None,
                       None]
        if target == 'prediction':
            input_list[2] = (self.Yhat_from_proxies==truth_or_prediction).astype(int)
            input_list[4] = (self.Yhat == truth_or_prediction).astype(int)
        else:
            input_list[2] = (self.Y_from_proxies==truth_or_prediction).astype(int)
            input_list[4] = (self.Y == truth_or_prediction).astype(int)
        if invert:
            input_list[1] = (1 - self.protected_class_prob[protected_class_name]).astype(int)
            input_list[3] = (1 - self.protected_class_membership[protected_class_name]).astype(int)
        else:
            input_list[1] = (self.protected_class_prob[protected_class_name]).astype(int)
            input_list[3] = (self.protected_class_membership[protected_class_name]).astype(int)

        return f(*input_list) * self.protected_class_membership[protected_class_name].mean()

    def compute_wbar_quadrants(self, protected_class_name, bound, truth, predicted, invert=False):
        """
        Calculates lower or upper bounds for confusion matrix quadrants by protected class

        :param protected_class_name: str Selected protected class name
        :param bound: 'lower' or 'upper' bound
        :param truth: binary int
        :param predicted: binary int
        :param invert: boolean
        :return: float
        """
        if bound == "lower":
            f = compute_Wbar_L_hyy
        else:
            f = compute_Wbar_U_hyy
        if not invert:
            input_tuple = (
                self.primary_index,
                self.protected_class_prob[protected_class_name],
                self.quadrant_dict[(truth, predicted)],
                self.protected_class_membership[protected_class_name],
                (self.Yhat == predicted) & (self.Y == truth),
            )
        else:
            input_tuple = (
                self.primary_index,
                1 - self.protected_class_prob[protected_class_name],
                self.quadrant_dict[(truth, predicted)],
                1 - self.protected_class_membership[protected_class_name],
                (self.Yhat == predicted) & (self.Y == truth),
            )
        return f(*input_tuple)

    def generate_report(self):
        for protected_class in self.protected_class_names:
            print(('Demographic Disparity, %s vs all:\t' % protected_class), self.dd_intervals_manual(protected_class))
            print(('True Positive Rate Disparity, %s vs all:\t' % protected_class),
                  self.tprd_intervals(protected_class))
            print(('True Negative Rate Disparity, %s vs all:\t' % protected_class),
                  self.tnrd_intervals(protected_class))
#todo: bounding boxes
#todo: full full report
#todo: add variances
