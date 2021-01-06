from constants import *


def positive_or_negative_hemisphere_reading(hemisphere_readings, protected_class, result=POSITIVE,
                                            variable=MODEL_PREDICTION, within_protected_class=True):
    """
    For a given hemisphere and protected class, return the lower bound, expected value, and upper bound.
    Hemispheres:
        - Hemispheres can be one of the following:
            - The probability of a positive observed event (ie positive actual)
            - The probability of a negative observed event (ie negative actual)
            - The probability of a positive prediction according to the original model (ie positive prediction)
            - The probability of a negative prediction according to the original model (ie negative actual)
    Args:
        hemisphere_readings (dict): Dictionary containing lower/expected/upper bounds for all 4 hemispheres. Passed
        protected_class (str): Name of protected class
        result (int): Numeric result to look up bounds for. Should be 0 or 1. If zero, looks up negative readings, if
        variable (int): Target variable to look up bounds for. This can be 'prediction' (MODEL_PREDICTION) or 'truth' \
(GROUND_TRUTH)
        within_protected_class (bool): If True, compute bounds on data belonging to the protected class.\
If false, compute bound on data not belonging to the protected class.
    Returns:
        tuple containing lower bound, expected value, and upper bound
    """
    return (
        hemisphere_readings[(protected_class, LOWER_BOUND, result, within_protected_class, variable)],
        hemisphere_readings[(protected_class, EXPECTATION, result, within_protected_class, variable)],
        hemisphere_readings[(protected_class, UPPER_BOUND, result, within_protected_class, variable)]
    )


def confusion_matrix_quadrant_reading(quadrant_readings, protected_class, truth=POSITIVE, prediction=POSITIVE, within_protected_class=True):
    """
    For a confusion matrix quadrant and protected class, return the lower bound, expected value, and upper bound.
    Args:
        quadrant_readings (dict): Dictionary containing lower/expected/upper bounds for all 4 hemispheres. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class
        result (int): Numeric result to look up bounds for. Should be 0 or 1. If zero, looks up negative readings, if one, looks up positive readings
        variable (int): Target variable to look up bounds for. This can be 'prediction' (MODEL_PREDICTION) or 'truth' (GROUND_TRUTH)
        within_protected_class (bool): If True, compute bounds on data belonging to the protected class. If false, compute bound on data not belonging to the protected class.
    Returns:
        tuple containing lower bound, expected value, and upper bound
    """
    return (
        quadrant_readings[(protected_class, LOWER_BOUND, truth, prediction, within_protected_class)],
        quadrant_readings[(protected_class, EXPECTATION, truth, prediction, within_protected_class)],
        quadrant_readings[(protected_class, UPPER_BOUND, truth, prediction, within_protected_class)]
    )


def confusion_matrix_quadrant_disparity(quadrant_readings, protected_class, truth=POSITIVE, prediction=POSITIVE, comparison_class=OTHERS):
    """
    For a given confusion matrix quadrant, a protected class, and a comparison class, return the set of potential
    disparities in probability.
    Args:
        quadrant_readings (dict): Dictionary containing lower/expected/upper bounds for all quadrants for all protected classes. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class
        result (int): Numeric result to look up bounds for. Should be 0 or 1. If zero, looks up negative readings, if one, looks up positive readings
        variable (int): Target variable to look up bounds for. This can be 'prediction' (MODEL_PREDICTION) or 'truth' (GROUND_TRUTH)
        comparison_class (str):
            if OTHERS, compare the protected class to all other protected classes.
            Otherwise, compare the protected class only to the comparison class.

    Returns:
        tuple containing lower bound, expected value, and upper bound
    """
    protected_class_result = confusion_matrix_quadrant_reading(quadrant_readings, protected_class, truth, prediction, IN_PROTECTED_CLASS)
    if comparison_class == OTHERS:
        comparison_class_result = confusion_matrix_quadrant_reading(quadrant_readings, protected_class, truth, prediction, NOT_IN_PROTECTED_CLASS)
    else:
        comparison_class_result = confusion_matrix_quadrant_reading(quadrant_readings, comparison_class, truth, prediction, IN_PROTECTED_CLASS)
    lower_pc, expected_pc, upper_pc = protected_class_result
    lower_cc, expected_cc, upper_cc = comparison_class_result
    return (lower_pc - upper_cc,
            expected_pc - expected_cc,
            upper_pc - lower_pc)


def positive_or_negative_hemisphere_disparity(hemisphere_readings, protected_class, comparison_class=OTHERS, variable=MODEL_PREDICTION, value=POSITIVE):
    """
    For a given hemisphere, a protected class, and a comparison class, return the set of potential disparities in probability of being in the hemisphere between the protected class and comparison class.
    Args:
        hemisphere_readings (dict): Dictionary containing lower/expected/upper bounds for all 4 hemispheres. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class
        result (int): Numeric result to look up bounds for. Should be 0 or 1. If zero, looks up negative readings, if one, looks up positive readings
        variable (int): Target variable to look up bounds for. This can be 'prediction' (MODEL_PREDICTION) or 'truth' (GROUND_TRUTH)
        comparison_class (str):
            if OTHERS, compare the protected class to all other protected classes.
            Otherwise, compare the protected class only to the comparison class.

    Returns:
        tuple containing lower bound, expected value, and upper bound
    """
    protected_class_result = positive_or_negative_hemisphere_reading(hemisphere_readings, protected_class, value, variable, IN_PROTECTED_CLASS)
    if comparison_class == OTHERS:
        comparison_class_result = positive_or_negative_hemisphere_reading(hemisphere_readings, protected_class, value, variable, NOT_IN_PROTECTED_CLASS)
    else:
        comparison_class_result = positive_or_negative_hemisphere_reading(hemisphere_readings, comparison_class, value, variable, IN_PROTECTED_CLASS)
    lower_pc, expected_pc, upper_pc = protected_class_result
    lower_cc, expected_cc, upper_cc = comparison_class_result
    return (lower_pc - upper_cc,
            expected_pc - expected_cc,
            upper_pc - lower_pc)


def tpr_disparity(tprd_tpnd_components, protected_class, comparison_class=OTHERS):
    """
    For a given hemisphere, a protected class, and a comparison class, return the set of potential true positive rate disparities
    Args:
        tprd_tpnd_components (dict): Dictionary containing lower/expected/upper bounds of true positives rates and true negative rates for all protected classes. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class
        comparison_class (str):
            if OTHERS, compare the protected class to all other protected classes.
            Otherwise, compare the protected class only to the comparison class.

    Returns:
        tuple containing lower bound, expected value, and upper bound of true positive rate disparity
    """
    if comparison_class == OTHERS:
        return (
            tprd_tpnd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, NOT_IN_PROTECTED_CLASS],
            tprd_tpnd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, NOT_IN_PROTECTED_CLASS],
            tprd_tpnd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, NOT_IN_PROTECTED_CLASS],
        )
    else:
        return (
            tprd_tpnd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[comparison_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
            tprd_tpnd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[comparison_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
            tprd_tpnd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[comparison_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        )


def tnr_disparity(tprd_tpnd_components, protected_class, comparison_class=OTHERS):
    """
    For a given hemisphere, a protected class, and a comparison class, return the set of potential true negative rate disparities
    Args:
        tprd_tpnd_components (dict): Dictionary containing lower/expected/upper bounds of true positives rates and true negative rates for all protected classes. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class
        comparison_class (str):
            if OTHERS, compare the protected class to all other protected classes.
            Otherwise, compare the protected class only to the comparison class.

    Returns:
        tuple containing lower bound, expected value, and upper bound of true negative rate disparity
    """
    if comparison_class == OTHERS:
        return (
            tprd_tpnd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, NOT_IN_PROTECTED_CLASS],
            tprd_tpnd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, NOT_IN_PROTECTED_CLASS],
            tprd_tpnd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, NOT_IN_PROTECTED_CLASS],
        )
    else:
        return (
            tprd_tpnd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[comparison_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
            tprd_tpnd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[comparison_class, EXPECTATION, NEGATIVE, NEGATIVE, NOT_IN_PROTECTED_CLASS],
            tprd_tpnd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - tprd_tpnd_components[comparison_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        )


def ppv_disparity(ppvd_npvd_components, protected_class, comparison_class=OTHERS):
    """
    For a given hemisphere, a protected class, and a comparison class, return the set of potential true negative rate disparities
    Args:
        tprd_tpnd_components (dict): Dictionary containing lower/expected/upper bounds of true positives rates and true negative rates for all protected classes. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class
        comparison_class (str):
            if OTHERS, compare the protected class to all other protected classes.
            Otherwise, compare the protected class only to the comparison class.

    Returns:
        tuple containing lower bound, expected value, and upper bound of true negative rate disparity
    """
    if comparison_class == OTHERS:

        return (
            ppvd_npvd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, NOT_IN_PROTECTED_CLASS],
            ppvd_npvd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, NOT_IN_PROTECTED_CLASS],
            ppvd_npvd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, NOT_IN_PROTECTED_CLASS],
        )
    else:
        return (
            ppvd_npvd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[comparison_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
            ppvd_npvd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[comparison_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
            ppvd_npvd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[comparison_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        )


def npv_disparity(ppvd_npvd_components, protected_class, comparison_class=OTHERS):
    """
    For a given a protected class, and a comparison class, return the set of potential negative predictive value disparities
    Args:
        tprd_tpnd_components (dict): Dictionary containing lower/expected/upper bounds of PPV and NPV for all protected classes. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class
        comparison_class (str):
            if OTHERS, compare the protected class to all other protected classes.
            Otherwise, compare the protected class only to the comparison class.

    Returns:
        tuple containing lower bound, expected value, and upper bound of true negative rate disparity
    """
    if comparison_class == OTHERS:
        return (
            ppvd_npvd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, NOT_IN_PROTECTED_CLASS],
            ppvd_npvd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, NOT_IN_PROTECTED_CLASS],
            ppvd_npvd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, NOT_IN_PROTECTED_CLASS],
        )
    else:
        return (
            ppvd_npvd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[comparison_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
            ppvd_npvd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[comparison_class, EXPECTATION, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
            ppvd_npvd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
            - ppvd_npvd_components[comparison_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        )

def tpr_reading(tprd_tnrd_components, protected_class):
    """
    For a given protected class, return the set of potential true positive rates.
    Args:
        tprd_tpnd_components (dict): Dictionary containing lower/expected/upper bounds of true positives rates and true negative rates for all protected classes. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class

    Returns:
        tuple containing lower bound, expected value, and upper bound of true positive rates
    """
    return (
        tprd_tnrd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        tprd_tnrd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        tprd_tnrd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
    )

def tnr_reading(ppvd_npvd_components, protected_class):
    """
    For a given protected class, return the set of potential true negative rates.
    Args:
        tprd_tpnd_components (dict): Dictionary containing lower/expected/upper bounds of true positives rates and true negative rates for all protected classes. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class

    Returns:
        tuple containing lower bound, expected value, and upper bound of true negative rates
    """
    return (
        ppvd_npvd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
    )

def ppv_reading(ppvd_npvd_components, protected_class):
    """
    For a given protected class, return the set of potential positive predictive values.
    Args:
        tprd_tpnd_components (dict): Dictionary containing lower/expected/upper bounds of PPV and NPV for all protected classes. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class

    Returns:
        tuple containing lower bound, expected value, and upper bound of positive predictive value
    """
    return (
        ppvd_npvd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
    )

def npv_reading(ppvd_npvd_components, protected_class):
    """
    For a given protected class, return the lower bound, expected value, and upper bound, of the negative predictive value
    Args:
        tprd_tpnd_components (dict): Dictionary containing lower/expected/upper bounds of PPV and NPV for all protected classes. Passed in automatically if called from a PartialIdentification object.
        protected_class (str): Name of protected class

    Returns:
        tuple containing lower bound, expected value, and upper bound of negative predictive value
    """
    return (
        ppvd_npvd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
    )