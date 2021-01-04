from constants import *


def positive_or_negative_hemisphere_reading(hemisphere_readings, protected_class, result=POSITIVE, variable=MODEL_PREDICTION, within_protected_class=True):
    """
    :param hemisphere_readings: Dictionary containing lower/expected/upper bounds for all 4 hemispheres
    :type hemisphere_readings: dict
    :param protected_class: Name of protected class
    :type protected_class: str, int
    :param result: Numeric result to look up bounds for. Should be 0 or 1. If zero, looks up negative readings, if one, looks up positive readings
    :type result: int
    :param variable: Target to look up bounds for. This can be 'prediction' (MODEL_PREDICTION) or 'truth' (GROUND_TRUTH)
    :type variable: str
    :param within_protected_class: If True, compute bounds on data belonging to the protected class. If false, compute bound on data not belonging to the protected class.
    :type within_protected_class: bool
    :return: tuple containing lower bound, expected value, and upper bound
    :rtype: tuple
    """
    return (
        hemisphere_readings[(protected_class, LOWER_BOUND, result, within_protected_class, variable)],
        hemisphere_readings[(protected_class, EXPECTATION, result, within_protected_class, variable)],
        hemisphere_readings[(protected_class, UPPER_BOUND, result, within_protected_class, variable)]
    )


def confusion_matrix_quadrant_reading(quadrant_readings, protected_class, truth=POSITIVE, prediction=POSITIVE, within_protected_class=True):
    """

    :param quadrant_readings: Dictionary containing lower/expected/upper bounds for all 4 confusion matrix quadrants
    :type quadrant_readings: dict
    :param protected_class: Name of protected class
    :type protected_class: str, int
    :param truth: (0,1) represents the ground truth of the data
    :type truth: int
    :param prediction: (0,1) represents the model's prediction on the data
    :type prediction: int
    :param within_protected_class: If True, compute bounds on data belonging to the protected class. If false, compute bound on data not belonging to the protected class.
    :type within_protected_class: bool
    :return: tuple containing lower bound, expected value, and upper bound
    :rtype: tuple
    """
    return (
        quadrant_readings[(protected_class, LOWER_BOUND, truth, prediction, within_protected_class)],
        quadrant_readings[(protected_class, EXPECTATION, truth, prediction, within_protected_class)],
        quadrant_readings[(protected_class, UPPER_BOUND, truth, prediction, within_protected_class)]
    )


def confusion_matrix_quadrant_disparity(quadrant_readings, protected_class, truth=POSITIVE, prediction=POSITIVE, comparison_class=OTHERS):
    """

    :param quadrant_readings: Dictionary containing lower/expected/upper bounds for all 4 confusion matrix quadrants
    :type quadrant_readings: dict
    :param protected_class: Name of protected class
    :type protected_class: str, int
    :param truth: (0,1) represents the ground truth of the data
    :type truth: int
    :param prediction: (0,1) represents the model's prediction on the data
    :type prediction: int
    :param comparison_class: Name of comparison class. Can be the name of a protected class or OTHER ('others')
    :type comparison_class: str
    :return: tuple containing lower bound, expected value, and upper bound
    :rtype: tuple
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

    :param hemisphere_readings:
    :type hemisphere_readings:
    :param protected_class:
    :type protected_class:
    :param comparison_class:
    :type comparison_class:
    :param variable:
    :type variable:
    :param value:
    :type value:
    :return:
    :rtype:
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
    
    :param tprd_tpnd_components: 
    :type tprd_tpnd_components: 
    :param protected_class: 
    :type protected_class: 
    :param comparison_class: 
    :type comparison_class: 
    :return: 
    :rtype: 
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
    
    :param tprd_tpnd_components: 
    :type tprd_tpnd_components: 
    :param protected_class: 
    :type protected_class: 
    :param comparison_class: 
    :type comparison_class: 
    :return: 
    :rtype: 
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

    :param ppvd_npvd_components:
    :type ppvd_npvd_components:
    :param protected_class:
    :type protected_class:
    :param comparison_class:
    :type comparison_class:
    :return:
    :rtype:
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

    :param ppvd_npvd_components:
    :type ppvd_npvd_components:
    :param protected_class:
    :type protected_class:
    :param comparison_class:
    :type comparison_class:
    :return:
    :rtype:
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

    :param tprd_tnrd_components:
    :type tprd_tnrd_components:
    :param protected_class:
    :type protected_class:
    :return:
    :rtype:
    """
    return (
        tprd_tnrd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        tprd_tnrd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        tprd_tnrd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
    )

def tnr_reading(ppvd_npvd_components, protected_class):
    """

    :param ppvd_npvd_components:
    :type ppvd_npvd_components:
    :param protected_class:
    :type protected_class:
    :return:
    :rtype:
    """
    return (
        ppvd_npvd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
    )

def ppv_reading(ppvd_npvd_components, protected_class):
    """

    :param ppvd_npvd_components:
    :type ppvd_npvd_components:
    :param protected_class:
    :type protected_class:
    :return:
    :rtype:
    """
    return (
        ppvd_npvd_components[protected_class, LOWER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, EXPECTATION, POSITIVE, POSITIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, UPPER_BOUND, POSITIVE, POSITIVE, IN_PROTECTED_CLASS]
    )

def npv_reading(ppvd_npvd_components, protected_class):
    """

    :param ppvd_npvd_components:
    :type ppvd_npvd_components:
    :param protected_class:
    :type protected_class:
    :return:
    :rtype:
    """
    return (
        ppvd_npvd_components[protected_class, LOWER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, EXPECTATION, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS],
        ppvd_npvd_components[protected_class, UPPER_BOUND, NEGATIVE, NEGATIVE, IN_PROTECTED_CLASS]
    )