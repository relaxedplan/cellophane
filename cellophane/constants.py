UPPER_BOUND = 'upper'
"""Defines the upper bound of a partial identification set"""
LOWER_BOUND = 'lower'
"""Defines the lower bound of a partial identification set"""
EXPECTATION = 'expectation'
"""Defines the expected value of a partial identification set"""

IN_PROTECTED_CLASS = True
"""True when a row is in the specified protected class"""
NOT_IN_PROTECTED_CLASS = False
"""True when a row is not in the specified protected class"""

GROUND_TRUTH = 'truth'
"""Represents the actual observed value of the target variable of the modelling project being audited, for a member of the primary set"""
MODEL_PREDICTION = 'prediction'
"""Represents the model's prediction of the row, for a member of the primary set"""
PROTECTED_CLASS = 'protected_class'
"""A specific category of protected class within the auxiliary set, eg "Black/African American" within a race column"""
GROUND_TRUTH_FROM_PROXIES = 'truth_from_proxies'
"""A specific category of protected class within the auxiliary set, eg "Black/African American" within a race column"""
MODEL_PREDICTION_FROM_PROXIES = 'prediction_from_proxies'
"""Represents the prediction of a model built only from proxies to emulate the model being audited"""
PROTECTED_CLASS_FROM_PROXIES = 'protected_class_from_proxies'
"""Represents the prediction of a model built only from proxies to estimate protected class membership"""

POSITIVE = 1
"""A positive prediction or observed event"""
NEGATIVE = 0
"""A negative prediction or observed event"""

OTHERS = 'others'
"""When comparing bias metrics for one protected class to all others (eg men vs all), this stands in for all others"""