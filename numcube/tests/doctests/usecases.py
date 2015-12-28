"""
Usecases
========

>>> import numpy as np
>>> from numcube import Index, Cube

Create cube with measured temperatures.

>>> hour_axis = Index("hour", range(1, 25))
>>> weekday_axis = Index("weekday", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
>>> values = np.arange(7 * 24).reshape(7, 24)
>>> temperature = Cube(values, [weekday_axis, hour_axis])

# print(temperature)

Replace outliers with NaNs. Use lambda to pass extra arguments to aggregation function.

>>> decile_1 = temperature.aggregate(func=lambda a: np.percentile(a, 10.0))
>>> decile_9 = temperature.aggregate(func=lambda a: np.percentile(a, 90.0))

Use lambda and np.vectorize to create a function which is applied to each cube element.

>>> outlier_replacement = np.vectorize(lambda x: x if decile_1 <= x <= decile_9 else np.nan)
>>> temperature_adjusted = temperature.apply(func=outlier_replacement)

# >>> print(temperature_adjusted)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
"""