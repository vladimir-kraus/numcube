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
Values less than 1st decile and greater than 9th decile are treated as outliers.

>>> decile_1 = temperature.reduce(func=lambda a: np.percentile(a, 10.0))
>>> decile_9 = temperature.reduce(func=lambda a: np.percentile(a, 90.0))
>>> outlier_replacement = lambda x: x if decile_1 <= x <= decile_9 else np.nan
>>> temperature_adjusted = temperature.apply(func=outlier_replacement)

# >>> print(temperature_adjusted)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
"""