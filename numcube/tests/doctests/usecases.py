"""
Usecases
========

>>> import numpy as np
>>> from numcube import Index, Cube

Create cube with measured temperatures.

>>> hour_axis = Index("hour", range(1, 25))
>>> weekday_axis = Index("weekday", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
>>> values = np.random.rand(7, 24)
>>> temperature = Cube(values, [weekday_axis, hour_axis])

Replace outliers with NaNs. Use lambda to pass extra arguments to aggregation function.
Values less than 1st decile and greater than 9th decile are treated as outliers.

>>> decile_1 = temperature.reduce(func=lambda a: np.percentile(a, 10.0))
>>> decile_9 = temperature.reduce(func=lambda a: np.percentile(a, 90.0))
>>> outlier_replacement = lambda x: x if decile_1 <= x <= decile_9 else np.nan
>>> temperature_adjusted = temperature.apply(func=outlier_replacement)

Now calculate the average with excluding the NaNs.

>>> avg_temperature = np.nanmean(temperature_adjusted.values)
>>> avg_temperature = temperature.masked(lambda c: c < decile_1 or c > decile_9).mean()
>>> avg_temperature = temperature_adjusted.masked(np.isnan).mean()

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
"""