"""In this example we are going to present several methods calculate the average value of certain measurements
which are corrupted with errors in measurements which introduce outlier values.

We are going to use two approaches:
1) replacing outliers with NaN (not-a-number) values and calculating the average value using nanmean() function.
2) using masked cubes which define which values in a cube shall be ignored when calculating the mean
"""

import numpy as np
from numcube import Index, Cube
import timeit

if __name__ == "__main__":
    # The cube will contain measurements taken each second during a period of 24 hours.
    hour_axis = Index("hour", range(1, 25))
    minute_axis = Index("minute", range(1, 61))
    second_axis = Index("second", range(1, 61))
    shape = (len(hour_axis), len(minute_axis), len(second_axis))

    # correct values which would ideally be measured
    values = np.random.rand(*shape)
    print("Theoretical sample:")
    print("avg={0} sigma={1}".format(np.mean(values), np.std(values)))

    # Measurements of correct values corrupted with outliers, e.g. every 20th value will be multiplied by 100.
    corrupted_values = values + np.random.binomial(n=1, p=0.01, size=shape) * 100
    print("\nCorrupted sample:")
    print("avg={0} sigma={1}".format(np.mean(corrupted_values), np.std(corrupted_values)))

    # create a Cube instance
    measurements = Cube(corrupted_values, [hour_axis, minute_axis, second_axis])

    # Now we have to decide how we are going to identify the outliers. There are several possible solutions.
    # For example, values less than 1st decile and greater than 9th decile are treated as outliers.
    lower = measurements.reduce(func=lambda a: np.percentile(a, 10.0))
    upper = measurements.reduce(func=lambda a: np.percentile(a, 90.0))

    # Alternatively we can calculate average value and standard deviation (sigma) with inclusion of all outliers,
    # and then exclude all values which are more distant from the average value than 3 sigmas.
    raw_avg = measurements.mean()
    raw_sigma = measurements.std()
    lower = raw_avg - 3 * raw_sigma
    upper = raw_avg + 3 * raw_sigma

    # Note: there are many more methods to identify outliers, however the goal of this example
    # is not to design the best method to exclude outliers. We will use the +/- 3 sigma method.

    # Replace outliers with NaNs. Use lambda to pass extra arguments to aggregation function.
    adjusted_measurements = measurements.apply(func=lambda x: x if lower <= x <= upper else np.nan)

    print("\nAfter exclusion of outliers (various methods):")

    # Now calculate the average with excluding the NaNs.
    avg = np.nanmean(adjusted_measurements.values)
    sigma = np.nanstd(adjusted_measurements.values)
    print("avg={0} sigma={1}".format(avg, sigma))

    # TODO: why it does not work?
    # avg_temperature = np.nanmean(temperature_adjusted)
    # avg = np.nanmean(adjusted_measurements)
    # sigma = np.nanstd(adjusted_measurements)
    # print("avg={0} sigma={1}".format(avg, sigma))

    avg = adjusted_measurements.nanmean()
    sigma = adjusted_measurements.nanstd()
    print("avg={0} sigma={1}".format(avg, sigma))

    # The following approach using reduce() function can be used for functions which
    # are defined in numpy package but they have not been added to Cube interface.
    avg = adjusted_measurements.reduce(np.nanmean)
    sigma = adjusted_measurements.reduce(np.nanstd)
    print("avg={0} sigma={1}".format(avg, sigma))

    # Here we are going to use much more powerful mechanism - masking cubes.
    # The following line masks all NaN values and they subsequently will be
    # ignored when calculating the mean value using normal mean() function.
    masked_measurements = adjusted_measurements.masked(np.isnan)
    avg = masked_measurements.mean()
    sigma = masked_measurements.std()
    print("avg={0} sigma={1}".format(avg, sigma))

    # However we can also use a function to define which values are going to be masked and which are not.
    # The following line will mask all values less than 1st decile and greater than 9th decile.
    # In other words, with this approach we can skip the replacement of outliers with NaNs.
    masked_measurements = measurements.masked(lambda c: c < lower or c > upper)
    avg = masked_measurements.mean()
    sigma = masked_measurements.std()
    print("avg={0} sigma={1}".format(avg, sigma))

    # The following line does exactly the same operation as the previous but it uses numpy vectorized expression,
    # therefore it is significantly faster than the previous when applied on large arrays.
    # Here we finally came to the potentially best solution to achieve what we asked for.
    masked_measurements = measurements.masked(lambda c: (c < lower) | (c > upper))  # Note the brackets!!!
    avg = masked_measurements.mean()
    sigma = masked_measurements.std()
    print("avg={0} sigma={1}".format(avg, sigma))

    # Here we are going to document the speed difference when masking with vectorized and non-vectorized
    # functions. As you can see, the vectorized operation is significantly (around 100x) faster.
    # Therefore use non-vectorized only when absolutely necessary!
    print("\nComparison of speed of masking operations:")
    repeat = 1
    setup = "from __main__ import measurements, lower, upper"
    func = "measurements.masked(lambda c: (c < lower) | (c > upper))"  # Note the brackets!!!
    timer = timeit.Timer(func, setup)
    t_vectorized = timer.timeit(repeat)
    print("vectorized mask: t={0} secs".format(t_vectorized))
    func = "measurements.masked(lambda c: c < lower or c > upper)"
    timer = timeit.Timer(func, setup)
    t_non_vectorized = timer.timeit(repeat)
    print("non-vectorized mask: t={0} secs".format(t_non_vectorized))
    print("speed gain={0}x".format(int(t_non_vectorized / t_vectorized)))

