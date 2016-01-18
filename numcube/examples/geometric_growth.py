"""This example illustrates how to create a geometrical growth series and how to analyze such a series it.
"""

from numcube import Cube, Axis

if __name__ == "__main__":
    years = Axis("year", range(2014, 2020))
    geom_growth_exponential = Cube(range(len(years)), years)  # cube contains values 0, 1, 2, etc.

    # there are three scenarios with different growth coefficients
    scenarios = Axis("scenario", ["low", "mid", "high"])
    growth_rate = Cube([0.9, 1, 1.1], scenarios)

    # now we generate a cube with coefficients for all years and scenarios
    growth_coefficient = growth_rate ** geom_growth_exponential

    # now we generate a cube with prices
    initial_price = 100.0
    price = initial_price * growth_coefficient
    # this we we can generate one or more with geometric growth

    # ...and vice versa, if we want to analyze data as geometric growth
    # we can calculate base indices, i.e. index relative to the first value in the series
    base_indices = price / price.take("year", 0)

    # or we can calculate running indices, i.e. index relative to the previous value
    # running_indices = ???



