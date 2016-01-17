"""Working with multiple scenarios."""

from numcube import Cube, Axis

if __name__ == "__main__":
    years = Axis("year", range(2014, 2020))
    geom_growth_exp = Cube(range(len(years)), years)

    scenarios = Axis("scenario", ["low", "mid", "high"])
    growth_rate = Cube([0.9, 1, 1.1], scenarios)

    growth_coef = growth_rate ** geom_growth_exp

    # gas_scenario_prices = ...
