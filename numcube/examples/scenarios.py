"""Working with multiple scenarios.

We want to calculate the likely price of the fuel mix given we are using two fuels - gas and oil.
"""

from numcube import Cube, Axis

if __name__ == "__main__":
    fuels = Axis("fuel", ["gas", "oil"])
    fuel_heat_rates = ([10, 15.5], fuels)  # in GJ / kg

    countries = Axis("country", ["CZ", "HU", "PL", "SK"])
    exchange_rates = Cube([28.1, 290, 45, 1], countries)  # in local currency / EUR

    local_prices = ([...], [countries, fuels])  # in local currency / kg

    eu_prices_per_GJ = local_prices / exchange_rates / fuel_heat_rates  # in EUR / GJ

