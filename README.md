numcube package
===============

Numcube extends the functionality of numpy multidimensional arrays by adding named and annotated axes. Such
structures are called cubes. Numcube allows operations involving multiple cubes with automatic axis matching and
alignment. It allows filtering and aggregations based on the axis values. One of the goals was to provide API similar
to numpy. Internally it uses numpy arrays for the underlying array and axes as well.

Axis matching
-------------

In operations involving multiple cubes, the axes are matched and aligned. Matching means that axis names are compared
and the axes with the same names are aligned (see below), while the unique axes are broadcast (see array
broadcasting).

The cube is meant to not depend on the specific order of axes in most of the features. Nevertheless, the output of
the operation has the same order of axes as the first cube in the operation, followed by unique axes from the other
cubes respecting their order.

Axis alignment
--------------

There are basically two types of axes. There are two types of axes - Index and Series. Series has a fixed order or
values and the values do not need to be unique. Index must have unique values, which can be used for look up during
axis alignment.

If one of the axes is Series and the other is Index, then the values in Series must be subset of those in Index; the
result axis is the Series axis.

If both axes are of type Index, they must contain the same values; the values however can have different order; the
result axis is the first Index axis.

If both axes are Series, they must be equivalent - they must contain the same values in the same order. The result
axis is the first Series axis.

Filtering values
----------------

Cubes can be filtered using three distinct methods: 
1) filtering by axis values - function filter(...)
2) filtering by index along axes - function take(...)
3) filtering by logical selectors along axes - function compress(...)

Functions take(...) and compress(...) have the same semantics as in numpy package.

```python
>> from numcube import Index, Cube
>> Y = Index("year", range(2014, 3))
>> Q = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
>> sales = Cube([[14, 16, 13, 20], [15, 15, 10, 19], [16, 17, 15, 21]], [Y, Q])
>> salesH1 = sales.filter("quarter", ["Q1", "Q2"])  # by dimension attribute
>> salesH1 = sales.take("quarter", [0, 1])  # by numeric indices
>> filter_q = np.array([True, True, False, False]
>> salesH1 = sales.compress("quarter", filter_q))  # by logical vector
```

Operators
---------

Cubes support arithmetical operations between two cubes. The cube axes are matched and aligned and the operations 
are performed element-wise.

```python
>> from numcube import Index, Cube
>> year_ax = Index("year", [2014, 2015])
>> quarter_ax = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
>> sales = Cube([[14, 16, 13, 20], [15, 15, 10, 19]], [year_ax, quarter_ax])
>> prices = Cube([1.50, 1.52, 1.53, 1.55], [year_ax])
>> revenues_q = sales * prices  # quarterly revenues
```

Cube can also be in operation with a scalar value, which is treated as dimensionless Cube. 

```python
>> Y = Index("year", range(2014, 2))
>> Q = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
>> prices = Cube([[1.50, 1.52, 1.53, 1.55], [1.48, 1.47, 1.46, 1.49], [Y, Q])
>> discount = 0.5
>> discounted_prices = sales * discount  # operation with scalar
```

Aggregations
------------

Cube values can be aggregated along axes using aggregation functions sum, mean, min, max, etc.

```python
>> total_revenues = revenues.sum()
>> annual_revenues = revenues.sum(keep="year")
>> avg_annual_revenues = revenues.mean("quarter")
```

Other
-----

- the interface of all classes is designed to support immutability
- Cube supports numerical functions such as sin, cos, log, exp etc.
- transposition (in n-dimensional space) changes the order of axes


Example 1
---------

We want to calculate the likely price of the fuel mix given we are using two fuels - gas and oil.

```python
>> fuels = Index("fuel", ["gas", "oil"])
>> fuel_heat_rates = ([10, 15.5], fuels)  # in GJ / kg

>> countries = Index("country", ["CZ", "HU", "PL", "SK"])
>> exchange_rates = Cube([28.1, 290, 45, 1], countries)  # in local currency / EUR

>> local_prices = ([...], [countries, fuels])  # in local currency / kg

>> eu_prices_per_GJ = local_prices / exchange_rates / fuel_heat_rates  # in EUR / GJ
```

Example 2
---------

Working with multiple scenarios.

```python
>> years = Index("year", range(2014, 2020))
>> geom_growth_exp = Cube(range[len(years)], years)

>> scenarios = Index("scenario", ["low", "mid", "high"])
>> growth_rate = Cube([0.9, 1, 1.1], scenarios)

>> growth_coef = growth_rate ** geom_growth_exp

>> gas_scenario_prices = ...
```

Example 3
---------

```python
>> revenues_y = revenues_q.sum("quarter")  # annual revenues
>> rel_revenues = revenues_y / revenues_y.mean("year")  # compare to overall annual average
>> revenue_coef = revenues_y / revenues.values[0]  # compare to first year revenue
```
