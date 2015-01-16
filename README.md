numcube
=======

Principles
----------
* similar interface to numpy array
* simple things should be made simple
* potentially surprising things to happen only if explicitly asked for
* immutability - no method or function changes the structure of existing cube; the values however can be changed

Cube
----
- Cube is a wrapper around numpy.ndarray object
- it has named axes (name must be a string)
- the axes have values
- Axes is a collection of axes
- cube can have axes of two types: Index and Series
- Index must have unique values
- Series does not need to have unique values
- Cube support normal operations such as multiplication, adding etc.
- Cube support aggregations
- transposition (in n-dimensional space)
- the operations work element wise

Axis matching
-------------
- axes with same names are aligned (see Axis alignment)
- axes with unique names are broadcasted
- the order of axes is the same as in the first cube followed by the unique axes from the other cube
    
Axis alignment
--------------
- if both axes are of type Index, they must contain the same values; the values however can have different order; the result axis is the first Index object
- if one of the axes is Series and the other is Index, then the values in Series must be subset of those in Index; the result axis is the Series
- if both axes are Series, error is raised; two Series objects cannot be matched

Operation with cubes
--------------------

Example:
```python
>> from numcube import Index, Cube
>> Y = Index("year", range(2014, 3))
>> Q = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
>> sales = Cube([[14, 16, 13, 20], [15, 15, 10, 19], [16, 17, 15, 21]], [Y, Q])
>> prices = Cube([[1.50, 1.52, 1.53, 1.55], [1.48, 1.47, 1.46, 1.49], [1.51, 1.57, 1.59, 1.61]], [Y, Q])
>> revenues_q = sales * prices  # quarterly revenues
>> revenues_y = revenues_q.sum("quarter")  # annual revenues
>> rel_revenues = revenues_y / revenues_y.mean("year")  # compare to overall annual average
>> revenue_coef = revenues_y / revenues.values[0]  # compare to first year revenue
```

Operations with non-Cube types
------------------------------
Cube can be used in operations with non-Cube types - scalar and numpy array. The result of such operation is Cube.
In this case however the axes are not matched and the operation is done solely with the cube.values and the other
non-Cube operand. For numpy array it means that the dimensions lengths must match or the array must be able to
be broadcasted to a compatible shape.

Example:
```python
C = Cube(np.random.rand(2, 2), Index("A", ("a1", "b1")), Index("B", ["b1", "b2"]))  # 2 x 2 cube
D = C * 10  # OK, scalar has no requirements for dimensions
E = C * np.random.rand(2, 2)  # OK, dimension lengths match
F = C * np.array([1, 2])  # OK, one dimensional array is broadcasted
G = C * np.arange(3)  # error, dimensions do not match
```

Example 1
---------
We want to calculate the likely price of the fuel mix given we are using two fuels - gas and oil.
We assume that the prices go up or 
```python
fuels = Index("fuel", ["gas", "oil"])
fuel_heat_rates = ([10, 15.5], [fuels])  # in GJ / kg

countries = Index("country", ["CZ", "HU", "PL", "SK"])
exchange_rates = ([28.1, 290, 45, 1], countries)  # in local currency / EUR

local_prices = ([...], [countries, fuels])  # in local currency / kg

eu_prices_per_GJ = local_prices / exchange_rates / fuel_heat_rates  # in EUR / GJ
```

Example 2
---------
Working with multiple scenarios.
```python
years = Index("year", range(2014, 10))  # we are planning 10 years ahead
geom_growth_exp = Cube(range[len(years)], [years])

scenarios = Index("scenario", ["low", "mid", "high"])
growth_rate = Cube([0.9, 1, 1.1], [scenarios])

growth_coef = growth_rate ** geom_growth_exp

gas_scenario_prices = ...
```
