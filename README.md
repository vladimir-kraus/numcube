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
>> # filter by dimension attribute
>> salesH1 = sales.filter("quarter", ["Q1", "Q2"])  
>> # filter by numeric indices
>> salesH1 = sales.take("quarter", [0, 1]) 
>> # filter by logical vector
>> filter_q = np.array([True, True, False, False]
>> salesH1 = sales.compress("quarter", filter_q))  
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

Cube values can be aggregated along axes using aggregation functions sum, mean, min, max, etc. All aggregation
functions allow defining which axes are going to be aggregated, which are to be kept or values on which are going
to be grouped.

```python
>> total_revenues = revenues.sum()
>> average_annual_revenues = revenues.mean("quarter")
>> total_annual_revenues = revenues.sum(keep="year")
```

Aggregations can be also used to group values along an axis with non-unique values., for example:
```python
>> subject = Axis('subject', [math', 'biology', 'math', 'physics', 'math', 'biology', 'math', 'physics'])
>> score = Cube([65, 80, 95, 52, 35, 50, 89, 95], subject)
>> score_by_subject = score.mean(group='subject')
```

General aggregation function is reduce(), which it is possible to provide with a user defined aggregation function.
```python
>> decile_9th = score.reduce(func=lambda x: np.percentile(x, 90.0))
```

Logical aggregation functions all() and any() can be used to test whether all or any logical value in a cube is True.
```python
>> c = Cube(...)
>> d = Cube(...)
>> if (c > d).all():  # to test if all values in c are greater than respective values in d
```
Note that comparison operators use the same axis matching, alignment adn broadcasting as normal arithmetic operators.

Other
-----

- the interface of all classes is designed to support immutability
- Cube supports numerical functions such as sin, cos, log, exp etc.
- transposition (in n-dimensional space) changes the order of axes