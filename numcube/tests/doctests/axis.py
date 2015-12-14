"""
Axis class
==========
Axis is a named ordered collection of values.

For these doctests to run we are going to import numcube.Axis and numpy.
>>> from numcube import Axis
>>> import numpy as np

Creation
--------
To create an Axis object, you have to supply it with name and values. Name must be a string,
values must be convertible to one-dimensional numpy array. The values should be of the same type,
otherwise they are converted to the most flexible type.

- initialized by explicit values:
(note: dtype=object is not necessary, it is here to pass the doctests below in both Python 2 and Python 3)
>>> months = Axis("month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
>>> months
Axis('month', ['jan' 'feb' 'mar' 'apr' 'may' 'jun' 'jul' 'aug' 'sep' 'oct' 'nov' 'dec'])

- initialized from a range:
>>> years = Axis("year", range(2010, 2020))
>>> years
Axis('year', [2010 2011 2012 2013 2014 2015 2016 2017 2018 2019])

Properties
----------

- 'name' returns a string
>>> months.name
'month'

- 'values' returns a numpy array
note: this is commented out since this test is not portable between Python 2 and Python 3
#>>> months.values  # doctest: +NORMALIZE_WHITESPACE
array(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
       'oct', 'nov', 'dec'])

- str(years) converts the axis to its string representation
>>> str(years)
"Axis('year', [2010 2011 2012 2013 2014 2015 2016 2017 2018 2019])"

- len(axis) returns the number of values
>>> len(months)
12

Slicing, indexing, filtering
----------------------------
The returned object is also Axis with the same name and subset of values.

>>> months[0:4]
Axis('month', ['jan' 'feb' 'mar' 'apr'])

>>> months[-1]
Axis('month', ['dec'])

>>> months[::2]
Axis('month', ['jan' 'mar' 'may' 'jul' 'sep' 'nov'])

When accessing values by their indices, you have to provide double square brackets!

>>> months[[0, 2, 4]]
Axis('month', ['jan' 'mar' 'may'])

The values can be repeated using repeated indices.

>>> months[[1, 2, 1, 2]]
Axis('month', ['feb' 'mar' 'feb' 'mar'])

To filter axis by index, you can also use method take(), which is similar to numpy.take().

>>> months.take([0, 2, 4])
Axis('month', ['jan' 'mar' 'may'])

You can filter the axis by using logical values in a numpy array.

>>> years[np.array([True, False, True, False, True, False, True, False, True, False])]
Axis('year', [2010 2012 2014 2016 2018])

The previous example was not very useful by itself. But numpy array of logical values is
the result of logical expression with axis values. Now this is much more useful.

>>> years[years.values % 2 == 0]  # even years
Axis('year', [2010 2012 2014 2016 2018])

>>> years[(years.values >= 2013) & (years.values <= 2016)]  # note the single '&', do not confuse with C/C++ '&&' style
Axis('year', [2013 2014 2015 2016])

To filter axis by logical values, you can also use method compress(), which is similar to numpy.compress().
In this case you do not need to convert logical values to numpy array.

>>> years.compress([True, False, True, False, True, False, True, False, True, False])
Axis('year', [2010 2012 2014 2016 2018])

Renaming
--------
We can rename the axis. Renaming returns a new axis (do not forget to assign it to a new variable!),
the original axis remains unchanged.

>>> m = months.rename("M")
>>> m
Axis('M', ['jan' 'feb' 'mar' 'apr' 'may' 'jun' 'jul' 'aug' 'sep' 'oct' 'nov' 'dec'])

This is the original axis, still with the old name:

>>> months
Axis('month', ['jan' 'feb' 'mar' 'apr' 'may' 'jun' 'jul' 'aug' 'sep' 'oct' 'nov' 'dec'])

Sorting
-------
Sorting is one of the places where numcube API and numpy API differs. Numcube sorting returns a copy
of the axis which is analogy to numpy.sort(array) function. On the other hand array.sort() sorts the
array in-place. The reason is that numcube aims to support immutability as much as possible.

>>> persons = Axis("person", ["Steve", "John", "Alex", "Peter", "Linda"])
>>> sorted_persons = persons.sort()
>>> sorted_persons
Axis('person', ['Alex' 'John' 'Linda' 'Peter' 'Steve'])

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
"""
