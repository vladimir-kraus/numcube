"""
Index class
-----------

Index is a named, ordered and indexed collection of unique values


>>> import numpy as np
>>> from numcube import Index, Cube


Index object
------------

>>> yr = Index("year", range(2010, 2020))

>>> yr
Index('year', [2010 2011 2012 2013 2014 2015 2016 2017 2018 2019])

>>> mo = Index("month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
>>> mo.name
'month'

>>> mo.values  # doctest: +NORMALIZE_WHITESPACE
array(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
       'oct', 'nov', 'dec'],
      dtype='<U3')


Slicing
-------

>>> mo[0:4]
Index('month', ['jan' 'feb' 'mar' 'apr'])

>>> mo[-1]
Index('month', ['dec'])

>>> mo[::2]
Index('month', ['jan' 'mar' 'may' 'jul' 'sep' 'nov'])


Indexing
--------

>>> mo[[0, 2, 4]]  # note the double square brackets!
Index('month', ['jan' 'mar' 'may'])

>>> yr[yr.values % 2 == 0]  # even years
Index('year', [2010 2012 2014 2016 2018])

>>> yr[(yr.values >= 2013) & (yr.values <= 2016)]
Index('year', [2013 2014 2015 2016])


Finding index of a values
-------------------------

>>> mo.index("nov")
10

>>> ix = mo.index(["dec", "may"])  # multiple indices are returned as numpy array of integers
>>> ix
array([11,  4])

>>> mo[ix]  # ix is np.array
Index('month', ['dec' 'may'])


Renaming
--------

>>> m = mo.rename("M")
>>> m
Index('M', ['jan' 'feb' 'mar' 'apr' 'may' 'jun' 'jul' 'aug' 'sep' 'oct' 'nov' 'dec'])

Note: the original is unchanged!
>>> mo
Index('month', ['jan' 'feb' 'mar' 'apr' 'may' 'jun' 'jul' 'aug' 'sep' 'oct' 'nov' 'dec'])


Sorting
-------

#>>> mo.sort()  # not sure why would you do that with month names... but you can
#Index("month", ["apr" "aug" "dec" "jul" "jun" "mar" "may" "nov" "oct" "sep"])

Cube.axes return a tuple representing ordered collection of axes
>>> X = Index("X", ["x1", "x2"])
>>> Y = Index("Y", ["y1", "y2", "y3"])
>>> C = Cube(np.arange(6).reshape(2, 3), [X, Y])
>>> C.axes
(Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3']))

Axis can be accessed by index
>>> C.axes[0]
Index('X', ['x1' 'x2'])

# axis can be accessed by name
>>> C.axis("Y")
Index('Y', ['y1' 'y2' 'y3'])

# Axes can be used as an iterator to generate list, tuple, dict, etc.
>>> ax = list(C.axes)
>>> ax
[Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3'])]

# or can be used as an iterator in cycle
>>> for ax in C.axes: print(ax.name)
X
Y

# Cube class
>>> X = Index("X", ["x1", "x2"])
>>> Y = Index("Y", ["y1", "y2", "y3"])
>>> C = Cube(np.arange(6).reshape(2, 3), [X, Y])

>>> C.values
array([[0, 1, 2],
       [3, 4, 5]])

>>> np.sin(C)
axes: Axes(Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3']))
values: [[ 0.          0.84147098  0.90929743]
 [ 0.14112001 -0.7568025  -0.95892427]]

>>> np.cos(C)
axes: Axes(Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3']))
values: [[ 1.          0.54030231 -0.41614684]
 [-0.9899925  -0.65364362  0.28366219]]

>>> np.tan(C)
axes: Axes(Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3']))
values: [[ 0.          1.55740772 -2.18503986]
 [-0.14254654  1.15782128 -3.38051501]]

>>> D = Cube(np.arange(3), Y)  # note: if cube has only one axis, the axis does not need to be passed as a list

>>> D.values
array([0, 1, 2])

>>> (C + D).values
array([[0, 2, 4],
       [3, 5, 7]])
 
>>> E = Cube([0, 1], X)

>>> E.values
array([0, 1])

>>> (C + E).values
array([[0, 1, 2],
       [4, 5, 6]])

# axis alignment
 
>>> Y2 = Index("Y", ["y3", "y1", "y2"])
>>> D2 = Cube([2, 0, 1], Y2) 
>>> C + D2  # indices in Y2 are aligned to Y
axes: Axes(Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3']))
values: [[0 2 4]
 [3 5 7]]

>>> C.sum("Y")
axes: Axes(Index('X', ['x1' 'x2']))
values: [ 3 12]

>>> C.mean("X")
axes: Axes(Index('Y', ['y1' 'y2' 'y3']))
values: [ 1.5  2.5  3.5]

>>> C.mean()
axes: Axes()
values: 2.5

Cube broadcasting:

>>> X = Index("X", ["x1", "x2"])
>>> Y = Index("Y", ["y1", "y2", "y3"])
>>> C = Cube(np.arange(2), X)
>>> D = Cube(np.arange(3), Y)
>>> E = C * D
>>> E
axes: Axes(Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3']))
values: [[0 0 0]
 [0 1 2]]

Analogy to numpy.ndarray.compress function:
>>> E.compress([True, False, True], "Y")
axes: Axes(Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y3']))
values: [[0 0]
 [0 2]]

Analogy to numpy.ndarray.take function:
>>> E.take([0, 2], "Y")
axes: Axes(Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y3']))
values: [[0 0]
 [0 2]]



if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
"""