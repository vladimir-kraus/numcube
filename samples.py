# Index class
# - Index is a named, indexed collection of unique values

>>> yr = Index("year", range(2010, 10))

>>> mo = Index("month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])

>>> mo.name

>>> mo.values

>>> # indexing is the same as indexing of one dimensional numpy array

>>> mo[1]

>>> mo[-1]

>>> # slicing

>>> mo[0:4]

>>> mo[::2]

>>> mo.index("nov")

>>> # index can take only values, cannot mix with indices because in case of integer values, it would be uncertain whether the number is considered a value or index

>>> ix = mo.index(["dec", "may"])

>>> ix  
[11 4]

>>> mo[ix]  # ix is np.array
Index("month", ["dec" "may"])

>>> mo.take([0, 2, 4])  # if you use a list of indices, use take method
Index("month", ["jan" "mar" "may"])

>>> yr[yr.values % 2 == 0]  # even years
Index("year", [2010 2012 2014 2016 2018])

>>> yr[2013 <= yr.values <= 2016]
Index("year", [2013 2014 2015 2016])

>>> mo.rename("M")

>>> mo.sort()  # not sure why would you do that with month names... but you can
Index("month", ["apr" "aug" "dec" "jul" "jun" "mar" "may" "nov" "oct" "sep"])

# Axes class
# - you usually do not need to create Axes manually
# - Axes is a part of Cube

>>> X = Index("X", ["x1", "x2"])
>>> Y = Index("Y", ["y1", "y2", "y3"])
>>> C = Cube(np.arange(6).reshape(2, 3), [X, Y])
>>> C.axes
Axes(Index("X", ["x1" "x2"]), Index("Y", ["y1" "y2" "y3"])

# axis can be accessed by index
>>> C.axes[0]
>>> Index("X", ["x1" "x2"])

# axis can be accessed by name
>>> C.axes["Y"]
>>> Index("Y", ["y1" "y2" "y3"])

# Axes can be used as an iterator to generate list, tuple, dict, etc.
>>> ax = list(C.axes)
[Index("X", ["x1" "x2"]) Index("Y", ["y1" "y2" "y3"])]

# or can be used as an iterator in cycle
>>> for ax in C.axes: print(ax.name)
X
Y

# Cube class
>>> X = Index("X", ["x1", "x2"])

>>> Y = Index("Y", ["y1", "y2", "y3"])

>>> C = Cube(np.arange(6).reshape(2, 3), [X, Y])

>>> C
[[0 1 2]
 [3 4 5]]

>>> D = Cube(np.arange(3), Y)  # note: if cube has only one axis, the axis does not need to be in list

>>> D
[0 1 2]

>>> C + D
[[0 2 4]
 [3 5 7]]

# axis alignment
 
>>> Y2 = Index("Y", ["y3", "y1", "y2"])
>>> D2 = Cube([2, 0, 1], Y2) 
>>> C + D2  # indices in Y2 are aligned to Y
[[0 2 4]
 [3 5 7]]

>>> C.sum("Y")
[3 12]

>>> C.mean("X")
[1.5 2.5 4.5]

>>> C.mean()
2.5

# cube broadcasting

>>> X = Index("X", ["x1", "x2"])
>>> Y = Index("Y", ["y1", "y2", "y3"])
>>> C = Cube(np.arange(2), X)
>>> D = Cube(np.arange(3), Y)
>>> C * D
[[0 0 0]
 [0 1 2]]
