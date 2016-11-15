"""
Cube class
==========

>>> from numcube import Index, Cube
>>> import numpy as np

To create a cube, you have to provide array of values and collection of axes. The shape of the array must
correspond to the lengths of the axes.

>>> X = Index("X", ["x1", "x2"])
>>> Y = Index("Y", ["y1", "y2", "y3"])
>>> C = Cube([[0, 1, 2], [3, 4, 5]], [X, Y])

Axes can be accessed by index (int) or by name (str). This holds true for most
of other functions which take axis as an argument.

>>> C.axis(0)
Index('X', ['x1' 'x2'])

>>> C.axis("Y")
Index('Y', ['y1' 'y2' 'y3'])

Axes can be used as an iterator to generate list, tuple, dict, etc.

>>> for axis in C.axes: print(axis)
Index('X', ['x1' 'x2'])
Index('Y', ['y1' 'y2' 'y3'])

>>> list(C.axes)
[Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3'])]

>>> tuple(C.axes)
(Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3']))

# Cube class
>>> X = Index("X", ["x1", "x2"])
>>> Y = Index("Y", ["y1", "y2", "y3"])
>>> C = Cube([[0, 1, 2], [3, 4, 5]], [X, Y])
>>> C  # doctest: +NORMALIZE_WHITESPACE
Cube([[0 1 2]
      [3 4 5]], (Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3'])))

>>> C.values  # doctest: +NORMALIZE_WHITESPACE
array([[0, 1, 2],
       [3, 4, 5]])

Mathematical functions available in numpy package such as sin, cos, tan, exp, etc.

>>> np.sin(C)  # doctest: +NORMALIZE_WHITESPACE
Cube([[ 0.          0.84147098  0.90929743]
      [ 0.14112001 -0.7568025  -0.95892427]], (Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3'])))

>>> np.cos(C)  # doctest: +NORMALIZE_WHITESPACE
Cube([[ 1.          0.54030231 -0.41614684]
     [-0.9899925  -0.65364362  0.28366219]], (Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3'])))

>>> np.tan(C)  # doctest: +NORMALIZE_WHITESPACE
Cube([[ 0.          1.55740772 -2.18503986]
      [-0.14254654  1.15782128 -3.38051501]], (Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3'])))

Operations with scalars.

>>> C + 1  # doctest: +NORMALIZE_WHITESPACE
Cube([[1 2 3]
      [4 5 6]], (Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3'])))

Note: if cube has only one axis, the axis does not need to be passed as a list.

>>> D = Cube(np.arange(3), Y)

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

The following operation will align values Y2 to Y values.

>>> C + D2  # doctest: +NORMALIZE_WHITESPACE
Cube([[0 2 4]
      [3 5 7]], (Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3'])))

>>> C.sum("Y")  # doctest: +NORMALIZE_WHITESPACE
Cube([ 3 12], (Index('X', ['x1' 'x2']),))

>>> C.mean("X")  # doctest: +NORMALIZE_WHITESPACE
Cube([ 1.5  2.5  3.5], (Index('Y', ['y1' 'y2' 'y3']),))

>>> C.mean()
2.5

Cube broadcasting:

>>> X = Index("X", ["x1", "x2"])
>>> Y = Index("Y", ["y1", "y2", "y3"])
>>> C = Cube(np.arange(2), X)
>>> D = Cube(np.arange(3), Y)
>>> E = C * D
>>> E  # doctest: +NORMALIZE_WHITESPACE
Cube([[0 0 0]
      [0 1 2]], (Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y2' 'y3'])))

Analogy to numpy.ndarray.compress function:
>>> E.compress("Y", [True, False, True])  # doctest: +NORMALIZE_WHITESPACE
Cube([[0 0]
      [0 2]], (Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y3'])))

Analogy to numpy.ndarray.take function:
>>> E.take("Y", [0, 2])  # doctest: +NORMALIZE_WHITESPACE
Cube([[0 0]
      [0 2]], (Index('X', ['x1' 'x2']), Index('Y', ['y1' 'y3'])))



if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
"""
