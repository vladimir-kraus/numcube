"""
Index class
===========
Index is an axis (currently implemented by sublassing Axis) with unique values, which can be used for lookups.
Currently the values are stored in a one-dimensional write-protected numpy array.

>>> from numcube import Index

>>> materials = Index("material", ["stone", "wood", "metal", "glass", "plastic"])
>>> materials
Index('material', ['stone' 'wood' 'metal' 'glass' 'plastic'])

Finding index of a values
-------------------------

Index of a single value is returned as a plain integer.

>>> materials.indexof("glass")
3

Multiple indices are returned as a numpy array of integers.

>>> ix = materials.indexof(["wood", "plastic"])
>>> ix
array([1, 4])

And here is how we can use the indices returned from the previous step.

>>> materials[ix]  # ix is np.array
Index('material', ['wood' 'plastic'])


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
"""