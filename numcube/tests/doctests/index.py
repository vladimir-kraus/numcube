"""
Index class
===========
Index is an axis (currently implemented by sublassing Axis) with unique values, which can be used for lookups.
Currently the values are stored in a one-dimensional write-protected numpy array.

>>> from numcube import Index

>>> materials = Index("material", ["stone", "wood", "metal", "glass", "plastic"])
>>> materials
Index('material', ['stone' 'wood' 'metal' 'glass' 'plastic'])

Checking existence of values
----------------------------

Checking whether a single value exists in the index returns a single bool value.

>>> materials.contains("glass")
True

Checking whether multiple values exist in the index returns an array of bool values.

>>> materials.contains(["glass", "unknown"])
array([ True, False], dtype=bool)

Finding index of a values
-------------------------

Index of a single value is returned as a plain integer.

>>> materials.indexof("glass")
3

Multiple indices are returned as a numpy array of integers.

>>> materials.indexof(["wood", "plastic"])
array([1, 4])

And here is how we can use the indices returned from the previous step.

>>> ix = materials.indexof(["wood", "plastic"])
>>> materials[ix]  # ix is np.array
Index('material', ['wood' 'plastic'])


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
"""
