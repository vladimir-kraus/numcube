import numpy as np


class MultiIndex(object):
    """A named sequence of values. Can be used as non-indexable axis in Cube.
    Name is a tuple of strings. Values are stored in one-dimensional numpy recarray.
    """
    # TODO

    @classmethod
    def from_axes(cls, axes):
        recarray = None  # prepare recarray
        return cls(recarray)
