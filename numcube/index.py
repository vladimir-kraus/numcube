import numpy as np

from numcube.axis import Axis


class Index(Axis):
    """A named sequence of unique indexed values. Can be used as indexable axis in Cube.
    Name is a string. Values are stored in one-dimensional numpy array.
    """

    def __init__(self, name, values):
        """
        Initialize a new Index object. The values must be unique, otherwise ValueError is raised.
        :param name: str
        :param values: sequence of values (must be unique)
        """
        super(Index, self).__init__(name, values)

        # create dictionary
        self._indices = {x: i for i, x in enumerate(self._values)}

        # values must not be change once the index has been created
        self._values.flags.writeable = False
        
        if len(self._indices) != len(self._values):
            raise ValueError('index has duplicate values')
        
        self._vec_index = np.vectorize(self._indices.__getitem__, otypes=[np.int])
        self._vec_contains = np.vectorize(self._indices.__contains__, otypes=[np.bool])
        
    def indexof(self, item):
        """If item is single value, then return a single integer value.
        If item is a sequence, then return numpy array of integers.
        :param item: a single value or a sequence of values
        :return: int or numpy array of ints
        :raises: KeyError
        """
        v = self._vec_index(item)
        if v.ndim > 0:
            return v
        return v.item()

    def contains(self, item):
        """If item is single value, then return a single boolean value.
        If item is a sequence, then return numpy array of booleans.
        :param item: a single value or a sequence of values
        :return: bool or numpy array of bools
        """
        v = self._vec_contains(item)
        if v.ndim > 0:
            return v
        return v.item()