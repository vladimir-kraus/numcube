import numpy as np
from .axis import Axis


class Index(Axis):
    """
    A series of unique indexed values.
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
            raise ValueError('index has duplicit values')
        
        self._vec_index = np.vectorize(self._indices.__getitem__, otypes=[np.int])
        self._vec_contains = np.vectorize(self._indices.__contains__, otypes=[np.bool])

    def index(self, item):
        """
        If item is single value, then return a single integer value.
        If item is a sequence, then return numpy array of booleans.
        :param item: a single value or a sequence of values
        :return: int or numpy array of ints
        :raises: KeyError
        """
        return self._vec_index(item)

    def contains(self, item):
        """
        If item is single value, then return a single boolean value.
        If item is a sequence, then return numpy array of booleans.
        :param item: a single value or a sequence of values
        :return: bool or numpy array of bools
        """
        return self._vec_contains(item)


