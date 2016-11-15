import numpy as np


class MultiAxis(object):
    """A named sequence of values. Can be used as non-indexable axis in Cube.
    Name is a tuple of strings. Values are stored in one-dimensional numpy recarray.
    """
    @classmethod
    def from_axes(cls, axes):
        recarray = None  # prepare recarray
        return cls(recarray)

    def __init__(self, recarray):
        """Initializes MultiAxis object.
        :param recarray: numpy.recarray object
        """
        self._values = np.atleast_1d(recarray)
        if self._values.ndim > 1:
            raise ValueError("values must not have more than 1 dimension")

    def __repr__(self):
        return "{}('{}', {})".format(self.__class__.__name__, self.name, self.values)
        
    def __len__(self):
        """Returns the number of elements."""
        return len(self._values)

    def __getitem__(self, item):
        return self.__class__(self._values[item])

    @property
    def name(self):
        return self._values.dtype.names

    @property
    def size(self):
        return self._values.size

    @property
    def values(self):
        return self._values

    def nseries(self):
        return len(self._values.dtype)

    def filter(self, values):
        """Filter axis elements which are contained in values. The axis order is preserved.
        :param values: a value or a list, set, tuple or numpy array of values
            the order or values is irrelevant, need not be unique
        """
        if isinstance(values, set):
            values = list(values)
        values = np.asarray(values)
        selection = np.in1d(self._values, values)
        return self.__class__(self._name, self._values[selection])
        
    def take(self, indices):
        """Analogy to numpy.ndarray.take."""
        return self.__class__(self._values.take(indices))
        
    def compress(self, condition):
        """Analogy to numpy.ndarray.compress."""
        return self.__class__(self._name, self._values.compress(condition))

    def rename(self, new_name):
        """Returns a new object (of type Axis or the actual derived type) with the new name and the same values.
        :param new_name: str
        :return: new axis (instance of actual derived type)
        """
        return self.__class__(new_name, self._values)
