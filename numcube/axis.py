import numpy as np


class Axis(object):
    """A named sequence of values. Can be used as non-indexable axis in Cube.
    Name is a string. Values are stored in one-dimensional numpy array.
    """

    def __init__(self, name, values):
        """Initializes Axis object.
        :param name: str
        :param values: sequence of values of the same type, are converted to 1-D numpy array
        :raise: ValueError if values cannot be converted, TypeError if name is not string
        """
        if not isinstance(name, str):
            raise TypeError("type of {} is not str".format(repr(name)))
        values = np.atleast_1d(values)
        if values.ndim > 1:
            raise ValueError("values must not have more than 1 dimension")
        self._name = name
        self._values = values

    def __repr__(self):
        """Returns textual representation of Axis object. Can be reused by inherited classes.
        :return: str
        """
        return "{}('{}', {})".format(self.__class__.__name__, self._name, self._values)
        
    def __len__(self):
        """Returns the number of elements in (the length) the axis.
        :return: int
        """
        return len(self._values)

    def __getitem__(self, item):
        """
        :param item:
        :return: a new Axis object
        """
        return self.__class__(self._name, self._values[item])

    @property
    def name(self):
        """Returns he name of the axis.
        :return: str
        """
        return self._name

    @property
    def values(self):
        """Returns one-dimensional numpy.ndarray of axis values.
        :return: numpy.ndarray
        """
        return self._values  # TODO: view?

    def filter(self, values):
        """Filter axis elements which are contained in values. The axis order is preserved.
        :param values: a value or a list, set, tuple or numpy array of values
            the order or values is irrelevant, need not be unique
        :return:
        """
        if isinstance(values, set):
            values = list(values)
        values = np.asarray(values)
        selection = np.in1d(self._values, values)
        return self.__class__(self._name, self._values[selection])
        
    def take(self, indices):
        """Analogy to numpy.ndarray.take.
        :return: a new Axis object
        """
        return self.__class__(self._name, self._values.take(indices))
        
    def compress(self, condition):
        """Analogy to numpy.ndarray.compress.
        :return: a new Axis object
        """
        return self.__class__(self._name, self._values.compress(condition))

    def rename(self, new_name):
        """Returns a new object (of type Axis or the actual derived type) with the new name and the same values.
        :param new_name: str
        :return: new axis (instance of actual derived type)
        """
        return self.__class__(new_name, self._values)

    def sort(self):
        """Sorts the values.
        :return: a new Axis object with sorted values
        """
        return self.__class__(self._name, np.sort(self._values))
