import numpy as np


class Series(object):
    """
    A named wrapper around numpy array.
    """

    def __init__(self, name, values):
        """
        :param name: str
        :param values: sequence of values (need not be unique)
        """
        if not isinstance(name, str):
            raise TypeError('type of {} is not str'.format(repr(name)))
        self._name = name
        self._values = np.asarray(values)
        
    def __str__(self):
        T = type(self)
        return "{}('{}', {})".format(T.__name__, self._name, self._values)

    def __len__(self):
        return len(self._values)

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        return self._values  # .view()

    def take(self, indices):
        """
        Returns a new object (of type Series or actual derived type) with the same name and reordered values.
        Analogy to ndarray.take.
        """
        T = type(self)
        return T(self._name, self._values.take(indices))

    def rename(self, new_name):
        """
        Returns a new object (of type Series or actual derived type) with the new name and the same values.
        """
        T = type(self)
        return T(new_name, self._values)
