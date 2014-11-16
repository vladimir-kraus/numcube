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
        return "Series('{}', {})".format(self._name, self._values)

    def __len__(self):
        return len(self._values)

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        return self._values.view()

    def rename(self, new_name):
        """Returns a new Series object with the new name and the same values."""
        return Series(new_name, self._values)

    def reorder(self, indices):
        """Returns a new Series object with the same name and reordered values."""
        return Series(self._name, self._values[indices])