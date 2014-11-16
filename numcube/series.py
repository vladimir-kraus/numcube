import numpy as np


class Series:
    """
    A named wrapper around numpy array.
    """

    def __init__(self, name, values):
        """
        Can also be created from a single dict item: Series({"a": [10, 20, 30]})
        """
        if not isinstance(name, str):
            raise TypeError('type of {} is not str'.format(repr(name)))
        self._name = name
        self._values = np.array(values)
        
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
        return Series(new_name, self._values)

    def reorder(self, indices):
        return Series(self._name, self._values[indices])