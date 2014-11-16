import numpy as np
from .series import Series


class Index(Series):
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
        
    def __str__(self):
        return "Index('{}', {})".format(self._name, self._values)        
        
    def index(self, value):
        """Return numpy array of indices of the given values."""
        return self._vec_index(value)

    def take(self, indices):
        """Analogy to ndarray.take."""
        return Index(self._name, self._values.take(indices))

    def rename(self, new_name):
        """Returns a new Index object with the new name and the same values."""
        return Index(new_name, self._values)

    def reorder(self, indices):
        """Returns a new Index object with the same name and reordered values."""
        return Index(self._name, self._values[indices])