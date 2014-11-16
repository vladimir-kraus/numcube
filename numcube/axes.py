from .series import Series


class Axes(object):
    """
    Ordered collection of axes.
    Axis is either a Series or an Index.
    
    Get axis name of an axis of given index:
    ax_name = cube.axes[0].name
    
    Get index of axis of a given name:
    ax_index = cube.axes.index('a')
    
    Axes can be indexed using integer or string index:
    ax1 = cube.axes[0]
    ax_a = cube.axes['a']
    
    Axes can be used as iterables:
    axes_list = list(cube.axes)  
    # the above is better than [axis for axis in cube.axes]
    axes_dict = dict(name: axis for name, axis in enumerate(cube.axes))
    """

    def __init__(self, axes):
        """
        If for non-unique axes are found, ValueError is raised.
        """
        if isinstance(axes, Series):
            axes = [axes]

        # the sequence of axes must be immutable
        self._axes = tuple(axes)

        self._update_axis_indices()

    def _update_axis_indices(self):
        """
        Create inner axis dictionary to allow lookup by name.
        If for non-unique axes are found, ValueError is raised.
        :return: None
        """
        self._axis_dict = dict()
        for i, axis in enumerate(self._axes):
            if not isinstance(axis, Series):
                raise TypeError("axis must be an instance of Series or Index")

            name = axis.name
            if name in self._axis_dict:
                raise ValueError("duplicate axis {}".format(repr(name)))
            self._axis_dict[name] = i
        
    def __str__(self):
        axes = [str(axis) for axis in self._axes]
        return '\n'.join(axes)

    def __len__(self):
        return len(self._axes)

    def __getitem__(self, item):
        """
        Items in the axis collection can be accessed by index (int) or by name (str).

        Negative index can be used to access the items from the end of the collection. For example
        calling axes[-1] returns the last axis.

        If the item is not found, an exception is raised.
        - if item is string, KeyError is raised
        - if item is integer, IndexError is raised
        - otherwise TypeError is raised

        Note: LookupError can be used to catch both KeyError and IndexError.
        """
        if isinstance(item, str):
            item = self._axis_dict[item]
        return self._axes[item]
        
    def index(self, axis_id):
        """
        Find index of an axis by the name. If not found return KeyError.
        """
        # TODO: ...
        if isinstance(axis_id, str):
            return self._axis_dict[axis_id]
        else:
            return axis_id

    def contains(self, axis_name):
        """
        Returns True if Axes contain an axis of the specified name. Otherwise return False.
        """
        return axis_name in self._axis_dict

    def names(self):
        """
        Return iterator over axis names.
        :return:
        """
        for axis in self._axes:
            yield axis.name

    def transpose(self, axes):
        """
        Reorder axes by specified names or indices. Return a new Axes object.
        """
        axes_count = len(axes)

        if axes_count != len(self._axes):
            raise ValueError("invalid number of axes")

        new_axes = [self[axis] for axis in axes]

        # check duplicate axes
        axis_set = set(new_axes)
        if len(axis_set) != axes_count:
            raise ValueError("duplicit axes")

        return Axes(new_axes)
        
    def subset(self, axes):
        """
        axes is an sequence of strings and/or integers
        """
        return [self[axis] for axis in axes]
        
    def _get_index(self, axis):
        if isinstance(axis, str):
            return self.index(axis)
        else:
            return axis

    def replace(self, old_axis_id, new_axis):
        """
        Replace an existing axis with a new axis and return the new Axes object.
        The new axes collection is checked for duplicit names.
        The old and new axes are NOT checked for equal lengths.
        :param old_axis_id: axis index (int) or name (str)
        :param new_axis: Series or Index object
        :return: new Axes object
        """
        old_axis_index = self._get_index(old_axis_id)
        new_axes = list(self._axes)
        new_axes[old_axis_index] = new_axis
        return Axes(new_axes)

    def swap(self, axis_id1, axis_id2):
        """
        Return a new Axes object with two axes swapped.
        :param axis_id1: name or index of the first axis
        :param axis_id2: name or index of the second axis
        :return: new Axes object
        """
        index1 = self.index(axis_id1)
        index2 = self.index(axis_id2)
        new_axes = list(self)
        new_axes[index1], new_axes[index2] = new_axes[index2], new_axes[index1]
        return Axes(new_axes)