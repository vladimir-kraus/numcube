import numpy as np
from .axis import Axis


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
        If axis has invalid type, TypeError is raised.
        :param axes: Axis or a collection of Axis objects (incl. another Axes object)
        """
        # special case with zero axes
        if axes is None:
            self._axes = tuple()
            self._shape = tuple()
            return

        # special case with a single axis
        if isinstance(axes, Axis):
            axes = [axes]

        unique_names = set()
        for axis in axes:
            # test correct types
            if not isinstance(axis, Axis):
                raise TypeError("axis must be instance of Index or Series")
            # test unique names - report the name of the first axis which is not unique
            if axis.name in unique_names:
                raise ValueError("multiple axes with name '{}'".format(axis.name))
            unique_names.add(axis.name)

        # the sequence of axes must be immutable
        self._axes = tuple(axes)
        self._shape = tuple(len(a) for a in axes)

    def __repr__(self):
        axes = [str(a) for a in self._axes]
        return "Axes(" + ', '.join(axes) + ")"

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
        if isinstance(item, int):
            if 0 <= item < len(self._axes):
                return self._axes[item]
            raise IndexError("invalid axis index: {}".format(item))
        elif isinstance(item, str):
            for a in self._axes:
                if a.name == item:
                    return a
            raise KeyError("invalid axis name: '{}'".format(item))
        else:
            raise TypeError("axis can be specified by index (int) or name (str)")
            
    @property
    def shape(self):
        return self._shape

    def axis_and_index(self, axis):
        if isinstance(axis, int):
            if 0 <= axis < len(self._axes):
                return self._axes[axis], axis
            raise IndexError("invalid axis index: {}".format(axis))
        elif isinstance(axis, str):
            for i, a in enumerate(self._axes):
                if a.name == axis:
                    return a, i
            raise KeyError("invalid axis name: '{}'".format(axis))
        else:
            raise TypeError("axis can be specified by index (int) or name (str)")
        
    def index(self, axis):
        """
        Find index of an axis by the name. If not found return KeyError.
        """
        if isinstance(axis, int):
            if 0 <= axis < len(self._axes):
                return axis
            raise IndexError("invalid axis index: {}".format(axis))
        elif isinstance(axis, str):
            for i, a in enumerate(self._axes):
                if a.name == axis:
                    return i
            raise KeyError("invalid axis name: '{}'".format(axis))
        else:
            raise TypeError("axis can be specified by index (int) or name (str)")

    def contains(self, axis_id):
        """
        Returns True if Axes contain an axis of the specified name. Otherwise return False.
        """
        try:
            self[axis_id]
            return True
        except LookupError:
            return False

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

    def insert(self, axis, index=0):
        """
        Insert a new axis at the specified position and return the new Axes object.
        :param axis: the new axis to be inserted
        :param index: the index of the new axis
        :return: new Axes object
        """
        axis_list = list(self._axes)
        axis_list.insert(index, axis)
        return Axes(axis_list)

    def remove(self, axis):
        """
        Remove axis or axes with a given index or name.
        Return new Axes object.
        """
        i = self.index(axis)
        new_axes = list(self._axes)
        del new_axes[i]
        return Axes(new_axes)

    def replace(self, old_axis_id, new_axis):
        """
        Replace an existing axis with a new axis and return the new Axes object.
        The new axes collection is checked for duplicit names.
        The old and new axes are NOT checked for equal lengths.
        :param old_axis_id: axis index (int) or name (str)
        :param new_axis: Series or Index object
        :return: new Axes object
        """
        old_axis_index = self.index(old_axis_id)
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


def intersect(axes1, axes2):
    """
    Return the space intersect of the common axes. The order of values on each axis correspond to the order on
    the corresponding axis on axes1.

    The result can be used for inner join operations.

    :param axes1: Axes object
    :param axes2: Axes object
    :return: Axes
    """
    common_axes = []
    for axis1 in axes1:
        try:
            axis2 = axes2[axis1.name]
        except LookupError:
            continue

        if axis1 is axes2:
            axis = axis1
        else:
            indices = np.in1d(axis1.values, axis2.values)  # assume_unique=True ?
            axis = axis1[indices]

        common_axes.append(axis)

    return Axes(common_axes)