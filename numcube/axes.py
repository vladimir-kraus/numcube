import numpy as np

from .axis import Axis
from .index import Index


class Axes(object):
    """Ordered collection of axes.
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
    axes_dict = dict(name: axis for name, axis in enumerate(cube.axes))"""

    def __init__(self, axes):
        """If for non-unique axes are found, ValueError is raised.
        If axis has invalid type, TypeError is raised.
        :param axes: Axis or a collection of Axis objects (incl. another Axes object)"""
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
                raise TypeError("axis must be an instance of Axis")
            # test unique names - report the name of the first axis which is not unique
            if axis.name in unique_names:
                raise ValueError("multiple axes with name '{}'".format(axis.name))
            unique_names.add(axis.name)

        # the sequence of axes must be immutable
        self._names = tuple(axis.name for axis in axes)
        self._axes = tuple(axes)
        self._shape = tuple(len(a) for a in axes)

    def __repr__(self):
        axes = [str(a) for a in self._axes]
        return "Axes(" + ', '.join(axes) + ")"

    def __len__(self):
        return len(self._axes)

    def __getitem__(self, item):
        """Return axis given by its name, by index or by axis object.
        Note: passing axis object will return the same object if the axis is contained
        in the Axes object, or will raise LookupError if it is not contained."""
        return self._axes[self.index(item)]

    @property
    def items(self):
        return self._axes

    @property
    def names(self):
        """Return the names of axes.
        :return: tuple of strings"""
        return self._names

    @property
    def shape(self):
        return self._shape

    def axis_by_index(self, index):
        return self._axes[index]

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
            
    def is_unique_subset(self, axes):
        """Tests whether all axes are contained in the Axes object and whether they are unique."""
        raise NotImplementedError
        
    def complement(self, axes):
        """Return a tuple of indices of axes from Axes object which are not
        contained in the specified collection of axes.
        :param axes: collection of axes specified by axis name or index
        :returns: tuple of int"""
        if isinstance(axes, str) or isinstance(axes, int):
            axes = (axes,)
        indices = set(self.index(a) for a in axes)
        if len(indices) != len(axes):
            raise ValueError("axes are not unique")
        return tuple(i for i in range(len(self)) if i not in indices)
        
    def index(self, axis):
        """Find axis index by name, by index, or by axis object. If not found then raise an exception.
        When looked up by wrong name (str), KeyError is raised.
        When looked up by index (int), IndexError is raised.
        When looked up by axis object (axis), LookupError is raised.
        Note: LookupError can be used to catch all of the above.
        Otherwise TypeError is raised."""
        
        # find by numeric index, normalize negative numbers
        if isinstance(axis, int):
            axis_count = len(self._axes)
            if 0 <= axis < axis_count:
                return axis
            if -axis_count <= axis:
                # negative index is counted from the last axis backward
                return axis_count + axis
            raise IndexError("invalid axis index: {}".format(axis))
        
        # find by name
        if isinstance(axis, str):
            for i, a in enumerate(self._axes):
                if a.name == axis:
                    return i
            raise KeyError("invalid axis name: '{}'".format(axis))
        
        # find by object identity
        if isinstance(axis, Axis):
            for i, a in enumerate(self._axes):
                if a is axis:
                    return i
            raise ValueError("axis not found: {}".format(axis))

        raise TypeError("invalid axis identification type")

    def contains(self, axis_id):
        """Returns True if Axes contain an axis of the specified name. Otherwise return False."""
        try:
            self[axis_id]
            return True
        except LookupError:
            return False

    def transposed_indices(self, front, back):
        """Reorder axes by specified names or indices. Return a list of axis
        indices which correspond to the new order of axes."""
        if isinstance(front, str) or isinstance(front, int) or isinstance(front, Axis):
            front = [front]

        if isinstance(back, str) or isinstance(back, int) or isinstance(back, Axis):
            back = [back]

        front_axes = list()
        back_axes = list()
        temp_axes = list(self._axes)
        for axis_id in front:
            index = self.index(axis_id)
            front_axes.append(index)
            if temp_axes[index] is None:
                raise ValueError("duplicate axes in transpose")
            temp_axes[index] = None

        for axis_id in back:
            index = self.index(axis_id)
            back_axes.append(index)
            if temp_axes[index] is None:
                raise ValueError("duplicate axes in transpose")
            temp_axes[index] = None

        middle_axes = [index for index, axis in enumerate(temp_axes) if axis is not None]
        return front_axes + middle_axes + back_axes

    def insert(self, axis, index=0):
        """Insert a new axis at the specified position and return the new Axes object.
        :param axis: the new axis to be inserted
        :param index: the index of the new axis
        :return: new Axes object"""
        axis_list = list(self._axes)
        axis_list.insert(index, axis)
        return Axes(axis_list)

    def remove(self, axis_id):
        """Remove axis or axes with a given index or name.
        Return new Axes object."""
        axis_index = self.index(axis_id)
        return self._remove(axis_index)

    def _remove(self, axis_index):
        new_axes = list(self._axes)
        del new_axes[axis_index]
        return Axes(new_axes)

    def replace(self, old_axis_id, new_axis):
        """Replace an existing axis with a new axis and return the new Axes object.
        The new axes collection is checked for duplicate names.
        The old and new axes are NOT checked for equal lengths.
        :param old_axis_id: axis index (int) or name (str)
        :param new_axis: Series or Index object
        :return: new Axes object"""
        old_axis_index = self.index(old_axis_id)
        return self._replace(old_axis_index, new_axis)

    def _replace(self, old_axis_index, new_axis):
        new_axes = list(self._axes)
        new_axes[old_axis_index] = new_axis
        return Axes(new_axes)

    def swap(self, axis_id1, axis_id2):
        """Return a new Axes object with two axes swapped.
        :param axis_id1: name or index of the first axis
        :param axis_id2: name or index of the second axis
        :return: new Axes object"""
        index1 = self.index(axis_id1)
        index2 = self.index(axis_id2)
        new_axes = list(self)
        new_axes[index1], new_axes[index2] = new_axes[index2], new_axes[index1]
        return Axes(new_axes)

    def make_index(self, axis_id):
        """Convert an axis into Index. If the axis is already an Index, then does nothing."""
        axis = self[axis_id]
        if not isinstance(axis, Index):
            self.replace(axis_id, Index(axis.name, axis.values))

    def make_series(self, axis_id):
        """Convert an axis into Series. If the axis is already a Series, then does nothing."""
        axis = self[axis_id]
        if not isinstance(axis, Series):
            self.replace(axis_id, Series(axis.name, axis.values))


def intersect(axes1, axes2):
    """Return the space intersect of the common axes. The order of values on each axis correspond to the order on
    the corresponding axis on axes1. The result can be used for inner join operations.
    :param axes1: Axes object
    :param axes2: Axes object
    :return: Axes"""
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
