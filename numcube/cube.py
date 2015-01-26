import numpy as np
import math
from .index import Index
from .series import Series
from .axes import Axes
from .axis import Axis
from .exceptions import AxisAlignError


class Cube(object):
    """
    Wrapper around numpy.ndarray with named and labelled axes.
    """

    # when numpy array is the first argument in operation and Cube is the second,
    # then __array_priority__ will force Cube to handle the operation rather than numpy array
    __array_priority__ = 10

    def __init__(self, values, axes, dtype=None):
        values = np.asarray(values, dtype)

        # convert a collection of axes to Axes object
        if not isinstance(axes, Axes):
            axes = Axes(axes)

        # if axes dimensions and value dimension do not match
        if values.ndim != len(axes):
            raise ValueError("invalid number of axes")

        for n, axis in zip(values.shape, axes):
            if n != len(axis):
                raise ValueError("invalid length of axis '{}'".format(axis.name))

        self._values = values
        self._axes = axes

    def __getitem__(self, items):
        """Similar rules apply as with indexing and slicing numpy ndarray.
        Notes:
        1) np.newaxis is not supported.
        2) axes indexed by integer are collapsed

        :param item:
        :return:
        """

        if not isinstance(items, tuple):
            items = (items,)
        new_axes = []

        # append the axes given by items
        for item, axis in zip(items, self._axes):
            if not isinstance(item, int):  # indexing by int collapses a dimension
                new_axes.append(axis[item])

        # append the rest of axes
        for i in range(len(items), len(self._axes)):
            new_axes.append(self._axes[i])

        return Cube(self._values[items], new_axes)

    def __bool__(self):
        """Return the truth value of the cube.
        If the cube is empty, returns False.
        If the cube is scalar, returns the truth value of the only element.
        If the cube has more than one element, ValueError is raised.
        Note: The function returns the truth value of the underlying numpy ndarray.
        """
        return bool(self._values)

    def __contains__(self, item):
        """Implements the behaviour of built-in 'in' operator."""
        return item in self._values

    def __repr__(self):
        return "axes: {}\nvalues: {}".format(self._axes, self._values)

    @property
    def shape(self):
        return self._values.shape

    @property
    def size(self):
        return self._values.size

    @property
    def ndim(self):
        """Number of array dimensions."""
        return self._values.ndim

    @property
    def values(self):
        return self._values  # .view()

    @property
    def axes(self):
        """
        :return: tuple of axes
        :rtype: tuple
        """
        return self._axes.items
        # return self._axes.items

    @property
    def axis_names(self):
        return self._axes.names

    def axis(self, item):
        """Return axis by name or index."""
        return self._axes[item]

    def axis_index(self, axis):
        return self._axes.index(axis)

    def has_axis(self, axis):
        return self._axes.contains(axis)

    def apply(self, func, *args):
        """Apply function to all values and return the new cube.
        :param func: function to be applied to values
        :param args: additional optional arguments of func
        :return: Cube
        """
        return Cube(func(self._values, *args), self._axes)

    def transpose(self, axes):
        """Analogy to numpy.transpose.
        :param axes: axis names or indices defining the new order of axes
        :return: new Cube object
        """
        if len(axes) != self.ndim:
            raise ValueError("invalid number of axes")

        new_axes = self._axes.transpose(axes)
        indices = np.array([self._axes.index(a) for a in axes])
        new_values = self._values.transpose(indices)
        return Cube(new_values, new_axes)
        
    """arithmetics operators"""

    # unary +
    def __pos__(self):
        return self

    # unary -
    def __neg__(self):
        return Cube(-self._values, self._axes)

    # A + B
    def __add__(self, other):
        return apply2(self, other, np.add)

    def __radd__(self, other):
        return apply2(other, self, np.add)

    # A * B
    def __mul__(self, other):
        return apply2(self, other, np.multiply)

    def __rmul__(self, other):
        return apply2(other, self, np.multiply)

    # A - B
    def __sub__(self, other):
        return apply2(self, other, np.subtract)

    def __rsub__(self, other):
        return apply2(other, self, np.subtract)

    # A / B
    def __truediv__(self, other):
        return apply2(self, other, np.true_divide)

    def __rtruediv__(self, other):
        return apply2(other, self, np.true_divide)

    # A // B
    def __floordiv__(self, other):
        return apply2(self, other, np.floor_divide)

    def __rfloordiv__(self, other):
        return apply2(other, self, np.floor_divide)

    # A ** B
    def __pow__(self, other):
        return apply2(self, other, np.power)

    def __rpow__(self, other):
        return apply2(other, self, np.power)

    # A % B
    def __mod__(self, other):
        return apply2(self, other, np.mod)

    def __rmod__(self, other):
        return apply2(other, self, np.mod)

    """bitwise operators"""

    def __invert__(self):
        """Compute bit-wise inversion, or bit-wise NOT, element-wise."""
        return Cube(np.invert(self._values), self._axes)

    # A & B
    def __and__(self, other):
        return apply2(self, other, np.bitwise_and)

    def __rand__(self, other):
        return apply2(other, self, np.bitwise_and)

    # A | B
    def __or__(self, other):
        return apply2(self, other, np.bitwise_or)

    def __ror__(self, other):
        return apply2(other, self, np.bitwise_or)

    # A ^ B
    def __xor__(self, other):
        return apply2(self, other, np.bitwise_xor)

    def __rxor__(self, other):
        return apply2(other, self, np.bitwise_xor)

    # A >> B
    def __lshift__(self, other):
        return apply2(self, other, np.left_shift)

    def __rlshift__(self, other):
        return apply2(other, self, np.left_shift)

    # A << B
    def __rshift__(self, other):
        return apply2(self, other, np.right_shift)

    def __rrshift__(self, other):
        return apply2(other, self, np.right_shift)

    """comparison operators"""

    # A == B
    def __eq__(self, other):
        return apply2(self, other, np.equal)

    # A != B
    def __ne__(self, other):
        return apply2(self, other, np.not_equal)

    # A < B
    def __lt__(self, other):
        return apply2(self, other, np.less)

    # A <= B
    def __le__(self, other):
        return apply2(self, other, np.less_equal)

    # A > B
    def __gt__(self, other):
        return apply2(self, other, np.greater)

    # A >= B
    def __ge__(self, other):
        return apply2(self, other, np.greater_equal)

    """mathematical functions"""

    def __abs__(self):
        """Implements behavior for the built in abs() function."""
        return Cube(abs(self._values), self._axes)

    def __round__(self, n):
        """Implements behavior for the built in round() function. n is the number of decimal places to round to."""
        return Cube(round(self._values, n), self._axes)

    def __floor__(self):
        """Implements behavior for math.floor(), i.e., rounding down to the nearest integer."""
        return Cube(math.floor(self._values), self._axes)

    def __ceil__(self):
        """Implements behavior for math.ceil(), i.e., rounding up to the nearest integer."""
        return Cube(math.ceil(self._values), self._axes)

    def __trunc__(self):
        """Implements behavior for math.trunc(), i.e., truncating to an integral."""
        return Cube(math.trunc(self._values), self._axes)

    def sin(self):
        """Sine, element-wise.
        Can be called as numpy.sin(C) or C.sin()."""
        return Cube(np.sin(self._values), self._axes)

    def cos(self):
        """Cosine, element-wise.
        Can be called as numpy.cos(C) or C.cos()."""
        return Cube(np.cos(self._values), self._axes)

    def tan(self):
        """Tangents, element-wise.
        Can be called as numpy.tan(C) or C.tan()."""
        return Cube(np.tan(self._values), self._axes)

    """aggregation functions"""

    def sum(self, axis=None, keep=None):
        """Sum of array elements over a given axis.

        :param axis: Axis or axes along which a sum is performed. The default (axis = None) is perform a sum
        over all the dimensions of the input array. axis may be negative, in which case it counts from the last
        to the first axis. If this is a tuple of ints, a sum is performed on multiple axes, instead of a single
        axis or all the axes as before.
         :return:
        """
        return self.aggregate(np.sum, axis, keep)

    def mean(self, axis=None, keep=None):
        """Compute the arithmetic mean along the specified axis."""
        return self.aggregate(np.mean, axis, keep)

    def min(self, axis=None, keep=None):
        """Return the minimum of a cube or minimum along an axis."""
        return self.aggregate(np.min, axis, keep)

    def max(self, axis=None, keep=None):
        """Return the maximum of a cube or maximum along an axis."""
        return self.aggregate(np.max, axis, keep)

    def all(self, axis=None, keep=None):
        """Test whether all cube elements along a given axis evaluate to True."""
        return self.aggregate(np.all, axis, keep)

    def any(self, axis=None, keep=None):
        """Test whether any cube element along a given axis evaluates to True."""
        return self.aggregate(np.any, axis, keep)

    def aggregate(self, func, axis=None, keep=None):
        """
        :param axis: axis which are eliminated by the aggregation
        :param keep: axis or axes which are kept after the aggregation
        Note: if keep is not None, then axis must be None, otherwise ValueError is raised.
        """

        # complete aggregation to a scalar
        if axis is None and keep is None:
            return Cube(func(self._values), None)

        if keep is not None and axis is not None:
            raise ValueError("either 'keep' or 'axis' argument must be None")

        if isinstance(axis, str) or isinstance(axis, int):
            axis = [axis]

        if isinstance(keep, str) or isinstance(keep, int):
            keep = [keep]

        if axis is not None:
            axis_indices_to_remove = tuple(self._axes.index(a) for a in axis)
            new_axes = list(a for i, a in enumerate(self._axes) if i not in axis_indices_to_remove)
        else:
            axis_index_set = set(self._axes.index(a) for a in keep)
            new_axes = list(a for i, a in enumerate(self._axes) if i in axis_index_set)
            axis_indices_to_remove = tuple(set(range(self.ndim)) - axis_index_set)

        new_values = self._values
        if axis_indices_to_remove:
            new_values = func(new_values, axis_indices_to_remove)

        return Cube(new_values, new_axes)

    def groupby(self, axis, func, sorted=True, *args):  # **kwargs): # since numpy 1.9
        """
        :param axis:
        :param func:
            - a function which takes two fixed arguments - array and axis (in this order) 
            - following these two can also take a variable number of other arguments passed in *args
            - must return array with one axes less then the input array
            - examples are np.sum, np.mean, etc.
        :param sorted:
        """
        old_axis, old_axis_index = self._axes.axis_and_index(axis)  # TODO - do not use private accessor
        
        if isinstance(old_axis, Index):
            return self  # Index already contains unique values
            
        sub_cubes = list()   
        
        # unique values to be sorted
        if sorted:
            unique_values = np.unique(old_axis.values)
        else:
            unique_values, unique_indices = np.unique(old_axis.values, return_index=True)
            index_array = np.argsort(unique_indices)
            unique_values = unique_values[index_array]
        
        old_values = old_axis.values
        all_indices = np.arange(len(old_values))
        for value in unique_values:
            indices = all_indices[old_values == value]
            sub_cube = self._values.take(indices, old_axis_index)
            sub_cube = np.apply_along_axis(func, old_axis_index, sub_cube, *args)  #, **kwargs) # since numpy 1.9
            sub_cube = np.expand_dims(sub_cube, old_axis_index)
            sub_cubes.append(sub_cube)
        
        # the created axis is Index because it has unique values
        new_axis = Index(old_axis.name, unique_values)
        new_axes = self._axes.replace(old_axis_index, new_axis)
        new_values = np.concatenate(sub_cubes, old_axis_index)
        return Cube(new_values, new_axes)

    def _filter_axis(self, axis):
        self_axis, self_axis_index = self._axes.axis_and_index(axis.name)
        value_indices = self_axis.index(axis.values)
        new_values = self._values.take(value_indices, self_axis_index)
        new_axes = self._axes.replace(self_axis_index, axis)
        return Cube(new_values, new_axes)

    def replace_axis(self, old_axis_id, new_axis):
        """
        Replace an existing axis with a new axis and return the new Cube object.
        The new axes collection is checked for duplicate names.
        The new axis must have the same length as the axis to be replaced.
        :param old_axis_id: axis index (int) or name (str)
        :param new_axis: Series or Index object
        :return: new Cube object
        """
        new_axes = self._axes.replace(old_axis_id, new_axis)
        return Cube(self._values, new_axes)

    def swap_axes(self, axis_id1, axis_id2):
        """
        :param axis_id1:
        :param axis_id2:
        :return: new Cube object
        """
        index1 = self._axes.index(axis_id1)
        index2 = self._axes.index(axis_id2)
        new_axes = self._axes.swap(index1, index2)
        new_values = self._values.swapaxes(index1, index2)
        return Cube(new_values, new_axes)

    def insert_axis(self, axis, index=0):
        """
        Adds a new axis and repeat the values to fill the new cube.
        :param axis: the new axis to be inserted
        :param index: the index of the new axis
        :return: new Cube object
        """
        new_axes = self._axes.insert(axis, index)
        new_values = np.expand_dims(self._values, index)
        new_values = np.repeat(new_values, repeats=len(axis), axis=index)
        return Cube(new_values, new_axes)

    def align_axis(self, new_axis):
        """
        Return a cube with values aligned to a new axis.
        The order of the axes in the cube remains the same.
        The new axis will become one of the cube axes.
        :param new_axis: axis which the values should be aligned to
        :return: new Cube object
        """
        old_axis, old_axis_index = self._axes.axis_and_index(new_axis.name)
        indices = old_axis.index(new_axis.values)
        new_values = self._values.take(indices, old_axis_index)
        new_axes = self._axes.replace(old_axis_index, new_axis)
        return Cube(new_values, new_axes)

    def extend(self, axis, fill):
        old_axis, old_axis_index = self._axes.axis_and_index(axis)
        if not isinstance(old_axis, Index):
            old_axis_indices = old_axis.index(axis.values)

    def rename_axis(self, old_axis_id, new_axis_name):
        """
        Return a cube with renamed axis.
        :param old_axis_id: axis index (int) or name (str)
        :param new_axis_name: the name of the new axis (str)
        :return: new Cube object
        """
        old_axis = self._axes[old_axis_id]
        new_axis = old_axis.rename(new_axis_name)
        new_axes = self._axes.replace(old_axis_id, new_axis)
        return Cube(self._values, new_axes)

    def combine_axes(self, axis_names, new_axis_name, format):
        count = len(axis_names)
        axes = list()
        array_list = list()
        size = 1
        axis_indices = list()
        unique_axis_indices = set()
        for axis_name in axis_names:
            axis, axis_index = self._axes.axis_and_index(axis_name)
            unique_axis_indices.add(axis_index)
            axis_indices.append(axis_index)
            axes.append(axis)
            array_list.append(axis.values)
            size *= len(axis)

        if len(unique_axis_indices) != len(axis_names):
            raise ValueError("axis names are not unique")

        other_indices = list()
        new_axes = list()
        for i, a in enumerate(self._axes):
            if i not in axis_indices:
                if a.name == new_axis_name:
                    raise ValueError("axis name '{}' is not unique".format(new_axis_name))
                other_indices.append(i)
                new_axes.append(a)

        axis_indices.extend(other_indices)
        axis_sizes = [len(self.axes[i]) for i in other_indices]
        axis_sizes.insert(0, size)

        new_values = self._values.transpose(axis_indices)
        new_values = new_values.reshape(axis_sizes)

        new_axis_values = list()
        indices = np.zeros(count)

        for pos in range(size):
            current_values = [array[indices[k]] for k, array in enumerate(array_list)]
            new_axis_values.append(format.format(*current_values))

            # increment indices
            i = 0
            while True:
                indices[i] += 1
                if indices[i] < len(axes[i]):
                    break
                indices[i] = 0
                i += 1
                if i == count:
                    break

        new_axis = Index(new_axis_name, new_axis_values)
        new_axes.insert(0, new_axis)
        return Cube(new_values, new_axes)
        
    def filter(self, axis, values=None):
        """
        """
        if isinstance(axis, Axis):
            values = axis.values
            axis = axis.name
        # TODO ...

    def take(self, indices, axis_id):
        """Filter cube along given axis on specified indices. This is analogy to ndarray.take method.
        :param indices: a collection of ints or int
        :param axis_id: string or int specifying the axis
        """
        axis, axis_id = self._axes.axis_and_index(axis_id)
        new_axis = axis[indices]
        if isinstance(indices, int):
            # if indices is not collection,
            # then will remove one dimension
            axes = self._axes.remove(axis_id)
        else:
            # otherwise the dimension is preserved,
            # even if the collection has one element
            axes = self._axes.replace(axis_id, new_axis)
        values = self._values.take(indices, axis_id)
        return Cube(values, axes)

    def make_index(self, axis_id):
        """Convert an axis into Index. If the axis is already an Index, then does nothing."""
        self._axes.make_index(axis_id)

    def make_series(self, axis_id):
        """Convert an axis into Series. If the axis is already an Series, then does nothing."""
        self._axes.make_series(axis_id)
        
    @staticmethod
    def full(axes, fill_value, dtype=None):
        """Return a new cube filled with `fill_value`."""
        if not isinstance(axes, Axes):
            axes = Axes(axes)
        values = np.full(axes.shape, fill_value, dtype)
        return Cube(values, axes)
        
    @staticmethod
    def zeros(axes, dtype=float):
        """Return a new cube filled with zeros."""
        if not isinstance(axes, Axes):
            axes = Axes(axes)
        values = np.zeros(axes.shape, dtype)
        return Cube(values, axes)

    @staticmethod
    def ones(axes, dtype=float):
        """Return a new cube filled with ones."""
        if not isinstance(axes, Axes):
            axes = Axes(axes)
        values = np.ones(axes.shape, dtype)
        return Cube(values, axes)


def _broadcast_values(values, old_axes, new_axes):
    """
    Add new virtual axes (length is 1) to a numpy array to correspond to the new axes.
    """
    new_values = values
    transpose_indices = []
    for axis in new_axes:
        try:
            axis_index = old_axes.index(axis.name)
        except KeyError:
            # if axis is not present in the cube, add virtual axis at the end
            axis_index = new_values.ndim
            new_values = np.expand_dims(new_values, axis=axis_index)
        transpose_indices.append(axis_index)

    # handle the trailing axes
    if new_values.ndim != len(new_axes):
        raise ValueError("cube broadcasting axis mismatch")

    # transpose the result
    return new_values.transpose(transpose_indices)


def _align_axes(axis1, axis2, axis_index1, axis_index2, values1, values2):
    """

    :param axis1:
    :param axis2:
    :param axis_index1:
    :param axis_index2:
    :param values1:
    :param values2:
    :return: tuple (axis, values1, values2)
    """
    if isinstance(axis1, Index):
        if isinstance(axis2, Index):
            axis = axis1
            value_indices = _align_index_to_index(axis2, axis1)
            values2 = values2.take(value_indices, axis_index2)
        else:  # axis_b is Series
            # only in this case the new axis will be from cube b
            axis = axis2
            value_indices = _align_index_to_series(axis1, axis2)
            values1 = values1.take(value_indices, axis_index1)
    else:  # axis_a is Series
        if isinstance(axis2, Index):
            axis = axis1
            value_indices = _align_index_to_series(axis2, axis1)
            values2 = values2.take(value_indices, axis_index2)
        else:  # axis_b is Series
            axis = axis1
            _assert_align_series(axis2, axis1)
    return axis, values1, values2


def apply2(a, b, func, *args):
    """Apply function elementwise on values of two cubes.
    The cube axes are matched and aligned before the function is applied.
    :param a:
    :param b:
    :param func:
    :param args:
    :return:
    """

    if not isinstance(a, Cube):
        return Cube(func(a, b.values, *args), b.axes)

    if not isinstance(b, Cube):
        return Cube(func(a.values, b, *args), a.axes)

    values_a = a.values
    values_b = b.values
    all_axes = list()
    
    for axis_index_a, axis_a in enumerate(a.axes):
        
        try:
            axis_b, axis_index_b = b._axes.axis_and_index(axis_a.name)  # TODO - do not use private accessor
        except KeyError:
            # axis not found in cube b --> do not align
            axis_b = axis_a

        # if axes are identical or if axis_b has not been found --> do not align
        if axis_b is axis_a:
            all_axes.append(axis_a)
            continue

        axis, values_a, values_b = _align_axes(axis_a, axis_b, axis_index_a, axis_index_b, values_a, values_b)
        all_axes.append(axis)

    # add axes from b which have not been aligned
    for axis_b in b.axes:
        if not a.has_axis(axis_b.name):
            all_axes.append(axis_b)
                
    values_a = _broadcast_values(values_a, a._axes, all_axes)  # TODO - do not use private accessor
    values_b = _broadcast_values(values_b, b._axes, all_axes)  # TODO - do not use private accessor

    return Cube(func(values_a, values_b, *args), all_axes)
    
    
def _align_index_to_index(axis_from, axis_to):
    """
    """
    if len(axis_from) != len(axis_to):
        raise AxisAlignError("cannot align two Index axes - axes '{}' have different lengths".format(axis_to.name))
        
    try:
        return axis_from.index(axis_to.values)
        
    except KeyError:
        raise AxisAlignError("cannot align two Index axes - axes '{}' have different values".format(axis_to.name))
        
        
def _align_index_to_series(axis_from, axis_to):
    """
    """
    try:
        return axis_from.index(axis_to.values)
        
    except KeyError:
        raise AxisAlignError("cannot align Index to Series - axes '{}' have different values".format(axis_to.name))   

        
def _assert_align_series(axis_from, axis_to):
    """
    Series can be aligned to another axis if and only if it has the same values, in the same order.
    No alignment indices are returned, only the equality of axes is checked.
    """
    if not np.array_equal(axis_from.values, axis_to.values):
        raise AxisAlignError("cannot align Series - axes '{}' have different values".format(axis_to.name))


def _unique_axes_from_cubes(cube_list):
    unique_axes_list = list()
    unique_axes_dict = dict()

    # create a list of unique axes - except the main axis
    for cube in cube_list:
        for axis in cube.axes:
            try:
                base_axis_index = unique_axes_dict[axis.name]
            except KeyError:
                unique_axes_dict[axis.name] = len(unique_axes_list)
                unique_axes_list.append(axis)
                continue
            base_axis = unique_axes_list[base_axis_index]

            # Series has priority over Index
            if isinstance(base_axis, Index) and isinstance(axis, Series):
                unique_axes_list[base_axis_index] = axis

    return unique_axes_list


def _align_broadcast_and_concatenate(cube_list, axis_list, main_axis):
    array_list = [cube.values for cube in cube_list]

    for base_axis in axis_list:
        for cube_index, cube in enumerate(cube_list):
            try:
                axis_index = cube.axis_index(base_axis.name)
            except KeyError:
                continue
            axis = cube.axes[axis_index]

            if axis is base_axis:
                # axes are identical, no need to align
                continue

            if isinstance(axis, Index):
                if isinstance(base_axis, Index):
                    value_indices = _align_index_to_index(axis, base_axis)
                elif isinstance(base_axis, Series):
                    value_indices = _align_index_to_series(axis, base_axis)
                else:
                    raise TypeError("unsupported axis type")
            elif isinstance(axis, Series):
                # series must have the same values
                # if they do not, they cannot be aligned
                # if they do, they do not need to be aligned
                _assert_align_series(axis, base_axis)
                continue
            else:
                raise TypeError("unsupported axis type")

            array = array_list[cube_index]
            array_list[cube_index] = array.take(value_indices, axis_index)

    # put the new main axis in front of the list
    axis_list.insert(0, main_axis)

    # broadcast value arrays
    for cube_index, cube in enumerate(cube_list):
        array = array_list[cube_index]
        array = _broadcast_values(array, cube._axes, axis_list)  # TODO - do not use private accessor
        array_list[cube_index] = array

    array_list = np.broadcast_arrays(*array_list)
    new_values = np.concatenate(array_list)
    return Cube(new_values, axis_list)


def concatenate(cubes, axis_name, as_index=True):
    """
    :param cubes:
    :param axis_name:
    :param as_index:
    :return:
    """

    main_axis_values_list = list()
    for cube in cubes:
        try:
            axis = cube.axis(axis_name)
        except KeyError:
            raise ValueError("cube does not contain axis '{}'".format(axis_name))
        main_axis_values_list.append(axis.values)

    # concatenate the new main axis
    main_axis_values = np.concatenate(main_axis_values_list)
    if as_index:
        # will fail if does not have unique values
        main_axis = Index(axis_name, main_axis_values)
    else:
        main_axis = Series(axis_name, main_axis_values)

    unique_axes_list = _unique_axes_from_cubes(cubes)

    # create a unique list without the main axis
    unique_axes_list = [a for a in unique_axes_list if a.name != axis_name]

    return _align_broadcast_and_concatenate(cubes, unique_axes_list, main_axis)


def join(cubes, axis):
    """
    Note: join adds a new dimension, unlike concatenate which concatenates along axis which already exists in the cubes
    :param cubes:
    :param axis:
    :return:
    """

    for cube in cubes:
        if cube.has_axis(axis.name):
            raise ValueError("cube already contains axis '{}'".format(axis.name))

    if len(cubes) != len(axis):
        raise ValueError("invalid axis length")

    unique_axes_list = _unique_axes_from_cubes(cubes)

    return _align_broadcast_and_concatenate(cubes, unique_axes_list, axis)
