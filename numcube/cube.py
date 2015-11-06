import numpy as np
import math

from numcube.axes import Axis
from numcube.exceptions import AxisAlignError
from numcube.index import Index
from numcube.utils import make_axis_collection, make_Axes


class Cube(object):
    """Wrapper around numpy.ndarray with named and labelled axes. The API aims to be as similar to ndarray API as
    possible. Moreover it allows automatic axis matching and alignment in operations among cubes.
    """

    # when numpy array is the first argument in operation and Cube is the second,
    # then __array_priority__ will force Cube to handle the operation rather than numpy array
    __array_priority__ = 10

    def __init__(self, values, axes, dtype=None):
        values = np.asarray(values, dtype)

        axes = make_Axes(axes)

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
        :return: new Cube instance
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

    def __repr__(self):
        """Returns a textual representation of the object.
        :return: str
        """
        return "Cube({}, {})".format(repr(self._values), repr(tuple(self.axis_names)))

    @property
    def shape(self):
        """Returns the lengths of dimensions of the underlying numpy.ndarray.
        :return: tuple of ints
        """
        return self._values.shape

    @property
    def size(self):
        """Returns the number of elements in the underlying numpy.ndarray.
        :return: int
        """
        return self._values.size

    @property
    def ndim(self):
        """Returns the number of array dimensions.
        :return: int
        """
        return self._values.ndim

    @property
    def values(self):
        return self._values  # TODO: .view()?

    @property
    def axes(self):
        """Returns cube axes as iterator, which can be converted e.g. to tuple or list.
        :return: iterator
        Examples:
        - for ax in cube.axes: print(ax.name)
        - tuple(cube.axes)
        - list(cube.axes)
        """
        for axis in self._axes:
            yield axis

    @property
    def axis_names(self):
        """Returns axis names as iterator, which can be converted e.g. to tuple or list.
        To get the name of a specific axis, it is preferred to use cube.axis(index).name
        rather than for example tuple(cube.axi_names)[index].
        Examples:
        - for name in cube.axis_names: print(name)
        - tuple(cube.axis_names)
        - list(cube.axis_names)
        """
        for axis in self.axes:
            yield axis.name

    def axis(self, axis):
        """Returns axis by the name or by the index.
        Index can be a negative number, in that case, the axes are counted backwards from the last one.
        :param axis: axis name (str), axis index (int) or Axis instance
        :return: Axis instance
        :raise LookupError: if the axis does not exist, TypeError if wrong argument type is passed
        """
        return self._axes[self.axis_index(axis)]

    def axis_index(self, axis):
        """Returns the index of the axis specified by its name or axis instance.
        :param axis: name (str), index (int) or Axis instance
        :return: int
        :raise LookupError: if the axis does not exist, TypeError if wrong argument type is passed
        """
        return self._axes.index(axis)

    def has_axis(self, axis):
        """Returns True/False indicating whether the axis exists in the Cube.
        :param axis: name (str), index (int) or Axis instance
        :return: bool
        :raise TypeError: if wrong argument type is passed
        """
        return self._axes.contains(axis)

    def apply(self, func, *args):
        """Applies a function to all values and return the new cube.
        :param func: function to be applied to values
        :param args: additional optional arguments of func
        :return: new Cube instance
        """
        return Cube(func(self._values, *args), self._axes)

    def transpose(self, front=[], back=[]):
        """A generalized analogy to numpy.transpose.
        :param front: axes which will be in the front of other axes
        :param back: axes which will be at the back of other axes
        :return: new Cube instance with transposed axes

        The arguments 'front' and 'back' are expected in the form of an axis identifier or a collection
        of axis identifiers. Axis identifier is a name (str), index (int) or Axis instance.
        """
        indices = self._axes.transposed_indices(front, back)
        new_axes = tuple(self._axes.axis_by_index(index) for index in indices)
        new_values = self._values.transpose(indices)
        return Cube(new_values, new_axes)
        
    """arithmetic operators"""

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
        """Returns bit-wise inversion, or bit-wise NOT, element-wise."""
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
        """Implements behaviour for the built in abs() function.
        :return: new Cube instance
        """
        return self.apply(abs)

    def __round__(self, decimals):
        """Implements behaviour for the built in round() function.
        :param decimals: the number of decimal places to round to
        :return: new Cube instance
        """
        return self.apply(round, decimals)

    def __floor__(self):
        """Implements behaviour for math.floor(), i.e., rounding down to the nearest integer.
        :return: new Cube instance
        """
        return self.apply(math.floor)

    def __ceil__(self):
        """Implements behaviour for math.ceil(), i.e., rounding up to the nearest integer.
        :return: new Cube instance
        """
        return self.apply(math.ceil)

    def __trunc__(self):
        """Implements behavior for math.trunc(), i.e., truncating to an integral.
        :return: new Cube instance
        """
        return self.apply(math.trunc)

    def sin(self):
        """Sine, element-wise. Can be called as numpy.sin(C) or C.sin().
        :return: new Cube instance
        """
        return self.apply(np.sin)

    def cos(self):
        """Cosine, element-wise. Can be called as numpy.cos(C) or C.cos().
        :return: new Cube instance
        """
        return self.apply(np.cos)

    def tan(self):
        """Tangents, element-wise. Can be called as numpy.tan(C) or C.tan().
        :return: new Cube instance
        """
        return self.apply(np.tan)

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
        """Returns the arithmetic mean along the specified axis."""
        return self.aggregate(np.mean, axis, keep)

    def min(self, axis=None, keep=None):
        """Returns the minimum of a cube or minimum along an axis."""
        return self.aggregate(np.min, axis, keep)

    def max(self, axis=None, keep=None):
        """Returns the maximum of a cube or maximum along an axis."""
        return self.aggregate(np.max, axis, keep)

    def all(self, axis=None, keep=None):
        """Tests whether all cube elements along a given axis evaluate to True."""
        return self.aggregate(np.all, axis, keep)

    def any(self, axis=None, keep=None):
        """Tests whether any cube element along a given axis evaluates to True."""
        return self.aggregate(np.any, axis, keep)

    def aggregate(self, func, axis=None, keep=None):
        """Aggregation of values in the cube along one or more axes. This function works
        in two different modes. Either the axes to be eliminated are specified. Or the axes
        to be kept are specified, while the other axes are elimitated.

        :param axis: axis or axes to be eliminated by the aggregation
        :param keep: axis or axes which are kept after the aggregation

        Only one of 'axis' and 'keep' arguments can be non-None, otherwise ValueError is raised.
        If the both arguments are None, then the Cube is aggregated to a single scalar value.

        Example:
        # returns sum of all months, i.e. month axis is eliminated; other axes are kept
        cube.aggregate(np.sum, "month")

        # returns mean for each month, i.e. month axis is kept; other axes are eliminated
        cube.aggregate(np.mean, keep="month")
        """

        # complete aggregation to a scalar
        if axis is None and keep is None:
            return func(self._values)

        if keep is not None and axis is not None:
            raise ValueError("either 'keep' or 'axis' argument must be None")

        axis = make_axis_collection(axis)
        keep = make_axis_collection(keep)

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

    def group(self, axis, func, sorted=True, *args):  # **kwargs): # since numpy 1.9
        """Group the same values along a given axis by applying a function.
        :param axis: name (str) or index (int) of axis to group the cube values by
        :param func: aggregation function, e.g. np.sum, np.mean etc.
            There are the following requirements:
            - the function takes two fixed arguments - array and axis (given by index)
            - these two fixed arguments can be followed by a variable number of other arguments passed in *args
            - the function must return an array with one axis less then the input array
        :param sorted: True to sort the grouped values, False to keep the order of the first occurrences
        :return: new Cube instance
        """
        old_axis, old_axis_index = self._axes.axis_and_index(axis)
        
        sub_cubes = list()
        
        if sorted:
            # np.unique sorts the returned values by default
            unique_values = np.unique(old_axis.values)
        else:
            # special handling is required if the first occurrence order is to be kept
            unique_values, unique_indices = np.unique(old_axis.values, return_index=True)
            index_array = np.argsort(unique_indices)
            unique_values = unique_values[index_array]
        
        old_values = old_axis.values
        all_indices = np.arange(len(old_values))
        for value in unique_values:
            indices = all_indices[old_values == value]
            sub_cube = self._values.take(indices, old_axis_index)
            sub_cube = np.apply_along_axis(func, old_axis_index, sub_cube, *args)  # , **kwargs) # since numpy 1.9
            sub_cube = np.expand_dims(sub_cube, old_axis_index)
            sub_cubes.append(sub_cube)
        
        # the created axis is Index because it has unique values
        new_axis = Index(old_axis.name, unique_values)
        new_axes = self._axes.replace(old_axis_index, new_axis)
        new_values = np.concatenate(sub_cubes, old_axis_index)
        return Cube(new_values, new_axes)

    def replace_axis(self, old_axis_id, new_axis):
        """Replaces an existing axis with a new axis and return the new Cube instance.
        The new axes collection is checked for duplicate names.
        The new axis must have the same length as the axis to be replaced.
        :param old_axis_id: axis index (int) or name (str)
        :param new_axis: Series or Index instance
        :return: new Cube instance
        """
        new_axes = self._axes.replace(old_axis_id, new_axis)
        return Cube(self._values, new_axes)

    def swap_axes(self, axis1, axis2):
        """Swaps two axes.
        :param axis1: name (str), index (int) or Axis instance
        :param axis2: name (str), index (int) or Axis instance
        :return: new Cube instance with swapped axes
        :raise LookupError: if axis1 or axis2 is not found

        If axis1 is the same as axis2, the original Cube instance is returned.
        """
        index1 = self._axes.index(axis1)
        index2 = self._axes.index(axis2)
        if index1 == index2:
            return self
        new_axes = self._axes.swap(index1, index2)
        new_values = self._values.swapaxes(index1, index2)
        return Cube(new_values, new_axes)

    def insert_axis(self, axis, index=0):
        """Adds a new axis and repeats the values to fill the new cube.
        :param axis: the new axis to be inserted
        :param index: the index of the new axis after it is inserted
        :return: new Cube instance with inserted axis
        :raise: TODO
        """
        new_axes = self._axes.insert(axis, index)
        new_values = np.expand_dims(self._values, index)
        new_values = np.repeat(new_values, repeats=len(axis), axis=index)
        return Cube(new_values, new_axes)

    def align_axis(self, new_axis):
        """Returns a cube with values aligned to a new axis. The axis to be aligned has the same name as the new
        axis. The order of the axes in the cube remains the same. The new axis will become one of the cube axes.
        :param new_axis: Axis instance
        :return: new Cube instance
        """
        old_axis, old_axis_index = self._axes.axis_and_index(new_axis.name)
        indices = old_axis.indexof(new_axis.values)
        new_values = self._values.take(indices, old_axis_index)
        new_axes = self._axes.replace(old_axis_index, new_axis)
        return Cube(new_values, new_axes)

    def extend(self, axis, fill):
        old_axis, old_axis_index = self._axes.axis_and_index(axis)
        if not isinstance(old_axis, Index):
            old_axis_indices = old_axis.index(axis.values)
        # TODO...

    def rename_axis(self, old_axis, new_name):
        """Returns a cube with a renamed axis.
        :param old_axis: axis index (int), name (str) or Axis instance
        :param new_name: the name of the new axis (str)
        :return: new Cube instance
        :raise LookupError: if the old axis does not exist, ValueError is the name is duplicate
        """
        new_axes = self._axes.rename(old_axis, new_name)
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
        axis_sizes = [len(self.axis(i)) for i in other_indices]
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
        
    def filter(self, axis, values=None, exclude=None):
        """Returns a cube filtered by specified values on a given axis. Takes into account only values
        which exist on the axis. Other values are ignored.
        :param axis: name (str), index (int) or Axis instance
        :param values: a collection of values to be filtered (included)
        :param exclude: a collection of values to be filtered out (excluded)
        :return: new Cube instance
        The performance is dependent on the size of 'values' or 'exclude' collections. If a large collection
        is being filtered, it is beneficial to convert it to a set because it provides faster lookups.
        """
        # TODO - consider switching argument order to be consistent with take and compress
        axis, axis_index = self._axes.axis_and_index(axis)
        if values is not None and exclude is None:
            value_indices = [i for i, v in enumerate(axis.values) if v in values]
        elif values is None and exclude is not None:
            value_indices = [i for i, v in enumerate(axis.values) if v not in exclude]
        else:
            raise ValueError("either 'values' or 'exclude' must be non-None, the other must be None")
        return self.take(value_indices, axis_index)

    def take(self, indices, axis):
        """Filters the cube along an axis using specified indices. 
        Analogy to numpy.ndarray.take.
        :param indices: a collection of ints or int
        :param axis: axis name (str), axis index (int) or Axis instance
        :return: new Cube instance
        :raise LookupError: is the axis does not exist, ValueError for invalid indices

        If 'indices' is a single int, then the axis is removed from the cube.
        If 'indices' is a collection of ints, then the axis is preserved.
        """
        axis, axis_index = self._axes.axis_and_index(axis)
        new_axis = axis.take(indices)
        if isinstance(indices, int):
            # if indices is a single int,
            # then will remove one dimension
            axes = self._axes.remove(axis_index)
        else:
            # otherwise the dimension is preserved,
            # even if the collection has one element
            axes = self._axes.replace(axis_index, new_axis)
        values = self._values.take(indices, axis_index)
        return Cube(values, axes)

    def compress(self, condition, axis):
        """Filters the cube along an axis using a boolean mask along a specified axis. 
        Analogy to numpy.ndarray.compress.
        :param condition: collection of boolean values
        :param axis: axis name (str), axis index (int) or Axis instance
        :return: new Cube instance
        :raise LookupError: is the axis does not exist, # TODO - error if wrong type
        """
        axis, axis_index = self._axes.axis_and_index(axis)
        new_axis = axis.compress(condition)
        axes = self._axes.replace(axis_index, new_axis)
        values = self._values.compress(condition, axis_index)
        return Cube(values, axes)

    def make_index(self, axis):
        """Converts an axis into Index. If the axis is already an Index, then does nothing."""
        self._axes.make_index(axis)

    def make_series(self, axis):
        """Converts an axis into Series. If the axis is already a Series, then does nothing."""
        self._axes.make_series(axis)
        
    def squeeze(self):
        """Removes all the axes with the size of one from the cube. 
        Analogy to numpy ndarray.squeeze().
        :return: new Cube instance"""
        new_axes = tuple(a for a in self.axes if len(a) != 1)
        new_values = self._values.squeeze()
        return Cube(new_values, new_axes)        
        
    @staticmethod
    def full(axes, fill_value, dtype=None):
        """Returns a new cube filled with a uniform value.
        :param axes: a collection of Axis instances for form the new cube
        :param fill_value: the uniform value to fill the cube
        :param dtype: the value type of the new cube (usually int or float)
        :returns: new Cube instance
        """
        axes = make_Axes(axes)
        shape = tuple(len(axis) for axis in axes)
        values = np.full(shape, fill_value, dtype)
        return Cube(values, axes)
        
    @staticmethod
    def zeros(axes, dtype=float):
        """Returns a new cube filled with zeros.
        :param axes: a collection of Axis instances for form the new cube
        :param dtype: the value type of the new cube (usually int or float)
        :returns: new Cube instance
        """
        axes = make_Axes(axes)
        shape = tuple(len(axis) for axis in axes)
        values = np.zeros(shape, dtype)
        return Cube(values, axes)

    @staticmethod
    def ones(axes, dtype=float):
        """Returns a new cube filled with ones.
        :param axes: a collection of Axis instances for form the new cube
        :param dtype: the value type of the new cube (usually int or float)
        :returns: new Cube instance
        """
        axes = make_Axes(axes)
        shape = tuple(len(axis) for axis in axes)
        values = np.ones(shape, dtype)
        return Cube(values, axes)


def apply2(a, b, func, *args):
    """Apply function element-wise on values of two cubes.
    The cube axes are matched and aligned before the function is applied.
    :param a: Cube instance
    :param b: Cube instance
    :param func: function to be applied
    :param args: additional arguments which are passed to the function
    :return: new Cube instance
    """

    if not isinstance(a, Cube):
        return Cube(func(a, b.values, *args), tuple(b.axes))

    if not isinstance(b, Cube):
        return Cube(func(a.values, b, *args), tuple(a.axes))

    values_a = a.values
    values_b = b.values
    all_axes = list()

    for axis_index_a, axis_a in enumerate(a.axes):

        try:
            axis_b, axis_index_b = b._axes.axis_and_index(axis_a.name)
        except LookupError:
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

    values_a = _broadcast_values(values_a, a._axes, all_axes)
    values_b = _broadcast_values(values_b, b._axes, all_axes)

    return Cube(func(values_a, values_b, *args), all_axes)


def concatenate(cubes, axis_name, as_index=False, broadcast=False):
    """Joins cubes along one axis on which the cubes have non-overlapping values.
    :param cubes: a collection of Cube instances
    :param axis_name: the name of axis on which the cubes will be joined
    :param as_index: if True, the new joined axis will be created as Index; otherwise it will be Axis
    :param broadcast: allows automatic broadcasting of unique axes
    :return: new Cube instance
    :raise LookupError: if any cube does not contain the joined axis
    :raise ValueError: if Index instance shall be created but the values are not unique

    The joined axis becomes the first axis of the new cube regardless of its position in the original cubes.
    """

    main_axis_values_list = list()
    for cube in cubes:
        axis = cube.axis(axis_name)
        main_axis_values_list.append(axis.values)

    # concatenate the new main axis
    main_axis_values = np.concatenate(main_axis_values_list)
    if as_index:
        # will fail if does not have unique values
        main_axis = Index(axis_name, main_axis_values)
    else:
        main_axis = Axis(axis_name, main_axis_values)

    unique_axes_list = _unique_axes_from_cubes(cubes)

    # create a unique list without the main axis
    unique_axes_list = [a for a in unique_axes_list if a.name != axis_name]

    return _align_broadcast_and_concatenate(cubes, unique_axes_list, main_axis, broadcast)


def stack(cubes, axis, broadcast=False):
    """Adds a new dimension and stack uniformly shaped cubes along this axis.
    This is different from concatenate which joins cubes along axis which already exists in all the cubes.
    :param cubes: a collection of Cube instances
    :param axis: Axis instance which is used to stack the cubes
    :param broadcast: allows automatic broadcasting of unique axes
    :return: new Cube instance with the new axis
    :raise ValueError: is an axis of the same axis name already exists in any of the cubes in the collection;
        ValueError if the axis has different length from the number of cubes in the collection
    """
    for cube in cubes:
        if cube.has_axis(axis.name):
            raise ValueError("cube already contains axis '{}'".format(axis.name))

    if len(cubes) != len(axis):
        raise ValueError("invalid axis length")

    unique_axes_list = _unique_axes_from_cubes(cubes)

    return _align_broadcast_and_concatenate(cubes, unique_axes_list, axis, broadcast)


def _broadcast_values(values, old_axes, new_axes):
    """Add new virtual axes (length is 1) to a numpy array to correspond to the new axes."""
    new_values = values
    transpose_indices = []
    for axis in new_axes:
        try:
            axis_index = old_axes.index(axis.name)
        except LookupError:
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


def _align_index_to_index(axis_from, axis_to):
    """
    """
    if len(axis_from) != len(axis_to):
        raise AxisAlignError("cannot align two Index axes - axes '{}' have different lengths".format(axis_to.name))

    try:
        return axis_from.indexof(axis_to.values)
    except KeyError:
        raise AxisAlignError("cannot align two Index axes - axes '{}' have different values".format(axis_to.name))


def _align_index_to_series(axis_from, axis_to):
    """
    """
    try:
        return axis_from.indexof(axis_to.values)
    except KeyError:
        raise AxisAlignError("cannot align Index to Series - axes '{}' have different values".format(axis_to.name))


def _assert_align_series(axis_from, axis_to):
    """Series can be aligned to another axis if and only if it has the same values, in the same order.
    No alignment indices are returned, only the equality of axes is checked.
    """
    if not np.array_equal(axis_from.values, axis_to.values):
        raise AxisAlignError("cannot align Series - axes '{}' have different values".format(axis_to.name))


def _is_indexable(axis):
    """Return whether the axis can be indexed."""
    return hasattr(axis, "indexof")


def _unique_axes_from_cubes(cubes):
    """Creates common axis space for a collection of cubes. The following rules are observed:
    1) axes are identified only by their name, values are ignored
    2) axes are listed as they are listed in the cube collection
    3) non-indexable axes have priority over the indexable with the same name
    4) axes are not aligned by values, values are ignored
    """
    unique_axes_list = list()
    unique_axes_dict = dict()

    for cube in cubes:
        for axis in cube.axes:
            try:
                base_axis_index = unique_axes_dict[axis.name]
            except KeyError:
                # add new axis
                unique_axes_dict[axis.name] = len(unique_axes_list)
                unique_axes_list.append(axis)
            else:
                # axis with the same name was found
                base_axis = unique_axes_list[base_axis_index]
                if _is_indexable(base_axis) and not _is_indexable(axis):
                    # replace indexable with non-indexable
                    unique_axes_list[base_axis_index] = axis

    return unique_axes_list


def _align_broadcast_and_concatenate(cube_list, axis_list, main_axis, broadcast):
    array_list = [cube.values for cube in cube_list]

    for base_axis in axis_list:
        for cube_index, cube in enumerate(cube_list):
            try:
                axis_index = cube.axis_index(base_axis.name)
            except LookupError:
                if broadcast:
                    continue
                else:
                    raise
            axis = cube.axis(axis_index)

            if axis is base_axis:
                # axes are identical, no need to align
                continue

            if isinstance(axis, Index):
                if isinstance(base_axis, Index):
                    value_indices = _align_index_to_index(axis, base_axis)
                elif isinstance(base_axis, Axis):
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
        array = _broadcast_values(array, cube._axes, axis_list)
        array_list[cube_index] = array

    array_list = np.broadcast_arrays(*array_list)
    new_values = np.concatenate(array_list)
    return Cube(new_values, axis_list)
