import numpy as np
from .index import Index
from .series import Series
from .axes import Axes
from .exceptions import AxisAlignError


class Cube:
    """
    Wrapper around numpy.ndarray with named axes.
    Axes in numpy.ndarray are identified by their index. Cube allows any hashable value to identify the axes.
    """
    __array_priority__ = 20

    def __init__(self, values, axes):
        """
        'axes' must be a sequence with the number of values equal to the number of dimensions of 'values'.
        If 'axes' is empty, then sequence 0,1,2... is generated.
        """
        values = np.array(values)  # copy, view?

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
        
    def __str__(self):
        return "axes:\n{}\nvalues:\n{}".format(self._axes, self._values)

    @property
    def ndim(self):
        """
        :return: the number of axes
        """
        return self._values.ndim

    @property
    def values(self):
        return self._values.view()

    @property
    def axes(self):
        return self._axes

    def transpose(self, axis_ids):
        """
        Analogy to numpy.transpose.
        :param axis_ids: axis names or indices defining the new order of axes
        :return: new Cube object
        """
        if len(axis_ids) != self.ndim:
            raise ValueError("invalid number of axes")

        indices = np.array(self._indices(axis_ids))
        new_axes = self._axes.subset(axis_ids)
        new_values = self._values.transpose(indices)
        return Cube(new_values, new_axes)
        
    def _indices(self, axes):
        return [self._axes.index(axis) if isinstance(axis, str) else axis for axis in axes]

    def __mul__(self, other):
        return _eval_operation(self, other, np.multiply)

    def __rmul__(self, other):
        return _eval_operation(other, self, np.multiply)
        
    def __eq__(self, other):
        return _eval_operation(self, other, np.equal)

    def __ne__(self, other):
        return _eval_operation(self, other, np.not_equal)

    def __lt__(self, other):
        return _eval_operation(self, other, np.less)

    def __le__(self, other):
        return _eval_operation(self, other, np.less_equal)

    def __gt__(self, other):
        return _eval_operation(self, other, np.greater)

    def __ge__(self, other):
        return _eval_operation(self, other, np.greater_equal)
        
    def _get_axis_and_index(self, axis):
        if isinstance(axis, str):
            axis = self.axes.index(axis)
        return self.axes[axis], axis
        
    def groupby(self, axis, func, sorted=True, *args):  # **kwargs): # since numpy 1.9
        """
        func 
            - a function which takes two fixed arguments - array and axis (in this order) 
            - following these two can also take a variable number of other arguments passed in *args
            - must return array with one axes less then the input array
            - examples are np.sum, np.mean, etc.
        """
        old_axis, old_axis_index = self._get_axis_and_index(axis)
        
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

    def replace_axis(self, old_axis_id, new_axis):
        """
        Replace an existing axis with a new axis and return the new Cube object.
        The new axes collection is checked for duplicit names.
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

    def align_axis(self, new_axis):
        """
        Return a cube with values aligned to a new axis.
        :param new_axis: the name of the new axis (str)
        :return: new Cube object
        """
        old_axis_index = self._axes.index(new_axis.name)
        old_axis = self._axes[old_axis_index]
        indices = old_axis.index(new_axis.values)
        new_values = self._values.take(indices, old_axis_index)
        new_axes = self._axes.replace(old_axis_index, new_axis)
        return Cube(new_values, new_axes)

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
            axis_index = self._axes.index(axis_name)
            unique_axis_indices.add(axis_index)
            axis_indices.append(axis_index)
            axis = self._axes[axis_index]
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


def _eval_operation(a, b, func):

    if not isinstance(a, Cube):
        return Cube(func(a, b.values), b.axes)

    if not isinstance(b, Cube):
        return Cube(func(a.values, b), a.axes)

    values_a = a.values
    values_b = b.values
    all_axes = list(a.axes)
    
    for axis_index_a, axis_a in enumerate(a.axes):
        
        try:
            axis_index_b = b.axes.index(axis_a.name)
        except KeyError:  
            # axis not found in cube b --> do not align
            continue
        axis_b = b.axes[axis_index_b]
        
        # if axes are identical --> do not align
        if axis_a is axis_b:
            continue
        
        if isinstance(axis_a, Index):
            if isinstance(axis_b, Index):
                value_indices = _align_index_to_index(axis_b, axis_a)
                values_b = values_b.take(value_indices, axis_index_b)
            else:  # axis_b is Series
                # only in this case the new axis will be from cube b
                all_axes[axis_index_a] = axis_b
                value_indices = _align_index_to_series(axis_a, axis_b)
                values_a = values_a.take(value_indices, axis_index_a)
        else:  # axis_a is Series
            if isinstance(axis_b, Index):
                value_indices = _align_index_to_series(axis_b, axis_a)
                values_b = values_b.take(value_indices, axis_index_b)
            else:  # axis_b is Series
                _assert_align_series(axis_b, _axis_a)

    # add axes from b which have not been aligned
    for axis_b in b.axes:
        if not a.axes.contains(axis_b.name):
            all_axes.append(axis_b)
                
    values_a = _broadcast_values(values_a, a.axes, all_axes)
    values_b = _broadcast_values(values_b, b.axes, all_axes)

    return Cube(func(values_a, values_b), all_axes)
    
    
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
                axis_index = cube.axes.index(base_axis.name)
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
        array = _broadcast_values(array, cube.axes, axis_list)
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
            axis = cube.axes[axis_name]
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
        if cube.axes.contains(axis.name):
            raise ValueError("cube already contains axis '{}'".format(axis.name))

    if len(cubes) != len(axis):
        raise ValueError("invalid axis length")

    unique_axes_list = _unique_axes_from_cubes(cubes)

    return _align_broadcast_and_concatenate(cubes, unique_axes_list, axis)
