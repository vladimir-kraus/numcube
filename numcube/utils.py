from numcube.axes import Axes
from numcube.axis import Axis


def make_axes(axes):
    """Creates an Axes object from a collection of axes."""
    if not isinstance(axes, Axes):
        return Axes(axes)
    else:
        return axes


def make_axis_collection(axes):
    """Creates a list of axes if a single axis is passed in."""
    if isinstance(axes, int) or isinstance(axes, str) or isinstance(axes, Axis):
        return [axes]
    else:
        return axes


def axis_and_index(cube, axis_id):
    return cube._axes.axis_and_index(axis_id)


def filter_cube_by_axes(cube, axes):
    for axis in axes:
        cube = cube.filter_by_axis(axis)
    return cube


def filter_cube_by_axis(cube, axis):
    return filter_cube_by_values(cube, axis.name, axis.values)


def filter_cube_by_values(cube, axis_id, values):
    """Returns a cube filtered by specified values on a given axis. Takes into account only values
    which exist on the axis. Other values are ignored.
    :param axis: axis index (int)
    :param values: a collection of values to be filtered (included)
    :return: new Cube instance
    The performance is dependent on the size of 'values' or 'exclude' collections. If a large collection
    is being filtered, it is beneficial to convert it to a set because it provides faster lookups.
    """
    axis, axis_index = axis_and_index(cube, axis_id)
    value_indices = [i for i, v in enumerate(axis.values) if v in values]
    return cube.take(value_indices, axis_index)

