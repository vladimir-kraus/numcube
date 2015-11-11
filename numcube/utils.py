from numcube.axes import Axes
from numcube.axis import Axis
from numcube.index import Index


def is_axis_indexed(axis):
    return isinstance(axis, Index)


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
    if is_axis_indexed(axis):
        # we intentionally do not pass axis.values because
        # indexed axis provides 'in' operator
        return filter_cube_by_values(cube, axis.name, axis)
    else:
        # else we provide raw values, but then the lookup is slower
        return filter_cube_by_values(cube, axis.name, axis.values)


def filter_cube_by_values(cube, axis_id, values):
    """Returns a cube filtered by specified values on a given axis. Takes into account only values
    which exist on the axis. Other values are ignored.
    :param axis: axis index (int) or name (str)
    :param values: a collection of values providing 'in' operator
    :return: new Cube instance
    """
    axis, axis_index = axis_and_index(cube, axis_id)
    value_indices = [i for i, v in enumerate(axis.values) if v in values]
    return cube.take(value_indices, axis_index)

