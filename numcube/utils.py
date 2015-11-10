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

