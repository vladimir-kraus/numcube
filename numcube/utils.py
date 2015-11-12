"""Utility functions imported by numcube.cube and numcube.axes.
Do not import these in order to avoid circular imports.
"""

import numcube.axis
import numcube.index


def is_axis(obj):
    return isinstance(obj, numcube.axis.Axis)


def is_index(obj):
    return isinstance(obj, numcube.index.Index)


def is_series(obj):
    return is_axis(obj) and not is_index(obj)


def is_axis_indexed(axis):
    return is_index(axis)


def make_axis_collection(axes):
    """Creates a list of axes if a single axis is passed in."""
    if isinstance(axes, int) or isinstance(axes, str) or is_axis(axes):
        return [axes]
    else:
        return axes


