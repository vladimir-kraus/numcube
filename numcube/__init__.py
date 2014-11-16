from .cube import Cube, concatenate, join
from .index import Index
from .series import Series
from .axes import Axes


#from numcube.table import Table, Axis
"""

def cube_get_axes(cube, axes):
    axes_list = list()
    for axis in axes:
        axes_list.append(cube.axes[axis])


def cube_to_table(cube, row_axes=None, col_axes=None):
    cube_axes = cube.axes

    if row_axes is None:
        pass
        """