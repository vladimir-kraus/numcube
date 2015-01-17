import numpy as np
from numcube import Axis, Series


class Header(object):
    def __init__(self, series, format=None):
        self._series = tuple(series)
        self._size = len(series[0])  # TODO: what if there are no series
        for s in series:
            if self._size != len(s):
                raise ValueError('all header series must have equal lengths')
        self._format = format

    def __repr__(self):
        lst = [repr(s) for s in self._series]
        return "\n".join(lst)

    def __len__(self):
        return self._size

    def name(self, field):
        return self._series[field].name
        
    @property
    def nseries(self):
        return len(self._series)

    @property
    def series(self):
        return self._series

    def label(self, index):
        lst = [str(s.values[index]) for s in self._series]
        if self._format:
            return self._format.format(*lst)
        else:
            return " ".join(lst)
            
    def value(self, series, index):
        return self._series[series].values[index]

    @staticmethod
    def from_axes(axes, format=None):
        """
        :param axes: Axis object or a collection of Axis objects
        """

        if isinstance(axes, Axis):
            axes = [axes]
        n = 1
        rep_all = []
        for a in axes:
            rep_all.append(n)
            n *= len(a)

        rep_each = []
        for a in axes:
            n //= len(a)
            rep_each.append(n)

        series = []
        for a, r1, r2 in zip(axes, rep_all, rep_each):
            repeated__values = np.tile(np.repeat(a.values, r2), r1)
            series.append(Series(a.name, repeated__values))

        return Header(series, format)


class Table:
    """
    Table is a 2D array of data with row and column headers.
    Each header can contain multiple series of values.

    Differences from Cube:
    - cube is aimed at multidimensional operations with automatic
      axis matching and alignment
    - table is aimed at two dimensional presentation, e.g.
      in tabular form or in chart
    - cube must have axes with unique names; table does not
    - values in each cube axis must be unique within that axis;
      table does not
    """

    @staticmethod
    def from_cube(cube, row_axes, col_axes, row_label=None, col_label=None):
        if isinstance(row_axes, int) or isinstance(row_axes, str):
            row_axes = [row_axes]
        if isinstance(col_axes, int) or isinstance(col_axes, str):
            col_axes = [col_axes]
        row_axis_indices = [cube.axis_index(a) for a in row_axes]
        col_axis_indices = [cube.axis_index(a) for a in col_axes]

        row_axis_list = [cube.axes[i] for i in row_axis_indices]
        col_axis_list = [cube.axes[i] for i in col_axis_indices]

        row_header = Header.from_axes(row_axis_list, row_label)
        col_header = Header.from_axes(col_axis_list, col_label)

        cube = cube.transpose(row_axis_indices + col_axis_indices)
        values = cube.values.reshape(len(row_header), len(col_header))
        return Table(values, row_header, col_header)

    def __init__(self, values, row_header=None, col_header=None):

        self.__values = np.atleast_2d(values)
        if self.__values.ndim != 2:
            raise ValueError("values must have 2 dimensions")

        #nrow = self.__values.shape[0]
        #ncol = self.__values.shape[1]

        #if row_axis is not None:
        #    if nrow != row_axis.length:
        #        raise ValueError("invalid number of rows")

        #if col_axis is not None:
        #    if ncol != col_axis.length:
        #        raise ValueError("invalid number of columns")

        self.__row_header = row_header
        self.__col_header = col_header

    def __repr__(self):
        return "rows:\n{}\ncolumns:\n{}\nvalues:\n{}".format(self.__row_header, self.__col_header, self.__values)

    @property
    def nrows(self):
        return len(self.__row_header)

    @property
    def ncols(self):
        return len(self.__col_header)

    def value(self, row, col):
        return self.__values[row, col]

    @property
    def row_header(self):
        return self.__row_header

    @property
    def col_header(self):
        return self.__col_header

    def row_label(self, index):
        return self.__row_header.label(index)

    def col_label(self, index):
        return self.__col_header.label(index)
        
    @property
    def values(self):
        return self.__values

    def axis_field_count(self, axis_index):
        axis = self._axes[axis_index]
        if axis is None:
            return 0
        else:
            return axis.field_count

    def axis_label(self, axis_index, item_index, label_format=None):
        axis = self._axes[axis_index]
        if axis is None:
            return str(item_index)
        else:
            if label_format is None:
                label_format = axis.label_format
            return axis.label(item_index, label_format)

    def output(self):
        row_count = self.shape[0]
        col_count = self.shape[1]
        lst = [self.axis_label(1, c) for c in range(col_count)]
        print("{}, {}".format("", ", ".join(lst)))
        for r in range(row_count):
            lst = [str(self.__values[r, c]) for c in range(col_count)]
            print("{}, {}".format(self.axis_label(0, r), ", ".join(lst)))

    def filter(self, row_filter=None, col_filter=None):
        """
        :param row_filter:
        :param col_filter:
        :return:
        """
        values = self.__values
        if row_filter is not None:
            values = values[row_filter, :]
        if col_filter is not None:
            values = values[:, col_filter]

        row_axis = self.row_axis
        if row_axis is not None:
            row_axis = row_axis._filter_axis(row_filter)

        col_axis = self.col_axis
        if col_axis is not None:
            col_axis = col_axis._filter_axis(col_filter)

        return Table(values, row_axis, col_axis)
