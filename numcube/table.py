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
        
    def __getitem__(self, items):
        new_series = tuple(s[items] for s in self._series)
        return Header(new_series, self._format)

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
            if len(a) != 0:  # to protect against zero-length axes problem
                n //= len(a)
            rep_each.append(n)

        series = []
        for a, r1, r2 in zip(axes, rep_all, rep_each):
            repeated_values = np.tile(np.repeat(a.values, r2), r1)
            series.append(Series(a.name, repeated_values))

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
    def from_cube(cube, row_axes=None, col_axes=None, row_label=None, col_label=None):
        if row_axes is None and col_axes is None:
            # the last axis will become the column axis
            # other axes will become row axes
            col_axes = -1  
            
        if row_axes is not None:
            if isinstance(row_axes, int) or isinstance(row_axes, str):
                row_axes = (row_axes,)
                
        if col_axes is not None:
            if isinstance(col_axes, int) or isinstance(col_axes, str):
                col_axes = (col_axes,)
                
        if row_axes is not None:
            row_axis_indices = tuple(cube.axis_index(a) for a in row_axes)
        else:
            row_axis_indices = cube._axes.complement(col_axes)
         
        if col_axes is not None:
            col_axis_indices = tuple(cube.axis_index(a) for a in col_axes)
        else:
            col_axis_indices = cube._axes.complement(row_axes)

        row_axis_list = tuple(cube.axes[i] for i in row_axis_indices)
        col_axis_list = tuple(cube.axes[i] for i in col_axis_indices)

        row_header = Header.from_axes(row_axis_list, row_label)
        col_header = Header.from_axes(col_axis_list, col_label)

        cube = cube.transpose(row_axis_indices + col_axis_indices)
        values = cube.values.reshape(len(row_header), len(col_header))
        return Table(values, row_header, col_header)

    def __init__(self, values, row_header=None, col_header=None):

        self._values = np.atleast_2d(values)
        if self._values.ndim != 2:
            raise ValueError("values must have 2 dimensions")

        nrows = self._values.shape[0]
        ncols = self._values.shape[1]

        #if row_axis is not None:
        if nrows != len(row_header):
                raise ValueError("invalid number of rows")
                
        if ncols != len(col_header):
                raise ValueError("invalid number of columns")                

        #if col_axis is not None:
        #    if ncol != col_axis.length:
        #        raise ValueError("invalid number of columns")

        self._row_header = row_header
        self._col_header = col_header

    def __repr__(self):
        return "rows:\n{}\ncolumns:\n{}\nvalues:\n{}".format(self._row_header, self._col_header, self._values)
        
    def __getitem__(self, items):
        if not isinstance(items, tuple):
            items = (items, slice(None))
        row_items, col_items = items
        new_values = self._values[items]
        new_row_header = self._row_header[row_items]
        new_col_header = self._col_header[col_items]
        return Table(new_values, new_row_header, new_col_header)

    @property
    def nrows(self):
        return len(self._row_header)

    @property
    def ncols(self):
        return len(self._col_header)

    def value(self, row, col):
        return self._values[row, col]
        
    @property
    def size(self):
        return self._values.size

    @property
    def row_header(self):
        return self._row_header

    @property
    def col_header(self):
        return self._col_header

    def row_label(self, index):
        return self._row_header.label(index)

    def col_label(self, index):
        return self._col_header.label(index)
        
    @property
    def values(self):
        return self._values

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
            lst = [str(self._values[r, c]) for c in range(col_count)]
            print("{}, {}".format(self.axis_label(0, r), ", ".join(lst)))

    def filter(self, row_filter=None, col_filter=None):
        """
        :param row_filter:
        :param col_filter:
        :return:
        """
        # TODO: test if a header is None 
        # TODO: is it duplicate with take?
        
        values = self._values
        if row_filter is not None:
            values = values[row_filter, :]
        if col_filter is not None:
            values = values[:, col_filter]

        row_header = self._row_header
        col_header = self._col_header
            
        if row_header is not None:
            row_header = row_header[row_filter]        

        if col_header is not None:
            col_header = col_header[col_filter]        

        return Table(values, row_header, col_header)
        
    def take(self, row_indices=None, col_indices=None):
    
        # TODO: test if a header is None 
        
        values = self._values
        row_header = self._row_header
        col_header = self._col_header
        if row_indices is not None:
            row_header = row_header[row_indices]
            values = values.take(row_indices, 0)
        if col_indices is not None:
            col_header = col_header[col_indices]
            values = values.take(col_indices, 1)
        return Table(values, row_header, col_header)

    def compress(self, row_condition=None, col_condition=None):
    
        # TODO: test if a header is None 
        
        values = self._values
        row_header = self._row_header
        col_header = self._col_header
        if row_condition is not None:
            row_header = row_header[row_condition]
            values = values.compress(row_condition, 0)
        if col_condition is not None:
            col_header = col_header[col_condition]
            values = values.compress(col_condition, 1)
        return Table(values, row_header, col_header)
        