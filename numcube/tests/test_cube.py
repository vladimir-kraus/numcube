import unittest
import functools
import numpy as np

from numcube import Index, Axis, Cube, stack, concatenate


def year_quarter_cube():
    """Creates a sample 2D cube with axes "year" and "quarter" with shape (3, 4)."""
    values = np.arange(12).reshape(3, 4)
    ax1 = Index("year", [2014, 2015, 2016])
    ax2 = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
    return Cube(values, [ax1, ax2])


def year_quarter_weekday_cube():
    """Creates 3D cube with axes "year", "quarter", "weekday" with shape (3, 4, 7)."""
    values = np.arange(3 * 4 * 7).reshape(3, 4, 7)
    ax1 = Index("year", [2014, 2015, 2016])
    ax2 = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
    ax3 = Index("weekday", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
    return Cube(values, [ax1, ax2, ax3])


class CubeTests(unittest.TestCase):

    def test_empty_cube(self):
        # TODO C = Cube([], [])
        pass

    def test_create_scalar(self):

        C = Cube(1, None)
        self.assertEqual(C.ndim, 0)
        self.assertEqual(C.values.ndim, 0)

        C = Cube(1, [])
        self.assertEqual(C.ndim, 0)
        self.assertEqual(C.values.ndim, 0)

    def test_create_cube(self):
    
        a = Index("A", [10, 20, 30])
        b = Index("B", ["a", "b", "c", "d"])
        c = Index("C", [1.1, 1.2])

        # test Cube.zeros()
        A = Cube.zeros([a, c])
        self.assertTrue(np.array_equal(A.values, [[0, 0], [0, 0], [0, 0]]))

        # test Cube.ones()
        A = Cube.ones([a, c])
        self.assertTrue(np.array_equal(A.values, [[1, 1], [1, 1], [1, 1]]))

        # test Cube.full()
        A = Cube.full([a, c], np.inf)
        self.assertTrue(np.array_equal(A.values, [[np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]]))

        # test Cube.full with NaNs
        # note: be careful because NaN != NaN so np.array_equal does not work
        A = Cube.full([a, c], np.nan)
        np.testing.assert_equal(A.values, [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
        
        # create one-dimensional cube
        values = np.arange(3)
        try:
            Cube(values, (a,))
            Cube(values, a)  # no need to pass axes as collection if there is only one axis
        except Exception:
            self.fail("raised exception unexpectedly")
        
        # two-dimensional cubes
        values = np.arange(12).reshape(3, 4)
        try:
            Cube(values, (a, b))
            Cube(values, [a, b])
        except Exception:
            self.fail("raised exception unexpectedly")

        # wrong number of dimensions
        self.assertRaises(ValueError, Cube, values, [a, b, c])

        # wrong lengths of dimensions
        self.assertRaises(ValueError, Cube, values, [a, c])
        self.assertRaises(ValueError, Cube, values, [b, a])

    def test_axes(self):
        """Counting axes, accessing axes by name or index, etc."""
        C = year_quarter_cube()

        # number of dimensions (axes)
        self.assertEqual(C.ndim, 2)

        # get axis by index, by name and by axis object
        axis1 = C.axis(0)
        axis2 = C.axis("year")
        self.assertEqual(axis1, axis2)
        axis3 = C.axis(-2)  # counting backwards
        self.assertEqual(axis1, axis3)
        axis4 = C.axis(axis1)
        self.assertEqual(axis1, axis4)

        # invalid axis identification raises LookupError
        self.assertRaises(LookupError, C.axis, "bad_axis")
        self.assertRaises(LookupError, C.axis, 3)
        self.assertRaises(LookupError, C.axis, Axis("bad_axis", []))

        # invalid argument types
        self.assertRaises(TypeError, C.axis, 1.0)
        self.assertRaises(TypeError, C.axis, None)

    def test_axis_index(self):
        C = year_quarter_cube()

        # get axis index by name
        self.assertEqual(C.axis_index("year"), 0)
        self.assertEqual(C.axis_index("quarter"), 1)

        # get axis index by Axis object
        self.assertEqual(C.axis_index(C.axis(0)), 0)
        self.assertEqual(C.axis_index(C.axis(1)), 1)

        # get axis index by index (negative turns to positive)
        self.assertEqual(C.axis_index(0), 0)
        self.assertEqual(C.axis_index(-2), 0)

        # invalid axes
        self.assertRaises(LookupError, C.axis_index, "bad_axis")
        self.assertRaises(LookupError, C.axis_index, Axis("bad_axis", []))
        self.assertRaises(LookupError, C.axis_index, 2)

        # invalid argument types
        self.assertRaises(TypeError, C.axis_index, 1.0)
        self.assertRaises(TypeError, C.axis_index, None)

    def test_has_axis(self):
        C = year_quarter_cube()

        # whether axis exists - by name
        self.assertTrue(C.has_axis("year"))
        self.assertFalse(C.has_axis("bad_axis"))

        # whether axis exists - by index (negative index means counting backwards)
        self.assertTrue(C.has_axis(1))
        self.assertFalse(C.has_axis(2))
        self.assertTrue(C.has_axis(-1))
        self.assertFalse(C.has_axis(-3))

        # whether axis exists - by Axis object
        ax1 = C.axis(0)
        self.assertTrue(C.has_axis(ax1))
        ax2 = Axis("year", [])  # same name but different identity
        self.assertFalse(C.has_axis(ax2))

        # invalid argument types
        self.assertRaises(TypeError, C.has_axis, 1.0)
        self.assertRaises(TypeError, C.has_axis, None)

    def test_getitem(self):
        """Getting items by row and column, slicing etc. using __getitem__(item)"""
        C = year_quarter_cube()

        D = C[0:2, 0:3]
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2], [4, 5, 6]]))
        self.assertEqual(D.ndim, 2)
        self.assertEqual(tuple(D.axis_names), ("year", "quarter"))
        
        # indexing - will collapse (i.e. remove) axis
        D = C[0]
        self.assertEqual(D.ndim, 1)
        self.assertEqual(D.axis(0).name, "quarter")
        self.assertTrue(np.array_equal(D.values, [0, 1, 2, 3]))

        D = C[:, 0]
        self.assertEqual(D.ndim, 1)
        self.assertEqual(D.axis(0).name, "year")
        self.assertTrue(np.array_equal(D.values, [0, 4, 8]))

        # slicing - will not collapse axis
        D = C[0:1]
        self.assertEqual(D.ndim, 2)
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2, 3]]))

        D = C[slice(0, 1)]
        self.assertEqual(D.ndim, 2)
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2, 3]]))

        D = C[:, 0:1]
        self.assertEqual(D.ndim, 2)
        self.assertTrue(np.array_equal(D.values, [[0], [4], [8]]))

        D = C[(slice(0, None), slice(0, 1))]
        self.assertEqual(D.ndim, 2)
        self.assertTrue(np.array_equal(D.values, [[0], [4], [8]]))

        # using negative indices
        self.assertTrue(np.array_equal(C[-1].values, [8, 9, 10, 11]))
        self.assertTrue(np.array_equal(C[:, -1].values, [3, 7, 11]))

        # wrong index
        self.assertRaises(IndexError, C.__getitem__, 5)
        # eq. C[0, 0, 0] raises IndexError: too many indices
        self.assertRaises(IndexError, C.__getitem__, (0, 0, 0))
        # wring number of slices
        self.assertRaises(IndexError, C.__getitem__, (slice(0, 2), slice(0, 2), slice(0, 2)))

        # np.newaxis is not supported
        self.assertRaises(ValueError, C.__getitem__, (0, 0, np.newaxis))
        self.assertRaises(ValueError, C.__getitem__, (np.newaxis, 0, 0))

    def test_filter(self):
        """Testing function Cube.filter()"""
        C = year_quarter_cube()  # the year values are [2014, 2015, 2016]

        D = C.filter("year", [2014, 2018])  # 2018 is ignored
        self.assertEqual(D.ndim, 2)
        self.assertTrue((D.values == C.values[0]).all())

        D = C.filter("year", exclude=[2015, 2016, 2018])  # 2018 is ignored
        self.assertEqual(D.ndim, 2)
        self.assertTrue((D.values == C.values[0]).all())

        # for large collections it is better for performance to pass the parameter as a set
        D = C.filter("year", exclude=set(range(2015, 3000)))
        self.assertEqual(D.ndim, 2)
        self.assertTrue((D.values == C.values[0]).all())

    def test_apply(self):
        """Applies a function on each cube element."""
        C = year_quarter_weekday_cube()
        D = C.apply(np.sin)
        self.assertTrue(np.array_equal(np.sin(C.values), D.values))
        
    def test_squeeze(self):
        """Removes axes which have only one element."""
        axis1 = Index("A", [1])  # has only one element, thus can be collapsed
        axis2 = Index("B", [1, 2, 3])

        C = Cube([[1, 2, 3]], [axis1, axis2])
        self.assertEqual(C.ndim, 2)
        D = C.squeeze()
        self.assertEqual(D.ndim, 1)
        self.assertEqual(D.axis(0).name, "B")

        C = Cube([[1], [2], [3]], [axis2, axis1])
        self.assertEqual(C.ndim, 2)
        D = C.squeeze()
        self.assertEqual(D.ndim, 1)
        self.assertEqual(D.axis(0).name, "B")

        axis3 = Index("C", [1])  # has only one element, thus can be collapsed
        C = Cube([[1]], [axis1, axis3])
        self.assertEqual(C.ndim, 2)
        D = C.squeeze()  # will collapse both axes
        self.assertEqual(D.ndim, 0)

    def test_compress(self):
        C = year_quarter_cube()
        
        D = C.compress([True, False, False], 0)
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2, 3]]))
        self.assertEqual(D.ndim, 2)
        self.assertEqual(tuple(D.axis_names), ("year", "quarter"))
        
        E = C.compress([True, False, True, False], "quarter")
        self.assertTrue(np.array_equal(E.values, [[0, 2], [4, 6], [8, 10]]))

        # using numpy array of bools
        E = C.compress(np.arange(1, 4) <= 1, "quarter")
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2, 3]]))
        self.assertEqual(D.ndim, 2)
        self.assertEqual(tuple(D.axis_names), ("year", "quarter"))
        
        # ints instead of bools; 0 = False, other = True
        # similarly for other types; Python bool conversion is used
        D = C.compress([1, 0, 0], 0)
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2, 3]]))
        self.assertEqual(D.ndim, 2)
        self.assertEqual(tuple(D.axis_names), ("year", "quarter"))

        # wrong length of bool collection - too short ...
        D = C.compress([True, False], 0)  # unspecified means False
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2, 3]]))

        # ... and too long
        D = C.compress([True, False, False, False], 0)  # this is OK, the extra False is ignored
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2, 3]]))
        self.assertRaises(IndexError, C.compress, [True, False, False, True], 0)  # but this is not OK

    def test_transpose(self):
        C = year_quarter_weekday_cube()

        # transpose by axis indices
        D = C.transpose([1, 0, 2])

        self.assertEqual(D.values.shape, (4, 3, 7))

        # check that original cube has not been changed
        self.assertEqual(C.values.shape, (3, 4, 7))

        # compare with numpy transpose
        self.assertTrue(np.array_equal(D.values, C.values.transpose([1, 0, 2])))

        # transpose by axis names
        E = C.transpose(["quarter", "year", "weekday"])
        self.assertTrue(np.array_equal(D.values, E.values))
        
        # transpose axes specified by negative indices
        E = C.transpose([-2, -3, -1])
        self.assertTrue(np.array_equal(D.values, E.values))

        # specify 'front' argument (does not need to be specified explicitly)
        E = C.transpose(["quarter", "year"])
        self.assertTrue(np.array_equal(D.values, E.values))
        E = C.transpose([1, 0])
        self.assertTrue(np.array_equal(D.values, E.values))

        # specify 'back' argument
        E = C.transpose(back=["year", "weekday"])
        self.assertTrue(np.array_equal(D.values, E.values))
        E = C.transpose(back=[0, 2])
        self.assertTrue(np.array_equal(D.values, E.values))

        # specify 'front' and 'back' argument
        E = C.transpose(front="quarter", back="weekday")
        self.assertTrue(np.array_equal(D.values, E.values))

        # transpose with wrong axis indices
        self.assertRaises(LookupError, C.transpose, [3, 0, 2])
        self.assertRaises(LookupError, C.transpose, [-5, 0, 1])

        # transpose with wrong axis names
        self.assertRaises(LookupError, C.transpose, ["A", "B", "C"])

        # invalid axis identification raises LookupError
        self.assertRaises(LookupError, C.transpose, ["year", "weekday", "quarter", "A"])
        self.assertRaises(LookupError, C.transpose, [1, 0, 2, 3])

        # duplicate axes raise ValueError
        self.assertRaises(ValueError, C.transpose, [0, 0, 2])
        self.assertRaises(ValueError, C.transpose, ["year", "year", "quarter"])
        self.assertRaises(ValueError, C.transpose, front=["year", "weekday"], back=["year", "quarter"])
        self.assertRaises(ValueError, C.transpose, front=[1, 2], back=[0, 1])

        # invalid types
        self.assertRaises(TypeError, C.transpose, [1.1, 0, 2])
        self.assertRaises(TypeError, C.transpose, [None, "weekday", "year"])

    def test_operations(self):
        values = np.arange(12).reshape(3, 4)
        c1 = Index("a", [10, 20, 30])
        c2 = Index("b", ["a", "b", "c", "d"])
        C = Cube(values, [c1, c2])

        d1 = Index("a", [10, 20, 30])
        d2 = Index("b", ["a", "b", "c", "d"])
        D = Cube(values, [d1, d2])

        X = C * D
        self.assertTrue(np.array_equal(X.values, values * values))

        E = Cube([0, 1, 2], [Index("a", [10, 20, 30])])

        X2 = C * E
        self.assertTrue(np.array_equal(X2.values, values * np.array([[0], [1], [2]])))

        C3 = Cube([0, 1, 2, 3], [Index("b", ["a", "b", "c", "d"])])
        X3 = C * C3
        self.assertTrue(np.array_equal(X3.values, values * np.array([0, 1, 2, 3])))

        C3 = Cube([0, 1, 2, 3], [Index("b", ["b", "a", "c", "d"])])
        X3 = C * C3
        self.assertTrue(np.array_equal(X3.values, values * np.array([1, 0, 2, 3])))

        values_d = np.array([0, 1])
        D = Cube(values_d, [Index("d", ["d1", "d2"])])
        X = C * D
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.axis(0).name, "a")
        self.assertEqual(X.axis(1).name, "b")
        self.assertEqual(X.axis(2).name, "d")

        self.assertTrue(np.array_equal(X.values, values.reshape(3, 4, 1) * values_d))

        # operations with scalar
        D = 10
        X = C * D
        self.assertTrue(np.array_equal(X.values, values * D))
        X = D * C
        self.assertTrue(np.array_equal(X.values, values * D))
        
        # operations with numpy.ndarray
        D = np.arange(4)
        X = C * D
        self.assertTrue(np.array_equal(X.values, values * D))
        X = D * C
        self.assertTrue(np.array_equal(X.values, values * D))
        
        D = np.arange(3).reshape(3, 1)
        X = C * D
        self.assertTrue(np.array_equal(X.values, values * D))
        X = D * C
        self.assertTrue(np.array_equal(X.values, values * D))
        
        # matching Index and Series
        values_d = np.array([0, 1])
        D = Cube(values_d, Axis("a", [10, 10]))
        X = C * D
        self.assertTrue(np.array_equal(X.values, values.take([0, 0], 0) * values_d[:, np.newaxis]))
        
        values_d = np.array([0, 1, 2, 3])
        D = Cube(values_d, Axis("b", ["d", "d", "c", "a"]))
        X = C * D
        self.assertTrue(np.array_equal(X.values, values.take([3, 3, 2, 0], 1) * values_d))

        # unary plus and minus
        C = year_quarter_cube()
        self.assertTrue(np.array_equal((+C).values, C.values))
        self.assertTrue(np.array_equal((-C).values, -C.values))

        C = year_quarter_cube() + 1  # +1 to prevent division by zero error
        import operator as op
        ops = [op.add, op.mul, op.floordiv, op.truediv, op.sub, op.pow, op.mod,  # arithmetics ops
               op.eq, op.ne, op.ge, op.le, op.gt, op.lt,  # comparison ops
               op.and_, op.or_, op.xor, op.rshift, op.lshift]  # bitwise ops

        # operations with scalar
        D = 2
        for op in ops:
            self.assertTrue(np.array_equal(op(C, D).values, op(C.values, D)))
            self.assertTrue(np.array_equal(op(D, C).values, op(D, C.values)))

        # oprations with numpy array
        D = (np.arange(12).reshape(3, 4) / 6 + 1).astype(np.int)  # +1 to prevent division by zero error
        for op in ops:
            self.assertTrue(np.array_equal(op(C, D).values, op(C.values, D)))
            self.assertTrue(np.array_equal(op(D, C).values, op(D, C.values)))

    def test_groupby(self):
        values = np.arange(12).reshape(3, 4)
        ax1 = Axis("year", [2014, 2014, 2014])
        ax2 = Axis("month", ["jan", "jan", "feb", "feb"])
        C = Cube(values, [ax1, ax2])
        
        D = C.groupby(0, np.mean)  # average by year
        self.assertTrue(np.array_equal(D.values, np.array([[4, 5, 6, 7]])))        
        self.assertTrue(isinstance(D.axis(0), Index))
        self.assertEqual(len(D.axis(0)), 1)
        self.assertEqual(D.values.shape, (1, 4))  # axes with length of 1 are not collapsed
        
        D = C.groupby(ax2.name, np.sum, sorted=False)  # sum by month
        self.assertTrue(np.array_equal(D.values, np.array([[1, 5], [9, 13], [17, 21]])))
        self.assertTrue(np.array_equal(D.axis(ax2.name).values, ["jan", "feb"]))

        D = C.groupby(ax2.name, np.sum)  # sum by month, sorted by default
        self.assertTrue(np.array_equal(D.values, np.array([[5, 1], [13, 9], [21, 17]])))
        self.assertTrue(np.array_equal(D.axis(ax2.name).values, ["feb", "jan"]))
        self.assertTrue(isinstance(D.axis(ax2.name), Index))
        self.assertEqual(len(D.axis(ax2.name)), 2)
        self.assertEqual(D.values.shape, (3, 2))
        
        # testing various aggregation functions
        funcs = [np.sum, np.mean, np.median, np.min, np.max, np.prod]  # , np.diff]
        C = Cube(values, [ax1, ax2])
        for func in funcs:
            D = C.groupby(ax1.name, func)
            self.assertTrue(np.array_equiv(D.values, np.apply_along_axis(func, 0, C.values)))
        
        # testing function with extra parameters which cannot be passed as *args
        third_quartile = functools.partial(np.percentile, q=75)
        D = C.groupby(ax1.name, third_quartile)
        self.assertTrue(np.array_equiv(D.values, np.apply_along_axis(third_quartile, 0, C.values)))

    def test_rename_axis(self):
        C = year_quarter_cube()

        # axes by name
        D = C.rename_axis("year", "Y")
        D = D.rename_axis("quarter", "Q")
        self.assertEqual(tuple(D.axis_names), ("Y", "Q"))

        # axes by index
        D = C.rename_axis(0, "Y")
        D = D.rename_axis(1, "Q")
        self.assertEqual(tuple(D.axis_names), ("Y", "Q"))
        
        # axes with negative indices
        D = C.rename_axis(-2, "Y")
        D = D.rename_axis(-1, "Q")
        self.assertEqual(tuple(D.axis_names), ("Y", "Q"))

        # invalid new axis name type
        self.assertRaises(TypeError, C.rename_axis, 0, 0.0)
        self.assertRaises(TypeError, C.rename_axis, "year", None)

        # duplicate axes
        self.assertRaises(ValueError, C.rename_axis, 0, "quarter")
        self.assertRaises(ValueError, C.rename_axis, "year", "quarter")

        # non-existing axes
        self.assertRaises(LookupError, C.rename_axis, 2, "quarter")
        self.assertRaises(LookupError, C.rename_axis, "bad_axis", "quarter")

    def test_aggregate(self):
        C = year_quarter_cube()

        self.assertTrue((C.sum("quarter") == C.sum(1)).all())
        self.assertTrue((C.sum("quarter") == C.sum(-1)).all())
        self.assertTrue((C.sum("year") == C.sum(keep=1)).all())
        self.assertTrue((C.sum("year") == C.sum(keep=-1)).all())
        self.assertTrue((C.sum(["year"]) == C.sum(keep=[-1])).all())
        self.assertTrue((C.sum("quarter") == C.sum(keep="year")).all())

        year_ax = C.axis("year")
        quarter_ax = C.axis("quarter")
        self.assertTrue((C.sum(year_ax) == C.sum("year")).all())
        self.assertTrue((C.sum(year_ax) == C.sum(keep=quarter_ax)).all())
        self.assertTrue((C.sum(quarter_ax) == C.sum(1)).all())
        self.assertTrue((C.sum(quarter_ax) == C.sum(keep=0)).all())

        self.assertEqual(C.sum(None), C.sum())
        self.assertEqual(C.sum(), np.sum(C.values))
        self.assertEqual(C.mean(), np.mean(C.values))
        self.assertEqual(C.min(), np.min(C.values))
        self.assertEqual(C.max(), np.max(C.values))

        self.assertRaises(LookupError, C.sum, "bad_axis")
        self.assertRaises(LookupError, C.sum, 2)

        self.assertRaises(TypeError, C.sum, 1.0)

    def test_swap_axis(self):
        C = year_quarter_weekday_cube()
        self.assertEqual(C.shape, (3, 4, 7))

        # swap by name
        D = C.swap_axes("year", "quarter")
        self.assertEqual(tuple(D.axis_names), ("quarter", "year", "weekday"))
        self.assertEqual(D.shape, (4, 3, 7))

        # swap by index
        D = C.swap_axes(0, 2)
        self.assertEqual(tuple(D.axis_names), ("weekday", "quarter", "year"))
        self.assertEqual(D.shape, (7, 4, 3))
        
        # swap by index and name
        D = C.swap_axes(0, "quarter")
        self.assertEqual(tuple(D.axis_names), ("quarter", "year", "weekday"))
        self.assertEqual(D.shape, (4, 3, 7))
        
        # swap Axis instances
        year_axis = C.axis("year")
        quarter_axis = C.axis("quarter")
        D = C.swap_axes(year_axis, quarter_axis)
        self.assertEqual(tuple(D.axis_names), ("quarter", "year", "weekday"))
        self.assertEqual(D.shape, (4, 3, 7))
        
        # wrong axis results in LookupError
        self.assertRaises(LookupError, C.sum, "bad_axis")
        self.assertRaises(LookupError, C.sum, 3)        
        
    def test_align_axis(self):
        C = year_quarter_cube()
        ax1 = Axis("year", [2015, 2015, 2014, 2014])
        ax2 = Index("quarter", ["Q1", "Q3"])
        
        D = C.align_axis(ax1)
        D = D.align_axis(ax2)

        # test identity of the new axis
        self.assertTrue(D.axis("year") is ax1)
        self.assertTrue(D.axis("quarter") is ax2)

        # test aligned values
        self.assertTrue(np.array_equal(D.values, [[4, 6], [4, 6], [0, 2], [0, 2]]))

    def test_concatenate(self):
        values = np.arange(12).reshape(3, 4)
        ax1 = Index("year", [2014, 2015, 2016])
        ax2 = Index("month", ["jan", "feb", "mar", "apr"])
        C = Cube(values, [ax1, ax2])

        values = np.arange(12).reshape(4, 3)
        ax3 = Index("year", [2014, 2015, 2016])
        ax4 = Index("month", ["may", "jun", "jul", "aug"])
        D = Cube(values, [ax4, ax3])

        E = concatenate([C, D], "month")
        self.assertEqual(E.ndim, 2)
        self.assertEqual(E.shape, (8, 3))  # the joined axis is always the first
        self.assertTrue(isinstance(E.axis("month"), Axis))

        E = concatenate([C, D], "month", as_index=True)
        self.assertEqual(E.ndim, 2)
        self.assertEqual(E.shape, (8, 3))
        self.assertTrue(isinstance(E.axis("month"), Index))

        # duplicate index values
        self.assertRaises(ValueError, concatenate, [C, C], "month", as_index=True)

        # broadcasting of an axis
        countries = Axis("country", ["DE", "FR"])
        F = D.insert_axis(countries)
        G = concatenate([C, F], "month", broadcast=True)
        self.assertEqual(G.ndim, 3)
        self.assertEqual(G.shape, (8, 3, 2))

        # if automatic broadcasting is not allowed
        self.assertRaises(LookupError, concatenate, [C, F], "month", broadcast=False)

    def test_stack(self):
        C = year_quarter_cube()
        D = year_quarter_cube()
        country_axis = Index("country", ["GB", "FR"])
        E = stack([C, D], country_axis)
        self.assertEqual(E.values.shape, (2, 3, 4))
        self.assertEqual(tuple(E.axis_names), ("country", "year", "quarter"))

        # axis with the same name already exists
        C = year_quarter_cube()
        D = year_quarter_cube()
        year_axis = Index("year", [2000, 2001])
        self.assertRaises(ValueError, stack, [C, D], year_axis)

        # different number of cubes and axis length
        C = year_quarter_cube()
        D = year_quarter_cube()
        country_axis = Index("country", ["GB", "FR", "DE"])
        self.assertRaises(ValueError, stack, [C, D], country_axis)

        # cubes do not have uniform shapes
        C = year_quarter_cube()
        D = year_quarter_weekday_cube()
        country_axis = Index("country", ["GB", "FR"])
        self.assertRaises(LookupError, stack, [C, D], country_axis)  # TODO: maybe change to different error type
        
    def test_combine_axes(self):
        C = year_quarter_weekday_cube()

        # duplicate axes
        self.assertRaises(ValueError, C.combine_axes, ["year", "year"], "period", "{}-{}")
        self.assertRaises(ValueError, C.combine_axes, ["year", "quarter"], "weekday", "{}-{}")

        D = C.combine_axes(["year", "quarter"], "period", "{}-{}")
        self.assertEqual(tuple(D.axis_names), ("period", "weekday"))

    def test_take(self):
        C = year_quarter_cube()

        # axis by name
        self.assertTrue(np.array_equal(C.take([0, 1], "year").values, C.values.take([0, 1], 0)))
        self.assertTrue(np.array_equal(C.take([0, 1], "quarter").values, C.values.take([0, 1], 1)))

        # axis by index
        self.assertTrue(np.array_equal(C.take([0, 1], 0).values, C.values.take([0, 1], 0)))
        self.assertTrue(np.array_equal(C.take([0, 1], 1).values, C.values.take([0, 1], 1)))

        # do not collapse dimension - a single int in a list or tuple
        D = C.take([2], 0)
        self.assertEqual(D.ndim, 2)
        self.assertTrue(np.array_equal(D.values, C.values.take([2], 0)))

        D = C.take((2,), 0)
        self.assertEqual(D.ndim, 2)
        self.assertTrue(np.array_equal(D.values, C.values.take([2], 0)))

        # collapse dimension - a single int
        D = C.take(2, 0)
        self.assertEqual(D.ndim, 1)
        self.assertTrue(np.array_equal(D.values, C.values.take(2, 0)))

        # negative index
        self.assertTrue(np.array_equal(C.take([-3, -2], "year").values, C.values.take([0, 1], 0)))

        # wrong axes
        self.assertRaises(LookupError, C.take, [0, 1], "bad_axis")
        self.assertRaises(LookupError, C.take, [0, 1], 2)

        # wrong indices
        self.assertRaises(IndexError, C.take, 4, "year")
        self.assertRaises(IndexError, C.take, [0, 4], "year")
        self.assertRaises(ValueError, C.take, ["X"], "year")
        self.assertRaises(TypeError, C.take, None, "year")

    def test_insert_axis(self):
        C = year_quarter_cube()
        countries = Axis("country", ["DE", "FR"])

        # add as the first axis
        D = C.insert_axis(countries, 0)
        self.assertEqual(D.ndim, 3)
        self.assertEqual(tuple(D.axis_names), ("country", "year", "quarter"))
        self.assertTrue((D.take(0, "country") == C).all())
        self.assertTrue((D.take(1, "country") == C).all())

        # add as the last axis
        D = C.insert_axis(countries, -1)
        self.assertEqual(D.ndim, 3)
        self.assertEqual(tuple(D.axis_names), ("year", "quarter", "country"))
        self.assertTrue((D.take(0, "country") == C).all())
        self.assertTrue((D.take(1, "country") == C).all())
