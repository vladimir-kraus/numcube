import unittest
import functools
import numpy as np
from numcube import Index, Series, Cube
from numcube.cube import join, concatenate


def year_quarter_cube():
    """
    Create a sample 2D cube with axes "year" and "quarter" with shape (3, 4)
    :return: new Cube object
    """
    values = np.arange(12).reshape(3, 4)
    ax1 = Index("year", [2014, 2015, 2016])
    ax2 = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
    return Cube(values, [ax1, ax2])


def year_quarter_weekday_cube():
    """
    Create 3D cube with axes "year", "quarter", "weekday" with shape (3, 4, 7)
    :return: new Cube object
    """
    values = np.arange(3 * 4 * 7).reshape(3, 4, 7)
    ax1 = Index("year", [2014, 2015, 2016])
    ax2 = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
    ax3 = Index("weekday", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
    return Cube(values, [ax1, ax2, ax3])


class CubeTests(unittest.TestCase):

    def test_empty_cube(self):
        #C = Cube([], [])
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
        
        # create one dimension caube        
        values = np.arange(3)
        try:
            Cube(values, (a,))
            Cube(values, a)  # no need to pass axes as collection if there is only one axis
        except Exception:
            self.fail("raised exception unexpectedly")
        
        # two dimension cubes
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

    def test_getitem(self):
        C = year_quarter_cube()

        D = C[0:2, 0:3]
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2], [4, 5, 6]]))
        self.assertEqual(D.ndim, 2)
        self.assertEqual(tuple(D.axis_names), ("year", "quarter"))
        
        # by logical vector
        sel_y = np.array([True, True, False, False])
        #sel_q = np.array([True, True, True, False])
        D = C[sel_y, :]
        self.assertTrue(np.array_equal(D.values, [[0, 1, 2, 3], [4, 5, 6, 7]]))
        self.assertEqual(D.ndim, 2)
        self.assertEqual(tuple(D.axis_names), ("year", "quarter"))

        # collapsing axis
        D = C[0]
        self.assertTrue(np.array_equal(D.values, [0, 1, 2, 3]))
        self.assertEqual(D.ndim, 1)
        self.assertEqual(D.axes[0].name, "quarter")

        D = C[:, 0]
        self.assertTrue(np.array_equal(D.values, [0, 4, 8]))
        self.assertEqual(D.ndim, 1)
        self.assertEqual(D.axes[0].name, "year")

        self.assertTrue(np.array_equal(C[-1].values, [8, 9, 10, 11]))
        self.assertTrue(np.array_equal(C[:, -1].values, [3, 7, 11]))

        # not collapsing axis
        self.assertTrue(np.array_equal(C[0:1].values, [[0, 1, 2, 3]]))
        self.assertTrue(np.array_equal(C[:, 0:1].values, [[0], [4], [8]]))

        # np.newaxis is not supported
        self.assertRaises(ValueError, C.__getitem__, (0, 0, np.newaxis))
        self.assertRaises(ValueError, C.__getitem__, (np.newaxis, 0, 0))

        # eq. C[0, 0, 0] raises IndexError: too many indices
        self.assertRaises(IndexError, C.__getitem__, (0, 0, 0))

    def test_contains(self):
        C = year_quarter_cube()
        self.assertTrue(0 in C)
        self.assertFalse(12 in C)

    def test_filter(self):
        C = year_quarter_cube()
        D = C.filter("year", [2014, 2018])
        self.assertTrue((D.values == C.values[0]).all())

    def test_apply(self):
        C = year_quarter_weekday_cube()
        D = C.apply(np.sin)
        self.assertTrue(np.array_equal(np.sin(C.values), D.values))
        
    def test_squeeze(self):
        a = Index("A", [1])
        b = Index("B", [1, 2, 3])
        C = Cube([[1, 2, 3]], [a, b])
        D = C.squeeze()
        self.assertEqual(D.ndim, 1)
        self.assertEqual(D.axis(0).name, "B")
        C = Cube([[1], [2], [3]], [b, a])
        D = C.squeeze()
        self.assertEqual(D.ndim, 1)
        self.assertEqual(D.axis(0).name, "B")

    def test_transpose(self):
        C = year_quarter_weekday_cube()

        #transpose by axis indices
        D = C.transpose([1, 0, 2])

        self.assertEqual(D.values.shape, (4, 3, 7))

        # check that original cube has not been changed
        self.assertEqual(C.values.shape, (3, 4, 7))

        # compare with numpy transpose
        tvalues = C.values.transpose([1, 0, 2])
        self.assertTrue(np.array_equal(D.values, tvalues))

        # transpose by axis names
        E = C.transpose(["quarter", "year", "weekday"])
        self.assertTrue(np.array_equal(D.values, E.values))
        
        # transpose axes specified by negative indices
        E = C.transpose([-2, -3, -1])
        self.assertTrue(np.array_equal(D.values, E.values))

        # transpose with wrong axis indices
        self.assertRaises(IndexError, C.transpose, [3, 0, 2])
        self.assertRaises(IndexError, C.transpose, [-5, 0, 1])

        # transpose with wrong axis names
        self.assertRaises(KeyError, C.transpose, ["A", "B", "C"])

        # invalid number of axes
        self.assertRaises(ValueError, C.transpose, ["year", "weekday"])
        self.assertRaises(ValueError, C.transpose, [1, 2])
        # note that number of axes is checked before accessing the axes
        # so the wrong number of axes is raised KeyError or IndexError
        self.assertRaises(ValueError, C.transpose, ["year", "weekday", "quarter", "A"])
        self.assertRaises(ValueError, C.transpose, [1, 0, 2, 3])

        # duplicate axes
        self.assertRaises(ValueError, C.transpose, [0, 0, 2])
        self.assertRaises(ValueError, C.transpose, ["year", "year", "quarter"])

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
        self.assertEqual(len(X.axes), 3)
        self.assertEqual(X.axes[0].name, "a")
        self.assertEqual(X.axes[1].name, "b")
        self.assertEqual(X.axes[2].name, "d")

        self.assertTrue(np.array_equal(X.values, values.reshape(3, 4, 1) * values_d))

        # operation with scalar
        D = 10
        X = C * D
        self.assertTrue(np.array_equal(X.values, values * D))
        X = D * C
        self.assertTrue(np.array_equal(X.values, values * D))
        
        #operation with numpy.ndarray
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
        D = Cube(values_d, Series("a", [10, 10]))
        X = C * D
        self.assertTrue(np.array_equal(X.values, values.take([0, 0], 0) * values_d[:, np.newaxis]))
        
        values_d = np.array([0, 1, 2, 3])
        D = Cube(values_d, Series("b", ["d", "d", "c", "a"]))
        X = C * D
        self.assertTrue(np.array_equal(X.values, values.take([3, 3, 2, 0], 1) * values_d))

        #unary plus and minus
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
        ax1 = Series("year", [2014, 2014, 2014])
        ax2 = Series("month", ["jan", "jan", "feb", "feb"])
        C = Cube(values, [ax1, ax2])
        
        D = C.groupby(0, np.mean)  # average by year
        self.assertTrue(np.array_equal(D.values, np.array([[4, 5, 6, 7]])))        
        self.assertTrue(isinstance(D.axes[0], Index))
        self.assertEqual(len(D.axes[0]), 1)
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
        funcs = [np.sum, np.mean, np.median, np.min, np.max, np.prod]  #, np.diff]
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
        self.assertRaises(IndexError, C.rename_axis, 2, "quarter")
        self.assertRaises(KeyError, C.rename_axis, "scenario", "quarter")
        self.assertRaises(LookupError, C.rename_axis, 2, "quarter")
        self.assertRaises(LookupError, C.rename_axis, "scenario", "quarter")

    def test_aggregate(self):
        C = year_quarter_cube()
        self.assertTrue((C.sum("quarter") == C.sum(1)).all())
        self.assertTrue((C.sum("quarter") == C.sum(-1)).all())
        self.assertTrue((C.sum("year") == C.sum(keep=1)).all())
        self.assertTrue((C.sum("year") == C.sum(keep=-1)).all())
        self.assertTrue((C.sum(["year"]) == C.sum(keep=[-1])).all())
        self.assertTrue((C.sum("quarter") == C.sum(keep="year")).all())
        self.assertEqual(C.sum(), 66)
        self.assertEqual(C.mean(), 5.5)
        self.assertEqual(C.min(), 0)
        self.assertEqual(C.max(), 11)

    def test_swap_axis(self):
        C = year_quarter_weekday_cube()

        # swap by name
        D = C.swap_axes("year", "quarter")
        self.assertEqual(tuple(D.axis_names), ("quarter", "year", "weekday"))

        # swap by index
        D = C.swap_axes(0, 2)
        self.assertEqual(tuple(D.axis_names), ("weekday", "quarter", "year"))
        
    def test_align_axis(self):
        C = year_quarter_cube()
        ax1 = Series("year", [2015, 2015, 2014, 2014])
        ax2 = Index("quarter", ["Q1", "Q3"])
        
        D = C.align_axis(ax1)
        D = D.align_axis(ax2)

        # test identity of the new axis
        self.assertTrue(D.axis("year") is ax1)
        self.assertTrue(D.axis("quarter") is ax2)

        # test aligned values
        self.assertTrue(np.array_equal(D.values, [[4, 6], [4, 6], [0, 2], [0, 2]]))

    def test_concatenate(self):
        # TODO
        values = np.arange(12).reshape(3, 4)
        ax1 = Index("year", [2014, 2015, 2016])
        ax2 = Index("month", ["jan", "feb", "mar", "apr"])
        C = Cube(values, [ax1, ax2])

        values = np.arange(12).reshape(3, 4)
        ax3 = Index("year", [2014, 2015, 2016])
        ax4 = Index("month", ["may", "jun", "jul", "aug"])
        D = Cube(values, [ax3, ax4])

        E = concatenate([C, D], "month")

    def test_join(self):
        C = year_quarter_cube()
        D = year_quarter_cube()
        ax = Index("country", ["GB", "FR"])
        E = join([C, D], ax)
        self.assertEqual(E.values.shape, (2, 3, 4))
        self.assertEqual(tuple(E.axis_names), ("country", "year", "quarter"))
        
    def test_combine_axes(self):
        C = year_quarter_weekday_cube()

        # duplicate axes
        self.assertRaises(ValueError, C.combine_axes, ["year", "year"], "period", "{}-{}")
        self.assertRaises(ValueError, C.combine_axes, ["year", "quarter"], "weekday", "{}-{}")

        D = C.combine_axes(["year", "quarter"], "period", "{}-{}")
        self.assertEqual(tuple(D.axis_names), ("period", "weekday"))

    def test_take(self):
        C = year_quarter_cube()
        self.assertTrue(np.array_equal(C.take([0, 1], "year").values, C.values.take([0, 1], 0)))
        self.assertTrue(np.array_equal(C.take([0, 1], 0).values, C.values.take([0, 1], 0)))
        # do not collapse dimension
        self.assertTrue(np.array_equal(C.take([2], 0).values, C.values.take([2], 0)))
        # collapse dimension
        self.assertTrue(np.array_equal(C.take(2, 0).values, C.values.take(2, 0)))
        self.assertTrue(np.array_equal(C.take([0, 1], "quarter").values, C.values.take([0, 1], 1)))
        self.assertTrue(np.array_equal(C.take([0, 1], 1).values, C.values.take([0, 1], 1)))
        # do not collapse dimension
        self.assertTrue(np.array_equal(C.take([2], 1).values, C.values.take([2], 1)))
        # collapse dimension
        self.assertTrue(np.array_equal(C.take(2, 1).values, C.values.take(2, 1)))

