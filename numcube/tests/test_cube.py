import unittest
import functools
import numpy as np
from numcube import Index, Series, Axes, Cube
from numcube.cube import join, concatenate


def year_quarter_cube():
    """
    Create 2D cube with axes "year" and "quarter".
    :return: new Cube object
    """
    values = np.arange(12).reshape(3, 4)
    ax1 = Index("year", [2014, 2015, 2016])
    ax2 = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
    return Cube(values, [ax1, ax2])


def year_quarter_weekday_cube():
    """
    Create 3D cube with axes "year", "quarter", "weekday".
    :return: new Cube object
    """
    values = np.arange(3 * 4 * 7).reshape(3, 4, 7)
    ax1 = Index("year", [2014, 2015, 2016])
    ax2 = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
    ax3 = Index("weekday", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
    return Cube(values, [ax1, ax2, ax3])


class ExpressionScriptTests(unittest.TestCase):

    def test_create_index(self):
        a = Series("A", [10, 20, 30])
        self.assertEqual(a.name, "A")
        self.assertEqual(len(a), 3)

        a = Series("A", ["a", "b", "c", "d"])
        self.assertEqual(a.name, "A")
        self.assertEqual(len(a), 4)
        
        # duplicit values
        self.assertRaises(ValueError, Series, "A", ["a", "b", "a"])

    def test_create_index(self):
        a = Index("A", [10, 20, 30])
        self.assertEqual(a.name, "A")
        self.assertEqual(len(a), 3)

        a = Index("Dim", ["a", "b", "c", "d"])
        self.assertEqual(a.name, "Dim")
        self.assertEqual(len(a), 4)
        
        # duplicit values
        self.assertRaises(ValueError, Index, "A", ["a", "b", "a"])

    def test_index_writeable(self):
        a = Index("A", [10, 20, 30])
        self.assertRaises(ValueError, a.values.__setitem__, 0, 40)

    def test_index_indexing(self):
        a = Index("A", [10, 20, 30])
        self.assertEqual(a.index(10), 0)
        self.assertTrue(np.array_equal(a.index([10, 30]), [0, 2]))

        b = Index("Dim", ["a", "b", "c", "d"])
        self.assertEqual(b.index("c"), 2)
        self.assertTrue(np.array_equal(b.index(["d", "c"]), [3, 2]))

        # invalid Index name
        self.assertRaises(TypeError, Index, 1, [1, 2, 3])

    def test_create_axes(self):
        ax = Axes([Index("A", [10, 20]), Index("B", ["a", "b", "c"])])
        self.assertEqual(len(ax), 2)
        self.assertEqual(ax[0].name, "A")
        self.assertEqual(len(ax[1]), 3)

    def test_create_cube(self):
        a = Index("A", [10, 20, 30])
        b = Index("B", ["a", "b", "c", "d"])
        c = Index("C", [1.1, 1.2])

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

    def test_cube_transpose(self):
        a = Index("A", [10, 20, 30])
        b = Index("Dim", ["a", "b", "c", "d"])
        c = Index("XXX", [1.1, 2.2])

        values = np.arange(24).reshape(3, 4, 2)
        C = Cube(values, [a, b, c])

        #transpose by Index indices
        D = C.transpose([1, 0, 2])

        self.assertEqual(D.values.shape, (4, 3, 2))

        # check that original cube has not been changed
        self.assertEqual(C.values.shape, (3, 4, 2))

        # compare with numpy transpose
        tvalues = values.transpose([1, 0, 2])
        self.assertTrue(np.array_equal(D.values, tvalues))

        # transpose by Index names
        E = C.transpose(["Dim", "A", "XXX"])
        self.assertTrue(np.array_equal(D.values, E.values))

        # transpose with wrong Index indices
        self.assertRaises(IndexError, C.transpose, [3, 0, 2])
        self.assertRaises(IndexError, C.transpose, [-5, 0, 1])

        # transpose with wrong Index names
        self.assertRaises(KeyError, C.transpose, ["Dim", "A", "Y"])

        # invalid number of axes
        self.assertRaises(ValueError, C.transpose, ["Dim", "A"])
        self.assertRaises(ValueError, C.transpose, [1, 2])
        # note that number of axes is checked before accessing the axes
        # so the wrong number of axes is raised KeyError or IndexError
        self.assertRaises(ValueError, C.transpose, ["Dim", "A", "XXX", "Y"])
        self.assertRaises(ValueError, C.transpose, [1, 0, 2, 3])

        # duplicit axes
        self.assertRaises(ValueError, C.transpose, [0, 0, 2])
        self.assertRaises(ValueError, C.transpose, ["A", "A", "Dim"])

        # invalid types
        self.assertRaises(TypeError, C.transpose, [1.1, 0, 2])
        self.assertRaises(TypeError, C.transpose, [None, "A", "Dim"])

    def test_cube_operations(self):
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
        
    def test_cube_groupby(self):
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
        self.assertTrue(np.array_equal(D.axes[ax2.name].values, ["jan", "feb"]))

        D = C.groupby(ax2.name, np.sum)  # sum by month, sorted by default
        self.assertTrue(np.array_equal(D.values, np.array([[5, 1], [13, 9], [21, 17]])))
        self.assertTrue(np.array_equal(D.axes[ax2.name].values, ["feb", "jan"]))
        self.assertTrue(isinstance(D.axes[ax2.name], Index))
        self.assertEqual(len(D.axes[ax2.name]), 2)
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

    def test_cube_rename_axis(self):
        C = year_quarter_cube()

        # successfull renaming
        D = C.rename_axis("year", "Y")
        D = D.rename_axis("quarter", "Q")
        self.assertEqual(tuple(D.axes.names()), ("Y", "Q"))

        D = C.rename_axis(0, "Y")
        D = D.rename_axis(1, "Q")
        self.assertEqual(tuple(D.axes.names()), ("Y", "Q"))

        # invalid new axis name type
        self.assertRaises(TypeError, C.rename_axis, 0, 0.0)
        self.assertRaises(TypeError, C.rename_axis, "year", None)

        # ducplicate axes
        self.assertRaises(ValueError, C.rename_axis, 0, "quarter")
        self.assertRaises(ValueError, C.rename_axis, "year", "quarter")

        # non-existing axes
        self.assertRaises(IndexError, C.rename_axis, 2, "quarter")
        self.assertRaises(KeyError, C.rename_axis, "scenario", "quarter")
        self.assertRaises(LookupError, C.rename_axis, 2, "quarter")
        self.assertRaises(LookupError, C.rename_axis, "scenario", "quarter")

    def test_cube_swap_axis(self):
        C = year_quarter_weekday_cube()

        # swap by name
        D = C.swap_axes("year", "quarter")
        self.assertEqual(tuple(D.axes.names()), ("quarter", "year", "weekday"))

        # swap by index
        D = C.swap_axes(0, 2)
        self.assertEqual(tuple(D.axes.names()), ("weekday", "quarter", "year"))

    def test_cube_concatenate(self):
        values = np.arange(12).reshape(3, 4)
        ax1 = Index("year", [2014, 2015, 2016])
        ax2 = Index("month", ["jan", "feb", "mar", "apr"])
        C = Cube(values, [ax1, ax2])
        
        values = np.arange(12).reshape(3, 4)
        ax3 = Index("year", [2014, 2015, 2016])
        ax4 = Index("month", ["may", "jun", "jul", "aug"])
        D = Cube(values, [ax3, ax4])
        
        E = concatenate([C, D], "month")
        
    def test_cube_combine_axes(self):
        values = np.arange(24).reshape(3, 4, 2)
        ax1 = Index("year", [2014, 2015, 2016])
        ax2 = Index("quarter", ["Q1", "Q2", "Q3", "Q4"])
        ax3 = Index("scenario", ["low", "high"])
        C = Cube(values, [ax1, ax2, ax3])

        self.assertRaises(ValueError, C.combine_axes, ["year", "year"], "period", "{}-{}")
        self.assertRaises(ValueError, C.combine_axes, ["year", "quarter"], "scenario", "{}-{}")

        D = C.combine_axes(["year", "quarter"], "period", "{}-{}")
