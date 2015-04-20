import unittest
import numpy as np
from numcube import Index, Cube, Table


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


class TableTests(unittest.TestCase):

    def test_create_table(self):
        pass
        
    def test_from_cube(self):
        C = year_quarter_weekday_cube()
        T = Table.from_cube(C)
        self.assertEqual(T.nrows, 12)
        self.assertEqual(T.ncols, 7)

        T = Table.from_cube(C, ['year', 'quarter'], ['weekday'])
        T = Table.from_cube(C, ['year'], ['weekday', 'quarter'])
        T = Table.from_cube(C)
        T = Table.from_cube(C, 'year')
        T = Table.from_cube(C, 0)
        
    def test_getitem(self):
        C = year_quarter_cube()
        T = Table.from_cube(C)
        self.assertTrue(T[0, 0].values == 0)
        self.assertTrue(T[2, 3].values == 11)
        self.assertTrue(np.array_equal(T[1].values, [[4, 5, 6, 7]]))
        self.assertTrue(np.array_equal(T[-2].values, [[4, 5, 6, 7]]))
        self.assertTrue(np.array_equal(T[0, :].values, [[0, 1, 2, 3]]))
        self.assertTrue(np.array_equal(T[0:2, 0:2].values, [[0, 1], [4, 5]]))
        
        # the following numpy style of indexing is not possible
        # note: compare to Table.take([0, 2], [0, 1])
        self.assertRaises(ValueError, T.__getitem__, ([0, 2], [0, 1]))
        
    def test_take(self):
        C = year_quarter_cube()
        T = Table.from_cube(C)    
        self.assertTrue(np.array_equal(T.take([0, 2], [0, 1]).values, [[0, 1], [8, 9]]))

    def test_transpose(self):
        t1 = Table.from_cube(year_quarter_weekday_cube())
        t2 = t1.transpose()
        t3 = t2.transpose()
        self.assertEqual(t2.nrows, t1.ncols)
        self.assertEqual(t2.ncols, t1.nrows)
        self.assertTrue(np.array_equal(t1.values.transpose(), t2.values))
        self.assertTrue(np.array_equal(t1.values, t3.values))