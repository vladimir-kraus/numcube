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
        C = year_quarter_weekday_cube()
        T = Table.from_cube(C, ['year', 'quarter'], ['weekday'])
        # print(T)
        T = Table.from_cube(C, ['year'], ['weekday', 'quarter'])
        # print(T)
