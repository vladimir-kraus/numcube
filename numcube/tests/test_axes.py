import unittest
import numpy as np
from numcube import Index, Series, Axes


class AxesTests(unittest.TestCase):

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

    def test_index_index(self):
        a = Index("A", [10, 20, 30])
        self.assertEqual(a.index(10), 0)
        self.assertTrue(np.array_equal(a.index([10, 30]), [0, 2]))

        b = Index("Dim", ["a", "b", "c", "d"])
        self.assertEqual(b.index("c"), 2)
        self.assertTrue(np.array_equal(b.index(["d", "c"]), [3, 2]))

        # invalid Index name
        self.assertRaises(TypeError, Index, 1, [1, 2, 3])

    def test_index_contains(self):
        a = Index("A", [10, 20, 30])
        self.assertEqual(a.contains(20), True)
        self.assertEqual(a.contains(40), False)
        self.assertTrue(np.array_equal(a.contains([0, 10, 20, 40]), [False, True, True, False]))

        b = Index("B", ["jan", "feb", "mar", "apr"])
        self.assertEqual(b.contains("feb"), True)
        self.assertEqual(b.contains("jun"), False)
        self.assertTrue(np.array_equal(b.contains(["jan", "dec", "feb"]), [True, False, True]))

    def test_create_axes(self):
        ax = Axes([Index("A", [10, 20]), Index("B", ["a", "b", "c"])])
        self.assertEqual(len(ax), 2)
        self.assertEqual(ax[0].name, "A")
        self.assertEqual(len(ax[1]), 3)

        # duplicate axes
        self.assertRaises(ValueError, Axes, [Index("A", [10, 20]), Index("A", ["a", "b", "c"])])

        # invalid axis type
        self.assertRaises(TypeError, Axes, [None, Index("A", ["a", "b", "c"])])