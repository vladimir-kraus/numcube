import unittest
import numpy as np

from numcube.axis import Axis


class AxisTests(unittest.TestCase):

    def test_create(self):
        a = Axis("A", [10, 20, 30])
        self.assertEqual(a.name, "A")
        self.assertEqual(len(a), 3)

        b = Axis("A", ["a", "b", "c", "d"])
        self.assertEqual(b.name, "A")
        self.assertEqual(len(b), 4)

        # duplicate values are OK
        try:
            Axis("A", ["a", "b", "a"])
        except:
            self.fail("Axis with duplicate values raised error unexpectedly")

    def test_indexing(self):
        a = Axis("A", [10, 20, 30])

        self.assertTrue(np.array_equal(a.values == 10, [True, False, False]))
        self.assertEqual(a[0].values, 10)
        self.assertEqual(a.values[0], 10)
