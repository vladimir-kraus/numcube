import unittest
import numpy as np

from numcube.experimental import MultiAxis


class MultiAxisTests(unittest.TestCase):

    def test_create(self):
        values = np.array([(1.5, 1, "x"), (0.5, 1, "y")], dtype=[('A', float), ('B', int), ('C', str)])
        a = MultiAxis(values)
        self.assertEqual(a.name, ("A", "B", "C"))
        self.assertEqual(len(a), 2)
