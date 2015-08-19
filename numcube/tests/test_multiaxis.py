import unittest
import numpy as np

from numcube import MultiAxis


class MultiAxisTests(unittest.TestCase):

    def test_create(self):
		# from numpy.recarray
		recarray = np.array([(1.5, 1), (0.5, 1)], dtype=[('A', float), ('B', int)])
        a = MultiAxis(recarray)
        self.assertEqual(a.name, ("A", "B"))
        self.assertEqual(len(a), 2)

