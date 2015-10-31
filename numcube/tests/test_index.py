import unittest
import numpy as np

from numcube import Index


class IndexTests(unittest.TestCase):

    def test_create_index(self):
        a = Index("A", [10, 20, 30])
        self.assertEqual(a.name, "A")
        self.assertEqual(len(a), 3)

        a = Index("Dim", ["a", "b", "c", "d"])
        self.assertEqual(a.name, "Dim")
        self.assertEqual(len(a), 4)

        # duplicate values
        self.assertRaises(ValueError, Index, "A", ["a", "b", "a"])
        
    def test_index_take(self):
        a = Index("A", ["a", "b", "c", "d"])
        self.assertEqual(a.take([0, 2]).name, "A")  # keep name
        self.assertTrue(np.array_equal(a.take([0, 2]).values, ["a", "c"]))
        self.assertTrue(np.array_equal(a.take([2, 0]).values, ["c", "a"]))
        self.assertRaises(ValueError, a.take, [0, 2, 0])  # duplicate values in Index
        
    def test_compress(self):
        a = Index("A", ["a", "b", "c", "d"])
        selector = [True, False, True, False]
        b = a.compress(selector)
        c = a[np.array(selector)]
        self.assertTrue(np.array_equal(b.values, c.values))
        self.assertEqual(a.name, b.name)  # keep name
        self.assertTrue(np.array_equal(b.values, a.values.compress(selector)))

    def test_writeable(self):
        a = Index("A", [10, 20, 30])
        self.assertRaises(ValueError, a.values.__setitem__, 0, 40)

    def test_index(self):
        a = Index("A", [10, 20, 30])
        self.assertEqual(a.indexof(10), 0)
        self.assertTrue(np.array_equal(a.indexof([10, 30]), [0, 2]))

        b = Index("Dim", ["a", "b", "c", "d"])
        self.assertEqual(b.indexof("c"), 2)
        self.assertTrue(np.array_equal(b.indexof(["d", "c"]), [3, 2]))

        # invalid Index name
        self.assertRaises(TypeError, Index, 1, [1, 2, 3])

    def test_contains(self):
        a = Index("A", [10, 20, 30])
        self.assertEqual(a.contains(20), True)
        self.assertEqual(a.contains(40), False)
        self.assertTrue(np.array_equal(a.contains([0, 10, 20, 40]), [False, True, True, False]))

        b = Index("B", ["jan", "feb", "mar", "apr"])
        self.assertEqual(b.contains("feb"), True)
        self.assertEqual(b.contains("jun"), False)
        self.assertTrue(np.array_equal(b.contains(["jan", "dec", "feb"]), [True, False, True]))
