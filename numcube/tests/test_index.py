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

        # invalid Index name
        self.assertRaises(TypeError, Index, 1, [1, 2, 3])
        
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
        # once index has been created, its values cannot be changed in order not to break lookup function
        a = Index("A", [10, 20, 30])
        self.assertRaises(ValueError, a.values.__setitem__, 0, 40)
        self.assertRaises(ValueError, a.values.sort)

    def test_index(self):
        a = Index("A", [10, 20, 30])
        b = Index("Dim", ["a", "b", "c", "d"])

        # a single value
        self.assertEqual(a.indexof(10), 0)
        self.assertEqual(b.indexof("c"), 2)

        # multiple values
        self.assertTrue(np.array_equal(a.indexof([10, 30]), [0, 2]))
        self.assertTrue(np.array_equal(b.indexof(["d", "c"]), [3, 2]))

        # non-existent value raises KeyError (similar to dictionary lookup)
        self.assertRaises(KeyError, a.indexof, 0)
        self.assertRaises(KeyError, b.indexof, "e")
        self.assertRaises(KeyError, b.indexof, None)
        self.assertRaises(KeyError, a.indexof, [0, 1])
        self.assertRaises(KeyError, b.indexof, ["d", "e"])

    def test_contains(self):
        a = Index("A", [10, 20, 30])
        b = Index("Dim", ["a", "b", "c", "d"])

        # a single value
        self.assertEqual(a.contains(20), True)
        self.assertEqual(a.contains(40), False)
        self.assertEqual(b.contains("b"), True)
        self.assertEqual(b.contains("e"), False)

        # multiple values returns one-dimensional numpy array of logical values
        self.assertTrue(np.array_equal(a.contains([0, 10, 20, 40]), [False, True, True, False]))
        self.assertTrue(np.array_equal(b.contains(["a", "e", "b"]), [True, False, True]))
