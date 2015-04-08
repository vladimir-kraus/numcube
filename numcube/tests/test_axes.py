import unittest
import numpy as np
from numcube import Index, Series, Axes, intersect


def create_axes():
    # create a list of axes of different types
    return [Series("A", [10, 20, 30, 40]), Index("B", [10, 20, 30, 40])]


class AxesTests(unittest.TestCase):

    def test_axis_getitem(self):
        axes = create_axes()

        for a in axes:
            # indexing with a single index
            self.assertEqual(a[1].values, 20)
            self.assertEqual(a[-1].values, 40)

            # slicing
            self.assertTrue(np.array_equal(a[1:3].values, [20, 30]))
            
            # indexing with list of ints
            sel = [2, 1, -1]
            self.assertTrue(np.array_equal(a[sel].values, [30, 20, 40]))
            
            # indexing with numpy array of ints
            sel = np.array([2, 1, -1])
            self.assertTrue(np.array_equal(a[sel].values, [30, 20, 40]))
            
            # selection with numpy array of bools
            sel = np.array([False, False, True, True])
            self.assertTrue(np.array_equal(a[sel].values, [30, 40]))
            
            sel = a.values > 20
            self.assertTrue(np.array_equal(a[sel].values, [30, 40]))
        
    def test_axis_filter(self):
        axes = create_axes()
        for a in axes:
            self.assertEqual(a.filter(10).values, 10)
            self.assertTrue(np.array_equal(a.filter([30, 10, 20]).values, [10, 20, 30]))
            self.assertTrue(np.array_equal(a.filter((30, 10, 20)).values, [10, 20, 30]))
            self.assertTrue(np.array_equal(a.filter({30, 10, 20}).values, [10, 20, 30]))
        
        # TODO: test nonexisting values

    def test_create_series(self):
        a = Series("A", [10, 20, 30])
        self.assertEqual(a.name, "A")
        self.assertEqual(len(a), 3)

        a = Series("A", ["a", "b", "c", "d"])
        self.assertEqual(a.name, "A")
        self.assertEqual(len(a), 4)

        # duplicit values are OK
        try:
            a = Series("A", ["a", "b", "a"])
        except ValueError:
            self.fail("Series raised ValueError unexpectedly")
            
    def test_complement(self):
        a = Index("A", [10, 20, 30])
        b = Index("B", [10, 20, 30])
        c = Index("C", [10, 20, 30])
        axs = Axes((a, b, c))
        self.assertEqual(axs.complement("A"), (1, 2))
        self.assertEqual(axs.complement(["B"]), (0, 2))
        self.assertEqual(axs.complement(("A", "C")), (1,))
        self.assertRaises(ValueError, axs.complement, ["A", "A"])
        self.assertRaises(ValueError, axs.complement, ["B", 1])

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
        # TODO - put to separate test_index file
        a = Index("A", ["a", "b", "c", "d"])
        self.assertEqual(a.take([0, 2]).name, "A")  # preserve name
        self.assertEqual(a.take([0, 2]).values, ("a", "c"))
        self.assertEqual(a.take([2, 0]).values, ("c", "a"))
        self.assertRaises(ValueError, a.take, [0, 2, 0])  # duplicate values in Index
        
    def test_index_compress(self):
        # TODO - put to separate test_index file
        a = Index("A", ["a", "b", "c", "d"])
        self.assertEqual(a.compress([True, False, True, False]).name, "A")  # preserve name
        self.assertEqual(a.compress([True, False, True, False]).values, ("a", "c"))
        
    def test_numpy_recarray_axis(self):
        array = np.array([(1, 1.0), (2, 2.0), (3, 3.0)], dtype=[("int", int), ("float", float)])
        #print(array)
        #for i in array:
        #    print(i, type(i))
        #a = Index("Dim", array)
        b = Series("Dim", array)

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
        # create from a single axis
        ax = Axes(Index("A", [10, 20]))
        self.assertEqual(len(ax), 1)
        self.assertEqual(ax[0].name, "A")
        self.assertEqual(len(ax[0]), 2)

        # create from a list of axes
        ax = Axes([Index("A", [10, 20]), Index("B", ["a", "b", "c"])])
        self.assertEqual(len(ax), 2)
        self.assertEqual(ax[0].name, "A")
        self.assertEqual(len(ax[1]), 3)

        # create from another Axes object
        ax2 = Axes(ax)
        self.assertEqual(len(ax2), 2)
        self.assertEqual(ax2[0].name, "A")
        self.assertEqual(len(ax2[1]), 3)

        # duplicate axes
        self.assertRaises(ValueError, Axes, [Index("A", [10, 20]), Index("A", ["a", "b", "c"])])

        # invalid axis type
        self.assertRaises(TypeError, Axes, [None, Index("A", ["a", "b", "c"])])

    def test_intersect(self):
        ax1 = Index("A", ["a1", "a2", "a3"])
        ax2 = Index("B", ["b1", "b2", "b4", "b3"])
        ax3 = Index("C", [1, 2, 3])
        ax4 = Index("A", ["a3", "a2"])
        ax5 = Index("B", ["b3", "b2", "b1", "b5"])
        ax6 = Index("D", [1, 2, 3])
        axes1 = Axes([ax1, ax2, ax3])
        axes2 = Axes([ax5, ax4, ax6])
        axes_int = intersect(axes1, axes2)
        self.assertTrue(isinstance(axes_int, Axes))
        self.assertEqual(len(axes_int), 2)
        self.assertEqual(len(axes_int[0]), 2)
        self.assertEqual(len(axes_int[1]), 3)
        self.assertTrue(np.array_equal(axes_int[0].values, ("a2", "a3")))
        self.assertTrue(np.array_equal(axes_int[1].values, ("b1", "b2", "b3")))

    def test_remove(self):
        ax1 = Index("A", ["a1", "a2", "a3"])
        ax2 = Index("B", ["b1", "b2", "b4", "b3"])
        ax3 = Index("C", [1, 2, 3])
        axs1 = Axes([ax1, ax2, ax3])
        axs2 = axs1.remove("A")
        self.assertEqual(tuple(axs2.names), ("B", "C"))
        axs2 = axs1.remove("C")
        self.assertEqual(tuple(axs2.names), ("A", "B"))
        axs2 = axs1.remove(0)
        self.assertEqual(tuple(axs2.names), ("B", "C"))
        axs2 = axs1.remove(2)
        self.assertEqual(tuple(axs2.names), ("A", "B"))