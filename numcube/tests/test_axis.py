import unittest

from numcube.axis import Axis


class AxisTests(unittest.TestCase):

    def test_create_series(self):
        a = Axis("A", [10, 20, 30])
        self.assertEqual(a.name, "A")
        self.assertEqual(len(a), 3)

        b = Axis("A", ["a", "b", "c", "d"])
        self.assertEqual(b.name, "A")
        self.assertEqual(len(b), 4)

        # duplicate values are OK
        try:
            Axis("A", ["a", "b", "a"])
        except ValueError:
            self.fail("Series raised ValueError unexpectedly")
