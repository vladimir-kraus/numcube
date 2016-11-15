import doctest
import os
import sys
import unittest

from nose.tools import nottest

import numcube.tests.doctests.axis
import numcube.tests.doctests.cube
import numcube.tests.doctests.index


@nottest
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(numcube.tests.doctests.axis))
    tests.addTests(doctest.DocTestSuite(numcube.tests.doctests.index))
    tests.addTests(doctest.DocTestSuite(numcube.tests.doctests.cube))
    return tests

if __name__ == "__main__":
    path = os.path.dirname(os.path.realpath(__file__))
    module_path = os.path.join(path, os.pardir, os.pardir)
    sys.path.insert(0, module_path)    
    print("Testing in {}".format(path))
    test_suite = unittest.defaultTestLoader.discover(path, pattern="*.py")
    unittest.TextTestRunner(verbosity=2).run(test_suite)
