import timeit

setup = """
import numpy as np
a = np.array(range(1000000))
b = np.array(range(1000001))
c = a
d = a.view()
e = b.view()
"""

codes = ["np.array_equal(a, b)",
         "np.array_equal(a, c)",
         "np.array_equal(a, d)",
         "np.array_equal(a, e)",
         "np.array_equiv(a, c)",
         "np.array_equiv(a, d)",
         "np.array_equiv(a, e)",
         "a is c"]

for code in codes:
    x = timeit.timeit(code, setup, number=100)
    print(x)

#print(id(a))
#print(id(b))
#print(id(c))
#print(id(d))
#print(id(e))
