numpy usecases
==============
This document should describe the case when multidimensional arrays with named and labeled axes can be used.

Cube manipulation
-----------------
Roll-up
* reducing granularity of an axis - groupby
* reducing dimensionality - sum, svg, min, max, all, any, ...

Drill-down
* ???

Slice
* by filtering C.filter("year", [2010])
* C[year[2010:2012]]

Dice
* same as slice but several dimensions

Pivot
* transpose


Numpy recordsets
----------------


Arrays of objects
-----------------

class Person(object):
    first_name = None
    last_name = None
    def __init__(self, fn, ln):
        first_name = fn
        last_name = ln


A = Person("Alan", "Turing")
I = Person("Isaac" "Newton")
N = Person("Nicola", "Tesla")

persons = Axis("persons", [A, I, N])
...

