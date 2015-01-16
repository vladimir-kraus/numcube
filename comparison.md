larry - https://pypi.python.org/pypi/la

* no named axes, only labels
* labels are lists, various types can mix
* implicit inner join
* axes have to be matched manually
* default labels are generated as range(n)

dataarray

* name can be None (or any type?)
* labels cannot be integers (because is in conflict with integer indexing), this is problem for example when the label should be year or month or hour
* implicit inner joins
* axes have to be matched automatically
* axes keep information about its index in the array
* axes can be unnamed (None)
* require the same order of axes (no automatic matching)
* interface experimental and no development since 2012, a dead project


1) protection against breaking the structure consistency

numcube     YES ("99 %")
larry       ?
dataarray   ?

2) named axes

numcube     YES (string only)
larry       NO
dataarray   YES (any object, including None)

3) labeled axes

numcube     YES (stored in 1D numpy array)
larry       YES (stored as list of lists)
dataarray   YES ?

4) unique and nonunique labels

numcube     BOTH
larry       UNIQUE
dataarray   ?

5) protection against implicit inner join

numcube     YES
larry       NO
dataarray   NO

6) automatic axis matching

numcube     YES
larry       NO
dataarray   ?

7) automatic axis alignment

numcube     YES
larry       YES
dataarray   YES

8) access to axes as attributes

numcube     NO
larry       ?
dataarray   ?

9) operators

numcube     YES
larry       ?
dataarray   ?

10) aggregations

11) inheritance from numpy array

numcube     NO (wrapper around numpy array)
larry       YES
dataarray   YES