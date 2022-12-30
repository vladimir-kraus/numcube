**larry - https://pypi.python.org/pypi/la**

* no named axes, only labels
* labels are lists, various types can mix
* implicit inner join
* axes have to be matched manually
* default labels are generated as range(n)

**dataarray**

* name can be None (or any type?)
* labels cannot be integers (because is in conflict with integer indexing), this is problem for example when the label should be year or month or hour
* implicit inner joins
* axes have to be matched automatically
* axes keep information about its index in the array
* axes can be unnamed (None)
* require the same order of axes (no automatic matching)
* interface experimental and no development since 2012, a dead project

**automatic axis matching**

numcube     YES
larry       NO
dataarray   ?

**automatic axis alignment**

numcube     YES
larry       YES
dataarray   YES

**named axes**

numcube     YES (string only)
larry       NO
dataarray   YES (any object, including None)

**annotated axes**

numcube     YES (stored in 1D numpy array)
larry       YES (stored as list of lists)
dataarray   YES ?

**protection against breaking the structure consistency**

numcube     YES ("99 %")
larry       ?
dataarray   ?

**unique and non-unique labels**

numcube     BOTH
larry       UNIQUE
dataarray   ?

**protection against implicit inner join (i.e. danger of unintentional leaving out of values)**

numcube     YES
larry       NO
dataarray   NO

**access to axes as attributes, e.g. cube.x do denote axis 'x'**

numcube     NO
larry       NO (does not have named axes)
dataarray   YES

**operators**

numcube     YES
larry       ?
dataarray   ?

**aggregations**

numcube     YES
larry       ?
dataarray   ?

**direct inheritance from numpy array**

numcube     NO (but is a wrapper around numpy array and provides access to underlying numpy array)
larry       YES
dataarray   YES