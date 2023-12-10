:tocdepth: 3

``mccube.formulae`` module
==========================

.. automodule:: mccube.formulae
.. currentmodule:: mccube.formulae


Abstract cubature
-----------------

.. autoclass:: AbstractCubature
    :members:

.. autofunction:: evaluate_cubature

.. autofunction:: search_cubature_registry

.. autodata:: builtin_cubature_registry
    :no-value:
    

Gaussian cubature
-----------------

We currently only consider formulae that can be constructed for any dimension 
$d$ of the :class:`~mccube.regions.GaussianRegion`. That is, formulae for the 
*"probabilist's"* Hermite measure, and appropriately rescaled formulae for the 
*"physicist's"* Hermite measure, denoted by Stroud :cite:p:`stroud1971` as formulae 
for $E^{r^2}_d$.


In principal, via an affine transformation, one can convert these formulae to be valid 
for any Gaussian measure (with arbitrary mean and covariance). However, at present, 
this package does not provide support for such transformations.

.. autosummary::
    :nosignatures:

    GaussianCubature
    Hadamard
    StroudSecrest63_31
    StroudSecrest63_32
    StroudSecrest63_52

.. autoclass:: AbstractGaussianCubature
    :members:


.. autoclass:: Hadamard
    :members:

    .. autoattribute:: degree
    .. autoattribute:: sparse

.. autoclass:: StroudSecrest63_31
    :members:

    .. autoattribute:: degree
    .. autoattribute:: sparse

.. autoclass:: StroudSecrest63_32
    :members:

    .. autoattribute:: degree
    .. autoattribute:: sparse

.. autoclass:: StroudSecrest63_52
    :members:

    .. autoattribute:: degree
    .. autoattribute:: sparse


Wiener Cubature
---------------

The following formulae define single-step cubatures on Wiener space, as first described 
in :cite:t:`lyons2004`. For consistency of terminology, we instead refer to a 
cubature on Wiener space as a Wiener cubature.

We currently only support linear-gaussian formulae of the form in proposition 5.2 of 
:cite:p`lyons2004`. More advanced algebraic construction techniques, which may yield 
more efficient formulae, are not currently implemented.

