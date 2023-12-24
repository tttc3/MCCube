:tocdepth: 3

``mccube._kernels`` module
==========================

.. automodule:: mccube._kernels
.. currentmodule:: mccube._kernels

Standard kernels
----------------

The following abstract and concrete implementations of standard kernels are provided:

.. autosummary::
    :nosignatures:

    AbstractKernel
    OverdampedLangevinKernel

Non-standard kernels are discussed in other subsections of this article.

.. autoclass:: AbstractKernel
    :members:

.. autoclass:: OverdampedLangevinKernel

Recombination kernels
---------------------

The :class:`AbstractRecombinationKernel` is central to the practical application of 
Markov Chain Cubature. At present the following abstract and concrete implementations 
are provided:

.. autosummary::
    :nosignatures:

    AbstractRecombinationKernel
    MonteCarloKernel


.. autoclass:: AbstractRecombinationKernel
    :members: validate

.. autoclass:: MonteCarloKernel