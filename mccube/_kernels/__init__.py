r"""This module provides the tools for constructing classes of Markov transition kernels 
which obey certain desired properties, along with a library of useful kernels for Markov 
Chain Cubature.
"""

from .base import (
    AbstractKernel as AbstractKernel,
    AbstractRecombinationKernel as AbstractRecombinationKernel,
)

from .diffusions import OverdampedLangevinKernel as OverdampedLangevinKernel

from .random import MonteCarloKernel as MonteCarloKernel
