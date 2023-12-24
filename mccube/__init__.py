import importlib.metadata

# MCC Framework
from ._diffrax import MCCubatureSolver as MCCubatureSolver
from ._inference import (
    mccubaturesolve as mccubaturesolve,
    MCCubatureState as MCCubatureState,
)
from ._kernels import (
    AbstractKernel as AbstractKernel,
    AbstractRecombinationKernel as AbstractRecombinationKernel,
    OverdampedLangevinKernel as OverdampedLangevinKernel,
    MonteCarloKernel as MonteCarloKernel,
)

# Cubature Library
from ._formulae import (
    AbstractCubature as AbstractCubature,
    AbstractGaussianCubature as AbstractGaussianCubature,
    AbstractWienerCubature as AbstractWienerCubature,
    Hadamard as Hadamard,
    StroudSecrest63_31 as StroudSecrest63_31,
    StroudSecrest63_32 as StroudSecrest63_32,
    StroudSecrest63_52 as StroudSecrest63_52,
    StroudSecrest63_53 as StroudSecrest63_53,
    LyonsVictoir04_512 as LyonsVictoir04_512,
)
from ._regions import (
    AbstractRegion as AbstractRegion,
    GaussianRegion as GaussianRegion,
    WienerSpace as WienerSpace,
)

__version__ = importlib.metadata.version("mccube")
