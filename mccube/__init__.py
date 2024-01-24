import importlib.metadata

# Cubature Library
from ._formulae import (
    AbstractCubature as AbstractCubature,
    AbstractGaussianCubature as AbstractGaussianCubature,
    Hadamard as Hadamard,
    search_cubature_registry as search_cubature_registry,
    StroudSecrest63_31 as StroudSecrest63_31,
    StroudSecrest63_32 as StroudSecrest63_32,
    StroudSecrest63_52 as StroudSecrest63_52,
    StroudSecrest63_53 as StroudSecrest63_53,
)

# Kernel Library
from ._kernels import (
    AbstractKernel as AbstractKernel,
    AbstractPartitioningKernel as AbstractPartitioningKernel,
    AbstractRecombinationKernel as AbstractRecombinationKernel,
    BinaryTreePartitioningKernel as BinaryTreePartitioningKernel,
    MonteCarloKernel as MonteCarloKernel,
    MonteCarloPartitioningKernel as MonteCarloPartitioningKernel,
    PartitioningRecombinationKernel as PartitioningRecombinationKernel,
    StratifiedPartitioningKernel as StratifiedPartitioningKernel,
)

# Metrics
from ._metrics import (
    center_of_mass as center_of_mass,
    gaussian_wasserstein_metric as gaussian_wasserstein_metric,
    pairwise_metric as pairwise_metric,
)

# MCC Solver
from ._path import (
    AbstractCubaturePath as AbstractCubaturePath,
    LocalLinearCubaturePath as LocalLinearCubaturePath,
)
from ._regions import (
    AbstractRegion as AbstractRegion,
    GaussianRegion as GaussianRegion,
)
from ._solvers import MCCSolver as MCCSolver
from ._term import MCCTerm as MCCTerm

# Utils
from ._utils import (
    all_subclasses as all_subclasses,
    pack_particles as pack_particles,
    unpack_particles as unpack_particles,
)

__version__ = importlib.metadata.version("mccube")
