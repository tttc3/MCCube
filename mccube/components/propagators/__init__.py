r"""Components that are discrete time-integrators.

Propagators are specialized components that accept a modified call signature of the form 
$h(g, t, p(t), args)$, where $g: (t, p(t), args) \to p^{\prime}(t)$ represents some 
standard component/suitable-callable, and which have a validator $v$ which ensures the 
transformation $f$ obeys the following properties:

1. The transformation **must return** the propagated particles $p^{\prime}(t)$ as a **rank-two tensor** (matrix).
2. The transformation **must not change the particle dimension** $d$. 
3. The transformation **must not reduce the particle count** ($m \ge n$).

.. admonition:: The importance of propagators in cubature
    :class: important

    A Propagator can be considered as a discrete time-integration step for a (stochastic)
    differential equation or some integral $\int h(g, t, p(t), args) \mathrm{d}W(t)$. 
    In this package, the integrand $h$ is usually an SDE of a fixed form, $g$ is an 
    interaction potential (log-density function), and $W(t)$ is a collection of one or 
    more Brownian motion paths, where each concrete implementation of a Propagator 
    implicitly specifies the form of $h$ and $W(t)$ via its `transform` method.
"""  # noqa: E501

from mccube.components.propagators.base import AbstractPropagator, WrappedPropagator
from mccube.components.propagators.diffusions import LangevinDiffusionPropagator

__all__ = ["AbstractPropagator", "WrappedPropagator", "LangevinDiffusionPropagator"]
