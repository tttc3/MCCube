r"""Components that are measure-integration respecting maps on discrete measures.

Recombinators are specialized components that accept a modified call signature 
$h(r_f, t, p(t), args)$, where $r_f \in \mathbb{R}$ represents some recombination factor, 
and which have a validator $v$ which ensures the transformation $f$ obeys the following 
properties:

1. The transformation **must return** the recombinant particles $p^{\prime}(t)$ as a **rank-two tensor** (matrix).
2. The transformation **must not change the particle dimension** $d$. 
3. The transformation **must not increase the particle count** ($m \ge n$).

.. admonition:: Is a Recombinator just a Compressor?
    :class: note

    A Recombinator is a strictly more general component than a Compressor, as it 
    doesn't imply that the transformation aims to minimize some information 
    discrepancy/loss.

.. admonition:: The importance of Recombinators in cubature
    :class: important

    A Recombinator can be considered as a map on discrete measures which respects
    measure-integration $\bigg|\int_{\Omega} f(x_i) d\mu(x_i) - \int_{\Omega} f(x_j) d\nu(x_j)\bigg|$
    where the elements of the support of the two measures can be disjoint, the
    support of $\nu$ has cardinality
    $\lfloor \text{card}(\text{supp}(\mu)) / r_f \rfloor \le \text{card}(\text{supp}(\mu))$,
    and $i$ and $j$ index each co-ordinate dimension of the particle $x$.
    The support of $\nu$ is the recombinant particles $p^{\prime}(t)$, and would ideally (but not 
    neccessarily) be selected to minimize the above integral metric, while the support 
    of $\mu$ is simply the input particles $p(t)$. Each concrete implementation of a 
    Recombinator implicitly specifies the form of $f$, $\nu$ and $\mu$ via its 
    :meth:`AbstractComponent.transform() <mccube.components.base.AbstractComponent.transform>` method.

A typical usage pattern in this package is to compose a Recombinator and a Propagator
(see :class:`MCCubatureStep <mccube.cubature.MCCubatureStep>`). This is important in cases 
where repeated application of a Propagator could lead to particle count explosion. In 
such a case, the Recombinator is usually designed to `best` represent the expanded 
Propagated particles, by minimizing some integral metric, such as the one presented above.
"""  # noqa: E501

from mccube.components.recombinators.base import (
    AbstractRecombinator,
    WrappedRecombinator,
)
from mccube.components.recombinators.random import MonteCarloRecombinator

__all__ = [
    "AbstractRecombinator",
    "WrappedRecombinator",
    "MonteCarloRecombinator",
]
