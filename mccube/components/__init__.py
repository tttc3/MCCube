r"""Standard components useful in the construction of MCCubature rules.

The primary purpose of the components API is to provide a convenient object-oriented 
interface for defining arbitrary mathematical transformations that are expected to obey 
certain transformation properties. In this package, these benefits are realized through:
    
* **Type Hinting**: when a callable is expected to have a certain call signature, and to 
  obey certain transformation properties, a suitable component is used as a type hint.

* **Transform Validation**: components consist of a transform and a validate method. The 
  validate method wraps the transform and allows for trace time validation of most
  transform properties (via Chex).

The latter case is particularly useful for developers wishing to implement new
specialized components as subclasses of existing ones, as they need only be concerned 
with defining a transform method that satisfies the existing validate method.

.. admonition:: What if I want a purely functional approach?
  :class: tip

  There is no obligation to use the component classes for any of the core functionality 
  in this package. If you want a purely functional approach, you can simply provide a 
  suitable callable (with a call and return signature identical to the expected 
  Component). However, in this case, the validity of the callable (with respect to the 
  expected component) is not checked. 

In the most general sense, a component can be mathematically defined as the composition 
$h = v \circ f$ of a transformation function $f: (t, p(t), args) \to p^{\prime}(t)$, 
where $t \in \mathbb{R}$ is a particle existence time, $p(t) \in \mathbb{R}^{n \times d}$ 
is the state of $n$ $d\text{-dimensional}$ particles at time $t$, $args$ is a set 
of arbitrary values that parameterise $f$, and $p^{\prime}(t) \in \mathbb{R}^{m \times d}$
are the transformed state of $m$ $d\text{-dimensional}$ particles; and a validation 
operator $v: f \to \{f, \varnothing\}$, with $\varnothing$ only returned if $f$ is not 
a valid transformation.

Note: 
  The particles could be considered as either a rank-two tensor ($n \times d$ matrix) or 
  as a set of $n$ rank-one tensors ($d\text{-vectors}$ $p(t) = \{p_i : p \in \mathbb{R^d}, i=1,\dots,n\}$.
  In this package the matrix form is always assumed unless explicitly stated otherwise.

Warning:
  Some specialized components modify the call-signature/domain of the transformation $f$
  and subsequently the composition $h$, from that presented above. In addition, while 
  all components are documented as accepting PyTrees of inputs, at present PyTrees are
  only fully supported for the specialized components in :mod:`mccube.components.recombinators`.
"""  # noqa: E501

from mccube.components.base import AbstractComponent, Component
from mccube.components.propagators import (
    AbstractPropagator,
    WrappedPropagator,
    LangevinDiffusionPropagator,
)
from mccube.components.recombinators import (
    AbstractRecombinator,
    WrappedRecombinator,
    MonteCarloRecombinator,
)


__all__ = [
    "AbstractComponent",
    "Component",
    "AbstractPropagator",
    "WrappedPropagator",
    "LangevinDiffusionPropagator",
    "AbstractRecombinator",
    "WrappedRecombinator",
    "MonteCarloRecombinator",
]
