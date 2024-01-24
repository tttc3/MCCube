# API Reference

A complete listing of all parts of the A complete listing of all parts of the MCCube API
can be found below:

## Markov chain cubature framework
These sub-modules form the core of MCCube, providing an integration with [`diffrax.diffeqsolve`][].

- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._term`](_term.md)
- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._path`](_path.md)
- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._solvers`](_solvers.md)

## Kernel library
These sub-modules are helpful for defining (partitioned) recombination kernels, required
in Markov chain cubature.

- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._kernels.base`](_kernels/base.md)
- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._kernels.random`](_kernels/random.md)
- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._kernels.stratified`](_kernels/stratified.md)
- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._kernels.tree`](_kernels/tree.md)

## Cubature libraray
These sub-modules define standard cubature formulae, and are helpful in the construction
of control paths, which define cubatures on Wiener space.

- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._formulae`](_formulae.md)
- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._regions`](_regions.md)

## Miscellaneous
These sub-modules, while central to the design and operation of this package, are 
likely to be of little interest to most practitioners.

- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._custom_types`](_custom_types.md)
- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._utils`](_utils.md)
- <code class="doc-symbol doc-symbol-nav doc-symbol-module"></code> [`mccube._metrics`](_metrics.md)

