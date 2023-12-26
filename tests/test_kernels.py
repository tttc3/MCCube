import chex
import jax.numpy as jnp
from absl.testing import absltest

from mccube._custom_types import Args, P, RealScalarLike, XP
from mccube._kernels.base import AbstractRecombinationKernel


class TestKernels(chex.TestCase):
    def test_recombination_kernel(self):
        # Only need to test invalid as valid is implicitly tested in the concrete
        # implementations above.
        class InvalidKernel(AbstractRecombinationKernel):
            def transform(
                self, t0: RealScalarLike, t1: RealScalarLike, particles: XP, args: Args
            ) -> P:
                return particles[:2]

        class InvalidKernel2(AbstractRecombinationKernel):
            def transform(
                self, t0: RealScalarLike, t1: RealScalarLike, particles: XP, args: Args
            ) -> P:
                return jnp.stack([particles, particles])

        test_particles = jnp.ones((10, 2, 15))
        with self.assertRaises(AssertionError):
            InvalidKernel((3, 2, 15))(0.0, 1.0, test_particles, None)

        with self.assertRaises(AssertionError):
            InvalidKernel2((20, 2, 15))(0.0, 1.0, test_particles, None)


if __name__ == "__main__":
    absltest.main()
