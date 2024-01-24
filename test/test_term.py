import diffrax
import equinox as eqx
import jax.numpy as jnp

import mccube


def test_mcc_term():
    def ode_vector_field(t, y, args):
        return {"y": -y["y"]}

    def cde_vector_field(t, y, args):
        return {"y": 1.0}

    class Control(diffrax.AbstractPath):
        t0 = 0
        t1 = 1

        def evaluate(self, t0, t1=None, left=True):
            return {"y": jnp.ones((8, 2))}

        def derivative(self, t, left=True):
            return {"y": jnp.zeros((8, 2))}

    control = Control()
    ode = diffrax.ODETerm(ode_vector_field)
    cde = diffrax.WeaklyDiagonalControlTerm(cde_vector_field, control)
    term = mccube.MCCTerm(ode, cde)
    args = None
    dx = term.contr(0, 1)
    y = {"y": jnp.array([[1.0, 2.0], [2.0, 4.0], [5.0, 6.0]])[:, None, :]}
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dx)
    assert jnp.shape(dx[0]) == () and jnp.shape(dx[1]["y"]) == (8, 2)
    assert jnp.shape(vf[0]["y"]) == (3, 2) and jnp.shape(vf[1]["y"]) == ()
    assert eqx.tree_equal(vf, ({"y": -y["y"].squeeze()}, {"y": 1.0}))
    assert jnp.shape(vf_prod["y"]) == (3, 8, 2)
    assert eqx.tree_equal(vf_prod, term.prod(vf, dx), rtol=1e-5, atol=1e-8)

    y = vf_prod
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dx)
    assert jnp.shape(vf[0]["y"]) == (3 * 8, 2)
    assert eqx.tree_equal(
        vf, ({"y": -y["y"].reshape(-1, y["y"].shape[-1])}, {"y": 1.0})
    )
    assert jnp.shape(vf_prod["y"]) == (3 * 8, 8, 2)
    assert eqx.tree_equal(vf_prod, term.prod(vf, dx), rtol=1e-5, atol=1e-8)
