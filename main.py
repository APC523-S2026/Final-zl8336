import jax
jax.config.update("jax_enable_x64", True)
try:
    jax.config.update('jax_platform_name', 'gpu')
    _ = jax.devices('gpu')
    print("Using GPU:", jax.devices('gpu'), flush=True)
except RuntimeError:
    jax.config.update('jax_platform_name', 'cpu')
    print("GPU not available, using CPU.", flush=True)
print(jax.local_device_count(), flush=True)
print(jax.devices(), flush=True)

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Callable, Optional

from diffrax import (
    diffeqsolve, ODETerm, MultiTerm,
    Dopri5, Tsit5, Heun, KenCarp4, SemiImplicitEuler,
    SaveAt, PIDController, ConstantStepSize,
)

# ─────────────────────────────────────────────────────────────────────────────
# Solver registry
# ─────────────────────────────────────────────────────────────────────────────

SOLVER_DICT = {
    'Dopri5':            Dopri5,
    'Tsit5':             Tsit5,
    'Heun':              Heun,
    'KenCarp4':          KenCarp4,
    'SemiImplicitEuler': SemiImplicitEuler,
}
_IMEX_SOLVERS       = frozenset({'KenCarp4'})
_SYMPLECTIC_SOLVERS = frozenset({'SemiImplicitEuler'})


# ─────────────────────────────────────────────────────────────────────────────
# Module-level utility functions
# ─────────────────────────────────────────────────────────────────────────────

def make_kz(eps_val: complex, kx: float, k0: float) -> complex:
    """
    Complex kz with Im(kz) ≥ 0 (evanescent waves decay in +z).
    kz = sqrt(k0²·ε − kx² + 0j)
    """
    val = complex(k0**2 * eps_val - kx**2)
    kz = np.sqrt(val + 0j)
    # Enforce Im(kz) ≥ 0 so that exp(-j·kz·z) decays for evanescent modes
    if np.imag(kz) < 0:
        kz = -kz
    return kz


def build_interp_arrays(
    eps_func: Callable,
    z1: float,
    z2: float,
    n: int = 10000,
) -> tuple:
    """
    Pre-sample eps_func on [z1, z2] and compute dε/dz numerically.
    Returns JAX arrays (z_grid, eps_grid, deps_dz_grid).
    z_grid is strictly increasing (required by jnp.interp).
    """
    z_grid = jnp.linspace(z1, z2, n)
    eps_vals = jnp.array([complex(eps_func(float(z))) for z in z_grid])
    deps_dz = jnp.gradient(eps_vals, z_grid)
    return z_grid, eps_vals, deps_dz


def ode_rhs_te(t, y, args):
    """
    diffrax-compatible TE ODE RHS.  State y = [Re(E), Im(E), Re(dE), Im(dE)].
    args = (z_grid, Re(eps_grid), Im(eps_grid), kx, k0)
    TE:  d²E/dz² = -(k0²·ε(z) − kx²)·E
    """
    z_grid, eps_re, eps_im, kx, k0 = args
    eps_r = jnp.interp(t, z_grid, eps_re)
    eps_i = jnp.interp(t, z_grid, eps_im)
    keff2_re = k0**2 * eps_r - kx**2
    keff2_im = k0**2 * eps_i
    # E = y[0]+j*y[1],  dE = y[2]+j*y[3]
    # d(dE)/dz = -(keff2_re + j*keff2_im)*(y[0]+j*y[1])
    #          = -(keff2_re*y[0] - keff2_im*y[1]) - j*(keff2_re*y[1] + keff2_im*y[0])
    dE_re_new = -(keff2_re * y[0] - keff2_im * y[1])
    dE_im_new = -(keff2_re * y[1] + keff2_im * y[0])
    return jnp.array([y[2], y[3], dE_re_new, dE_im_new])


def ode_rhs_tm(t, y, args):
    """
    diffrax-compatible TM ODE RHS.  State y = [Re(H), Im(H), Re(dH), Im(dH)].
    args = (z_grid, Re(eps_grid), Im(eps_grid), Re(deps_dz), Im(deps_dz), kx, k0)
    TM:  d²H/dz² = (1/ε)·(dε/dz)·dH/dz − [k0²·ε − kx²]·H
    drift = (1/ε)·dε/dz  (complex division)
    """
    z_grid, eps_re, eps_im, deps_re, deps_im, kx, k0 = args
    eps_r = jnp.interp(t, z_grid, eps_re)
    eps_i = jnp.interp(t, z_grid, eps_im)
    de_r  = jnp.interp(t, z_grid, deps_re)
    de_i  = jnp.interp(t, z_grid, deps_im)
    # drift = (de_r + j*de_i) / (eps_r + j*eps_i)
    eps_abs2 = eps_r**2 + eps_i**2
    drift_re = (de_r * eps_r + de_i * eps_i) / eps_abs2
    drift_im = (de_i * eps_r - de_r * eps_i) / eps_abs2
    keff2_re = k0**2 * eps_r - kx**2
    keff2_im = k0**2 * eps_i
    # d(dH)/dz = drift*(dH) - keff2*H
    # = (drift_re+j*drift_im)*(y[2]+j*y[3]) - (keff2_re+j*keff2_im)*(y[0]+j*y[1])
    dH_re_new = (drift_re*y[2] - drift_im*y[3]) - (keff2_re*y[0] - keff2_im*y[1])
    dH_im_new = (drift_re*y[3] + drift_im*y[2]) - (keff2_re*y[1] + keff2_im*y[0])
    return jnp.array([y[2], y[3], dH_re_new, dH_im_new])


# ─── IMEX split RHS for TE ────────────────────────────────────────────────────
# Explicit part: kinematic only (dF/dz = state derivative, no potential)
def ode_rhs_te_explicit(t, y, args):
    return jnp.array([y[2], y[3], 0.0, 0.0])


# Implicit part: stiff oscillatory potential  d²F/dz² = −keff²(z)·F
def ode_rhs_te_implicit(t, y, args):
    z_grid, eps_re, eps_im, kx, k0 = args
    eps_r = jnp.interp(t, z_grid, eps_re)
    eps_i = jnp.interp(t, z_grid, eps_im)
    keff2_re = k0**2 * eps_r - kx**2
    keff2_im = k0**2 * eps_i
    return jnp.array([0.0, 0.0,
                       -(keff2_re * y[0] - keff2_im * y[1]),
                       -(keff2_re * y[1] + keff2_im * y[0])])


# ─── IMEX split RHS for TM ────────────────────────────────────────────────────
# Explicit part: kinematic + drift term  (non-stiff)
def ode_rhs_tm_explicit(t, y, args):
    z_grid, eps_re, eps_im, deps_re, deps_im, kx, k0 = args
    eps_r = jnp.interp(t, z_grid, eps_re)
    eps_i = jnp.interp(t, z_grid, eps_im)
    de_r  = jnp.interp(t, z_grid, deps_re)
    de_i  = jnp.interp(t, z_grid, deps_im)
    eps_abs2 = eps_r**2 + eps_i**2
    drift_re = (de_r * eps_r + de_i * eps_i) / eps_abs2
    drift_im = (de_i * eps_r - de_r * eps_i) / eps_abs2
    return jnp.array([y[2], y[3],
                       drift_re * y[2] - drift_im * y[3],
                       drift_re * y[3] + drift_im * y[2]])


# Implicit part: stiff oscillatory potential only
def ode_rhs_tm_implicit(t, y, args):
    z_grid, eps_re, eps_im, deps_re, deps_im, kx, k0 = args
    eps_r = jnp.interp(t, z_grid, eps_re)
    eps_i = jnp.interp(t, z_grid, eps_im)
    keff2_re = k0**2 * eps_r - kx**2
    keff2_im = k0**2 * eps_i
    return jnp.array([0.0, 0.0,
                       -(keff2_re * y[0] - keff2_im * y[1]),
                       -(keff2_re * y[1] + keff2_im * y[0])])


def extract_rt(F_z1: complex, dF_z1: complex, kz1: complex) -> tuple:
    """
    Decompose field at z1 into forward (A) and backward (B) amplitudes.
    Convention: right-going wave ~ exp(-j·kz·z).
      F(z) = A·exp(-j·kz1·z) + B·exp(+j·kz1·z)
      dF/dz = -j·kz1·A·exp(-j·kz1·z) + j·kz1·B·exp(+j·kz1·z)
    → A_tilde = 0.5*(F + j·dF/kz1),  B_tilde = 0.5*(F - j·dF/kz1)
    r = B_tilde / A_tilde,   t = 1 / A_tilde
    """
    A = 0.5 * (F_z1 + 1j * dF_z1 / kz1)
    B = 0.5 * (F_z1 - 1j * dF_z1 / kz1)
    r = B / A
    t = 1.0 / A
    return complex(r), complex(t)


@partial(jit, static_argnames=('pol', 'order'))
def assemble_and_solve_fem(
    nodes:    jnp.ndarray,   # (N_nodes,) float64  — P1 node positions
    eps_vals: jnp.ndarray,   # (N_nodes,) complex128 — ε at P1 nodes
    k0:    float,
    kx:    float,
    kz1:   complex,
    eps1:  complex,
    kz2:   complex,
    eps2:  complex,
    n_ext: int,
    pol:   str,              # static: 'S' (TE) or 'P' (TM)
    order: int = 1,          # static: 1=P1 linear, 2=P2 quadratic
) -> jnp.ndarray:
    """
    JAX-native FEM assembly + dense solve.  GPU-ready and fully differentiable.

    order=1 (P1 linear elements):
      TE  K_el = (1/h)·[[1,−1],[−1,1]]        M_el = (k0²εm−kx²)·h/6·[[2,1],[1,2]]
      TM  K_el = (1/(εmh))·[[1,−1],[−1,1]]    M_el = (k0²−kx²/εm)·h/6·[[2,1],[1,2]]

    order=2 (P2 quadratic elements, midpoints inserted between P1 nodes):
      shape functions φ₁=(2ξ−1)(ξ−1), φ₂=4ξ(1−ξ), φ₃=ξ(2ξ−1) on [0,1]
      TE  K_el = (1/h)·[[7/3,−8/3,1/3],[−8/3,16/3,−8/3],[1/3,−8/3,7/3]]
          M_el = coeff·h/30·[[4,2,−1],[2,16,2],[−1,2,4]]
      TM  same with k_fac=1/(εm·h) and coeff=k0²−kx²/εm

    P2 returns a field vector of length 2·N_nodes−1 (P1 nodes interleaved with midpoints).
    Assembly: vectorized scatter-add with .at[].add() — no Python loops.
    Solve:    jnp.linalg.solve (dense LU on GPU, differentiable via IFT).
    """
    num_nodes_p1 = nodes.shape[0]
    num_el       = num_nodes_p1 - 1
    el_idx       = jnp.arange(num_el)

    h       = nodes[1:] - nodes[:-1]                    # (n_el,) element widths
    eps_mid = 0.5 * (eps_vals[:-1] + eps_vals[1:])      # (n_el,) midpoint ε

    if order == 1:
        # ── P1 linear elements ──────────────────────────────────────────────
        if pol == 'S':   # TE
            k_dd  =  1.0 / h
            k_od  = -1.0 / h
            coeff = k0**2 * eps_mid - kx**2
        else:            # TM
            k_dd  =  1.0 / (eps_mid * h)
            k_od  = -1.0 / (eps_mid * h)
            coeff = k0**2 - kx**2 / eps_mid

        a_dd = k_dd - coeff * h / 3.0
        a_od = k_od - coeff * h / 6.0

        A = jnp.zeros((num_nodes_p1, num_nodes_p1), dtype=jnp.complex128)
        A = A.at[el_idx,     el_idx    ].add(a_dd)
        A = A.at[el_idx + 1, el_idx + 1].add(a_dd)
        A = A.at[el_idx,     el_idx + 1].add(a_od)
        A = A.at[el_idx + 1, el_idx    ].add(a_od)

        b = jnp.zeros(num_nodes_p1, dtype=jnp.complex128)
        F_inc_left = jnp.exp(-1j * kz1 * (nodes[0] - nodes[n_ext]))
        if pol == 'S':
            A = A.at[0,  0 ].add(1j * kz1)
            b = b.at[0     ].add(2j * kz1 * F_inc_left)
            A = A.at[-1, -1].add(1j * jnp.conj(kz2))
        else:
            A = A.at[0,  0 ].add(1j * kz1 / eps1)
            b = b.at[0     ].add(2j * kz1 / eps1 * F_inc_left)
            A = A.at[-1, -1].add(1j * jnp.conj(kz2) / eps2)

    else:
        # ── P2 quadratic elements — midpoints inserted between P1 nodes ─────
        # P2 global indices: vertex nodes at 2e, 2e+2; midpoint at 2e+1
        num_nodes_p2 = 2 * num_nodes_p1 - 1
        idx_L = 2 * el_idx          # left vertex
        idx_M = 2 * el_idx + 1      # midpoint
        idx_R = 2 * el_idx + 2      # right vertex (= idx_L of next element)

        if pol == 'S':   # TE
            k_fac = 1.0 / h
            coeff = k0**2 * eps_mid - kx**2
        else:            # TM
            k_fac = 1.0 / (eps_mid * h)
            coeff = k0**2 - kx**2 / eps_mid

        # P2 element matrix entries  A_el = K_el − M_el
        # K_hat = [[7/3,−8/3,1/3],[−8/3,16/3,−8/3],[1/3,−8/3,7/3]]  (×k_fac)
        # M_hat = [[4,2,−1],[2,16,2],[−1,2,4]] / 30                   (×coeff·h)
        a_LL = 7.0/3.0 * k_fac - 4.0/30.0  * coeff * h
        a_LM = -8.0/3.0 * k_fac - 2.0/30.0 * coeff * h
        a_LR = 1.0/3.0 * k_fac  + 1.0/30.0 * coeff * h   # M[0,2]=−1 → -M = +1
        a_MM = 16.0/3.0 * k_fac - 16.0/30.0 * coeff * h
        a_MR = -8.0/3.0 * k_fac - 2.0/30.0 * coeff * h
        # a_RR = a_LL, a_RL = a_LR, a_RM = a_MR  (symmetric)

        A = jnp.zeros((num_nodes_p2, num_nodes_p2), dtype=jnp.complex128)
        # Diagonal blocks
        A = A.at[idx_L, idx_L].add(a_LL)
        A = A.at[idx_M, idx_M].add(a_MM)
        A = A.at[idx_R, idx_R].add(a_LL)   # a_RR = a_LL
        # Off-diagonal: L–M
        A = A.at[idx_L, idx_M].add(a_LM)
        A = A.at[idx_M, idx_L].add(a_LM)
        # Off-diagonal: L–R
        A = A.at[idx_L, idx_R].add(a_LR)
        A = A.at[idx_R, idx_L].add(a_LR)
        # Off-diagonal: M–R
        A = A.at[idx_M, idx_R].add(a_MR)
        A = A.at[idx_R, idx_M].add(a_MR)

        b = jnp.zeros(num_nodes_p2, dtype=jnp.complex128)
        # Robin BCs at P2 boundary nodes (indices 0 and -1 = 2*num_el)
        F_inc_left = jnp.exp(-1j * kz1 * (nodes[0] - nodes[n_ext]))
        if pol == 'S':
            A = A.at[0,  0 ].add(1j * kz1)
            b = b.at[0     ].add(2j * kz1 * F_inc_left)
            A = A.at[-1, -1].add(1j * jnp.conj(kz2))
        else:
            A = A.at[0,  0 ].add(1j * kz1 / eps1)
            b = b.at[0     ].add(2j * kz1 / eps1 * F_inc_left)
            A = A.at[-1, -1].add(1j * jnp.conj(kz2) / eps2)

    return jnp.linalg.solve(A, b)


# ─────────────────────────────────────────────────────────────────────────────
# JIT-compiled ODE core (shared by solve_ode and sweep_ode)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jit, static_argnames=('pol', 'n_trans', 'solver_name', 'fixed_step'))
def _ode_solve_core(
    z_grid,
    eps_re, eps_im,
    deps_re, deps_im,
    kx, k0,
    kz2_re, kz2_im,
    z1, z2,
    n_trans,
    pol,
    solver_name='Dopri5',
    fixed_step=False,
    rtol=1e-8,
    atol=1e-10,
):
    """
    JIT-compiled ODE shooting core: integrates from z2 → z1.

    Returns ``sol.ys`` of shape ``(n_trans, 4)`` ordered z2 → z1,
    where columns are [Re(F), Im(F), Re(dF/dz), Im(dF/dz)].

    Parameters
    ----------
    solver_name : str (static)
        One of ``SOLVER_DICT`` keys.  Each distinct value produces a
        separate compiled kernel.
    fixed_step : bool (static)
        Use ``ConstantStepSize`` instead of ``PIDController``.
        Always ``True`` for ``SemiImplicitEuler`` (which has no error
        estimator).
    """
    dt0    = -(z2 - z1) / n_trans
    z_save = jnp.linspace(z2, z1, n_trans)

    # IC at z2: decaying transmitted wave F(z2)=1, dF/dz(z2) = −j·conj(kz2)
    #   Re(dF/dz) = −Im(kz2),  Im(dF/dz) = −Re(kz2)
    y0_flat = jnp.array([1.0, 0.0, -kz2_im, -kz2_re])

    # SemiImplicitEuler has no error estimator → always fixed step.
    # KenCarp4 (implicit) requires PIDController to pass tolerances to Newton solver.
    # Other solvers: respect caller's fixed_step flag.
    _use_fixed = (solver_name in _SYMPLECTIC_SOLVERS or
                  (fixed_step and solver_name not in _IMEX_SOLVERS))
    controller = ConstantStepSize() if _use_fixed else PIDController(rtol=rtol, atol=atol)

    # ── SemiImplicitEuler: symplectic split with tuple state ──────────────────
    if solver_name in _SYMPLECTIC_SOLVERS:
        # y = (pos=(E_re, E_im),  vel=(dE_re, dE_im))
        # step 1: pos_new = pos + h·vel_old   (f_vel returns vel)
        # step 2: vel_new = vel + h·(-keff²·pos_new + drift·vel_old)
        #         For TM the drift·vel term is approximated at step 2 using vel_old
        #         baked into explicit; only the -keff² potential is truly implicit here.
        if pol == 'S':
            args = (z_grid, eps_re, eps_im, kx, k0)

            def f_vel(t, y_vel, _args):
                return y_vel

            def f_force(t, y_pos, _args):
                z_g, ep_r, ep_i, kx_, k0_ = _args
                er = jnp.interp(t, z_g, ep_r)
                ei = jnp.interp(t, z_g, ep_i)
                k2r = k0_**2 * er - kx_**2
                k2i = k0_**2 * ei
                return jnp.array([-(k2r * y_pos[0] - k2i * y_pos[1]),
                                   -(k2r * y_pos[1] + k2i * y_pos[0])])
        else:
            args = (z_grid, eps_re, eps_im, deps_re, deps_im, kx, k0)

            def f_vel(t, y_vel, _args):
                # kinematic + drift (explicit)
                z_g, ep_r, ep_i, de_r, de_i, kx_, k0_ = _args
                er = jnp.interp(t, z_g, ep_r)
                ei = jnp.interp(t, z_g, ep_i)
                der = jnp.interp(t, z_g, de_r)
                dei = jnp.interp(t, z_g, de_i)
                abs2 = er**2 + ei**2
                dr = (der * er + dei * ei) / abs2
                di = (dei * er - der * ei) / abs2
                return y_vel + jnp.array([dr * y_vel[0] - di * y_vel[1],
                                           dr * y_vel[1] + di * y_vel[0]])

            def f_force(t, y_pos, _args):
                # potential (implicit)
                z_g, ep_r, ep_i, de_r, de_i, kx_, k0_ = _args
                er = jnp.interp(t, z_g, ep_r)
                ei = jnp.interp(t, z_g, ep_i)
                k2r = k0_**2 * er - kx_**2
                k2i = k0_**2 * ei
                return jnp.array([-(k2r * y_pos[0] - k2i * y_pos[1]),
                                   -(k2r * y_pos[1] + k2i * y_pos[0])])

        y0_sym = (y0_flat[:2], y0_flat[2:])
        sol = diffeqsolve(
            (ODETerm(f_vel), ODETerm(f_force)),
            SemiImplicitEuler(),
            t0=z2, t1=z1, dt0=dt0,
            y0=y0_sym, args=args,
            saveat=SaveAt(ts=z_save),
            stepsize_controller=ConstantStepSize(),
            max_steps=n_trans * 2,
        )
        pos_ys, vel_ys = sol.ys          # each (n_trans, 2)
        ys = jnp.concatenate([pos_ys, vel_ys], axis=-1)   # (n_trans, 4)

    # ── KenCarp4: IMEX ARK via MultiTerm with flat state ─────────────────────
    elif solver_name in _IMEX_SOLVERS:
        if pol == 'S':
            args = (z_grid, eps_re, eps_im, kx, k0)
            term = MultiTerm(ODETerm(ode_rhs_te_explicit),
                             ODETerm(ode_rhs_te_implicit))
        else:
            args = (z_grid, eps_re, eps_im, deps_re, deps_im, kx, k0)
            term = MultiTerm(ODETerm(ode_rhs_tm_explicit),
                             ODETerm(ode_rhs_tm_implicit))
        sol = diffeqsolve(
            term, SOLVER_DICT[solver_name](),
            t0=z2, t1=z1, dt0=dt0,
            y0=y0_flat, args=args,
            saveat=SaveAt(ts=z_save),
            stepsize_controller=controller,
            max_steps=2_000_000,
        )
        ys = sol.ys   # (n_trans, 4)

    # ── Explicit ERK: Heun / Tsit5 / Dopri5 ──────────────────────────────────
    else:
        if pol == 'S':
            args = (z_grid, eps_re, eps_im, kx, k0)
            term = ODETerm(ode_rhs_te)
        else:
            args = (z_grid, eps_re, eps_im, deps_re, deps_im, kx, k0)
            term = ODETerm(ode_rhs_tm)
        sol = diffeqsolve(
            term, SOLVER_DICT[solver_name](),
            t0=z2, t1=z1, dt0=dt0,
            y0=y0_flat, args=args,
            saveat=SaveAt(ts=z_save),
            stepsize_controller=controller,
            max_steps=2_000_000,
        )
        ys = sol.ys   # (n_trans, 4)

    return ys


# ─────────────────────────────────────────────────────────────────────────────
# Solver class
# ─────────────────────────────────────────────────────────────────────────────

class HelmholtzSolver1D:
    """
    1D Helmholtz solver for EM wave propagation through a dielectric transition layer.

    Solves the reduced wave equation on the z-axis:
      TE: d²Ey/dz² + [k0²·ε(z) − kx²]·Ey = 0
      TM: d²Hy/dz² − (1/ε)·(dε/dz)·dHy/dz + [k0²·ε(z) − kx²]·Hy = 0

    Args:
        eps_func: callable z → complex ε_r(z). Constant outside [z1, z2].
        z1, z2: transition layer boundaries (z1 < z2).
        theta_deg: angle of incidence in degrees (from z-axis).
        pol: 'S' (TE) or 'P' (TM).
        lambda0: free-space wavelength (sets k0 = 2π/lambda0).
    """

    def __init__(
        self,
        eps_func: Callable,
        z1: float,
        z2: float,
        theta_deg: float = 0.0,
        pol: str = 'S',
        lambda0: float = 1.0,
    ):
        assert z1 < z2, "z1 must be less than z2."
        assert pol in ('S', 'P'), "pol must be 'S' (TE) or 'P' (TM)."
        assert 0.0 <= theta_deg < 90.0, "theta_deg must be in [0, 90)."

        self.eps_func  = eps_func
        self.z1        = float(z1)
        self.z2        = float(z2)
        self.theta_deg = float(theta_deg)
        self.pol       = pol
        self.lambda0   = float(lambda0)

        delta = 1e-10 * (z2 - z1)
        self.eps1 = complex(eps_func(z1 - delta))
        self.eps2 = complex(eps_func(z2 + delta))

        self.k0  = 2.0 * np.pi / lambda0
        self.kx  = self.k0 * np.sqrt(complex(self.eps1)) * np.sin(np.deg2rad(theta_deg))
        # enforce Im(kx) ≥ 0
        if np.imag(self.kx) < 0:
            self.kx = -self.kx
        self.kx = complex(self.kx)
        self.kz1 = make_kz(self.eps1, abs(self.kx), self.k0)
        self.kz2 = make_kz(self.eps2, abs(self.kx), self.k0)

        print(
            f"HelmholtzSolver1D: pol={pol}, θ={theta_deg}°, λ₀={lambda0}, "
            f"ε₁={self.eps1:.4g}, ε₂={self.eps2:.4g}, "
            f"kz1={self.kz1:.4g}, kz2={self.kz2:.4g}",
            flush=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Resolution helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _auto_res_trans(self, factor: float = 50.0, n_sample: int = 500) -> float:
        """Default transition resolution: factor × max√|εr(z)| over [z1, z2].

        ODE default factor = 50, FEM default factor = 100.
        """
        z_samp = np.linspace(self.z1, self.z2, n_sample)
        sqreps_max = max(abs(complex(self.eps_func(float(z)))) ** 0.5
                        for z in z_samp)
        return factor * sqreps_max

    def _auto_res_ext(self, res_trans: float) -> float:
        """Default exterior resolution: half of the transition resolution."""
        return res_trans / 2.0

    def _res_to_n(self, res: float, width: float) -> int:
        """Cells per λ₀ × physical width → number of grid points (min 10)."""
        return max(10, int(np.ceil(res * width / self.lambda0)))

    # ─────────────────────────────────────────────────────────────────────────
    # Method 1: ODE via diffrax
    # ─────────────────────────────────────────────────────────────────────────

    def solve_ode(
        self,
        res_trans: float = None,
        res_ext: float = None,
        L_ext_factor: float = 2.0,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        n_interp: int = 10000,
        solver_name: str = 'Dopri5',
        fixed_step: bool = False,
    ) -> dict:
        """
        Solve by backward ODE shooting (default: diffrax Dopri5, adaptive step).

        res_trans    : cells per λ₀ inside the transition layer [z1, z2].
                       Default = 50 × max√|εr(z)| over [z1, z2].
        res_ext      : cells per λ₀ in exterior plot regions (only affects sampling
                       density; ODE exterior is always analytical).
                       Default = res_trans / 2.
        L_ext_factor : width of each exterior plot region in units of λ₀.
                       The field is sampled on [z1 − L_ext_factor·λ₀, z2 + L_ext_factor·λ₀].
                       Default = 2.0, giving exterior regions of 2λ₀ on each side.
        solver_name  : key in ``SOLVER_DICT`` — 'Dopri5', 'Tsit5', 'Heun',
                       'KenCarp4', 'SemiImplicitEuler'.
        fixed_step   : force constant step size (always True for SemiImplicitEuler).

        Integration runs from z2 → z1 (transition region only).
        Exterior regions use analytical plane-wave expressions.

        Returns dict with keys: r, t, z, field, method, solver_name.
        """
        if res_trans is None:
            res_trans = self._auto_res_trans(factor=50.0)
        if res_ext is None:
            res_ext = self._auto_res_ext(res_trans)

        width   = self.z2 - self.z1
        n_trans = self._res_to_n(res_trans, width)

        print(
            f"  → ODE res_trans={res_trans:.2f}/λ₀ → n_trans={n_trans}",
            flush=True,
        )

        z1, z2 = self.z1, self.z2
        kz1, kz2 = self.kz1, self.kz2
        k0, kx = self.k0, self.kx
        L_ext = L_ext_factor * self.lambda0

        # Pre-sample ε over transition layer
        z_grid, eps_grid, deps_grid = build_interp_arrays(
            self.eps_func, z1, z2, n=n_interp
        )
        eps_re  = jnp.real(eps_grid)
        eps_im  = jnp.imag(eps_grid)
        deps_re = jnp.real(deps_grid)
        deps_im = jnp.imag(deps_grid)

        kx_real = float(np.real(kx))  # kx is real for lossless ε1
        kz2_re  = float(np.real(kz2))
        kz2_im  = float(np.imag(kz2))

        ys = _ode_solve_core(
            z_grid, eps_re, eps_im, deps_re, deps_im,
            kx_real, float(k0),
            kz2_re, kz2_im,
            float(z1), float(z2), n_trans, self.pol,
            solver_name=solver_name,
            fixed_step=fixed_step,
            rtol=rtol, atol=atol,
        )   # shape (n_trans, 4), ordered z2 → z1

        z_save = jnp.linspace(z2, z1, n_trans)

        # Field in transition from saved solution (z2→z1, flip to z1→z2)
        F_trans  = (ys[:, 0] + 1j * ys[:, 1])[::-1]
        z_trans  = z_save[::-1]

        # Extract r, t at z1 (last integration point = z1)
        F_z1  = complex(ys[-1, 0] + 1j * ys[-1, 1])
        dF_z1 = complex(ys[-1, 2] + 1j * ys[-1, 3])
        r, t  = extract_rt(F_z1, dF_z1, kz1)

        # Normalize: the ODE was initialized with F(z2)=1 (transmitted amplitude),
        # so the raw field has incident amplitude A_tilde = 1/t ≠ 1.
        # Dividing by A_tilde makes the incident amplitude equal 1 everywhere,
        # matching the exterior analytical expressions and removing boundary jumps.
        A_tilde = 0.5 * (F_z1 + 1j * dF_z1 / kz1)   # = 1/t
        F_trans = F_trans / A_tilde

        # Reconstruct field in exterior regions analytically.
        # Use the same step size as the transition region so the sampling density
        # is uniform across all three zones — eliminates visual kinks at z1/z2.
        dz     = (z2 - z1) / (n_trans - 1)
        z_left  = z1 - L_ext
        z_right = z2 + L_ext
        n_ext_l = int(np.ceil(L_ext / dz))
        n_ext_r = int(np.ceil(L_ext / dz))
        z_ext_l = jnp.linspace(z_left, z1 - dz, n_ext_l)
        z_ext_r = jnp.linspace(z2 + dz, z_right, n_ext_r)

        # Left exterior: incident + reflected (phase ref at z1)
        F_ext_l = (jnp.exp(-1j * kz1 * (z_ext_l - z1))
                   + r * jnp.exp(1j * kz1 * (z_ext_l - z1)))
        # Right exterior: transmitted (decaying evanescent), phase ref at z2
        F_ext_r = t * jnp.exp(-1j * np.conj(kz2) * (z_ext_r - z2))

        z_all = jnp.concatenate([z_ext_l, z_trans, z_ext_r])
        F_all = jnp.concatenate([F_ext_l, F_trans, F_ext_r])

        print(
            f"[ODE/{solver_name}] r = {abs(r):.4f}∠{np.angle(r)*180/np.pi:.1f}°, "
            f"t = {abs(t):.4f}∠{np.angle(t)*180/np.pi:.1f}°",
            flush=True,
        )
        return {
            'r': r, 't': t,
            'z': np.array(z_all), 'field': np.array(F_all),
            'z1': z1, 'z2': z2,
            'method': 'ODE', 'solver_name': solver_name,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Method 2: FEM (JAX-native, GPU-ready, differentiable)
    # ─────────────────────────────────────────────────────────────────────────

    def solve_fem(
        self,
        res_trans: float = None,
        res_ext: float = None,
        L_ext_factor: float = 2.0,
        order: int = 2,
    ) -> dict:
        """
        Solve by FEM on a non-uniform mesh with Robin (port) BCs.
        Assembly is vectorized (no Python loops); solve uses jnp.linalg.solve.
        Both steps run on GPU and support jax.grad differentiation.

        res_trans    : cells per λ₀ inside the transition layer [z1, z2].
                       Default = 100 × max√|εr(z)| over [z1, z2].
        res_ext      : cells per λ₀ in each exterior region of width L_ext_factor·λ₀.
                       Default = res_trans / 2.
        L_ext_factor : width of each exterior region in units of λ₀.
                       The mesh spans [z1 − L_ext_factor·λ₀, z2 + L_ext_factor·λ₀].
                       Default = 2.0, giving exterior regions of 2λ₀ on each side.
        order        : 1 = P1 linear elements, 2 = P2 quadratic elements (default).
                       P2 inserts midpoint nodes between all P1 nodes, reducing
                    dispersion error from O(h²) to O(h⁴) and eliminating
                    exterior standing-wave oscillations at the same mesh density.

        Returns dict with keys: r, t, z, field, method='FEM', order=order.
        """
        if res_trans is None:
            res_trans = self._auto_res_trans(factor=100.0)
        if res_ext is None:
            res_ext = self._auto_res_ext(res_trans)

        width   = self.z2 - self.z1
        L_ext   = L_ext_factor * self.lambda0
        n_trans = self._res_to_n(res_trans, width)
        n_ext   = self._res_to_n(res_ext, L_ext)

        print(
            f"  → FEM res_trans={res_trans:.2f}/λ₀ → n_trans={n_trans},"
            f"  res_ext={res_ext:.2f}/λ₀ → n_ext={n_ext}",
            flush=True,
        )

        z1, z2 = self.z1, self.z2
        kz1, kz2 = self.kz1, self.kz2
        k0  = float(self.k0)
        kx  = float(np.real(self.kx))
        eps1, eps2 = self.eps1, self.eps2

        # Build non-uniform node array as JAX array (GPU)
        z_ext_l = jnp.linspace(z1 - L_ext, z1,          n_ext,          endpoint=False)
        z_trans = jnp.linspace(z1,          z2,          n_trans)
        z_ext_r = jnp.linspace(z2,          z2 + L_ext,  n_ext + 1)[1:]
        nodes   = jnp.concatenate([z_ext_l, z_trans, z_ext_r])

        # Evaluate ε at transition nodes; clamp exterior to constant ε1/ε2
        eps_trans = jnp.array([complex(self.eps_func(float(z))) for z in z_trans],
                               dtype=jnp.complex128)
        eps_vals  = jnp.concatenate([
            jnp.full(n_ext, complex(eps1), dtype=jnp.complex128),
            eps_trans,
            jnp.full(n_ext, complex(eps2), dtype=jnp.complex128),
        ])

        F = assemble_and_solve_fem(
            nodes, eps_vals,
            k0, kx,
            complex(kz1), complex(eps1), complex(kz2), complex(eps2),
            n_ext, self.pol, order,
        )

        if order == 2:
            # P2: P1 node i maps to P2 index 2i; midpoints at odd indices
            idx_z1 = 2 * n_ext
            idx_z2 = 2 * (n_ext + n_trans - 1)
            nodes_mid = 0.5 * (nodes[:-1] + nodes[1:])
            n_p2 = 2 * nodes.shape[0] - 1
            z_p2 = jnp.zeros(n_p2)
            z_p2 = z_p2.at[::2].set(nodes)
            z_p2 = z_p2.at[1::2].set(nodes_mid)
            z_out = z_p2
        else:
            idx_z1 = n_ext
            idx_z2 = n_ext + n_trans - 1
            z_out  = nodes

        r = complex(F[idx_z1] - 1.0)
        t = complex(F[idx_z2])

        print(
            f"[FEM] r = {abs(r):.4f}∠{np.angle(r)*180/np.pi:.1f}°, "
            f"t = {abs(t):.4f}∠{np.angle(t)*180/np.pi:.1f}°",
            flush=True,
        )
        return {
            'r': r, 't': t,
            'z': np.array(z_out), 'field': np.array(F),
            'z1': z1, 'z2': z2,
            'method': 'FEM', 'order': order,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Energy conservation check
    # ─────────────────────────────────────────────────────────────────────────

    def energy_check(self, r: complex, t: complex) -> float:
        """
        Check energy conservation: R + T ≈ 1 for lossless media.
          TE:  T = |t|² · Re(kz2) / Re(kz1)
          TM:  T = |t|² · Re(kz2/ε2) / Re(kz1/ε1)
        """
        R = abs(r) ** 2
        kz1, kz2 = self.kz1, self.kz2
        if self.pol == 'S':
            T = abs(t) ** 2 * np.real(kz2) / np.real(kz1)
        else:
            T = abs(t) ** 2 * np.real(kz2 / self.eps2) / np.real(kz1 / self.eps1)
        print(
            f"Energy check: R={R:.6f}, T={T:.6f}, R+T={R+T:.6f}",
            flush=True,
        )
        return float(R + T)

    # ─────────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────────

    def plot_field(self, result: dict, ax=None, title: str = ''):
        """
        Plot |F(z)| and Re{F(z)} vs z, shade transition layer.
        Works for both ODE and FEM results.
        """
        z   = np.asarray(result['z'], dtype=float)
        F   = np.asarray(result['field'])
        z1  = result['z1']
        z2  = result['z2']
        method = result['method']
        r, t = result['r'], result['t']

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(z, np.abs(F),  'k-',  lw=1.5, label='|F(z)|')
        ax.plot(z, np.real(F), 'b--', lw=1.0, alpha=0.7, label='Re{F(z)}')
        ax.axvspan(z1, z2, color='gray', alpha=0.2, label='Transition layer')
        ax.axvline(z1, color='k', lw=0.8, ls=':')
        ax.axvline(z2, color='k', lw=0.8, ls=':')

        field_label = 'Ey' if self.pol == 'S' else 'Hy'
        ax.set_xlabel('z / λ₀', fontsize=13)
        ax.set_ylabel(f'Normalized {field_label}', fontsize=13)
        info = (f"{method}  pol={self.pol}  θ={self.theta_deg}°  "
                f"|r|={abs(r):.3f}  |t|={abs(t):.3f}")
        ax.set_title(title or info, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_eps(self, n: int = 2000, L_ext_factor: float = 1.5, ax=None):
        """
        Plot Re{ε(z)} and Im{ε(z)} vs z.
        Gray band = transition layer [z1, z2]; dashed verticals at boundaries.
        Exterior regions shown as flat lines at ε₁ (left) and ε₂ (right).
        """
        d = self.z2 - self.z1
        z_left  = self.z1 - L_ext_factor * d
        z_right = self.z2 + L_ext_factor * d
        z = np.linspace(z_left, z_right, n)
        eps_vals = np.array([complex(self.eps_func(zi)) for zi in z])

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(z, np.real(eps_vals), 'b-',  lw=2.0, label=r'Re{$\varepsilon_r(z)$}')
        ax.plot(z, np.imag(eps_vals), 'r--', lw=2.0, label=r'Im{$\varepsilon_r(z)$}')
        ax.axvspan(self.z1, self.z2, color='gray', alpha=0.18, label='Transition layer')
        ax.axvline(self.z1, color='k', lw=0.9, ls=':')
        ax.axvline(self.z2, color='k', lw=0.9, ls=':')
        ax.annotate(f'ε₁={self.eps1:.3g}', xy=(self.z1, np.real(self.eps1)),
                    xytext=(self.z1 - 0.35*d, np.real(self.eps1)),
                    fontsize=10, color='b', ha='right')
        ax.annotate(f'ε₂={self.eps2:.3g}', xy=(self.z2, np.real(self.eps2)),
                    xytext=(self.z2 + 0.15*d, np.real(self.eps2)),
                    fontsize=10, color='b', ha='left')
        ax.set_xlabel('z / λ₀', fontsize=13)
        ax.set_ylabel(r'$\varepsilon_r(z)$', fontsize=13)
        ax.set_title(
            f'Permittivity profile  [z₁={self.z1}, z₂={self.z2}]', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_comparison(self, res_ode: dict, res_fem: dict):
        """Side-by-side comparison of ODE and FEM field profiles."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"pol={self.pol}, θ={self.theta_deg}°, ε₁={self.eps1:.3g}, ε₂={self.eps2:.3g}",
            fontsize=14,
        )
        self.plot_field(res_ode, ax=ax1, title='Method 1: ODE (diffrax)')
        self.plot_field(res_fem, ax=ax2, title='Method 2: FEM (JAX, GPU)')
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized (vmap) parameter sweeps
# ─────────────────────────────────────────────────────────────────────────────

def make_kz_jax(eps_val, kx, k0):
    """JAX-traceable kz with Im(kz) ≥ 0; works inside vmap/jit."""
    val = k0**2 * eps_val - kx**2 + 0j
    kz = jnp.sqrt(val)
    return jnp.where(jnp.imag(kz) < 0, -kz, kz)


@partial(jit, static_argnames=('pol', 'n_trans', 'return_field'))
def sweep_ode(
    kx_arr, kz1_arr, kz2_arr,
    eps_re_arr, eps_im_arr,
    deps_re_arr, deps_im_arr,
    z_grid, k0, z1, z2, n_trans, pol,
    return_field=False,
):
    """
    Generic ODE parameter sweep via vmap — runs all N cases in parallel on GPU.

    **All** of the first 7 arguments must have a leading batch dimension N.
    Use ``jnp.tile`` / ``jnp.full`` to broadcast non-swept parameters:

      # angle sweep  (kx/kz vary, ε fixed)
      eps_re_arr = jnp.tile(eps_re[None], (N, 1))   # repeat 1-D array N times
      kx_arr, kz1_arr, kz2_arr = ...                 # shape (N,)

      # profile sweep  (ε varies, kx/kz fixed)
      kx_arr = jnp.full((N,), kx_scalar)             # broadcast scalar
      eps_re_arr = ...                                # shape (N, n_interp)

    kx_arr, kz1_arr, kz2_arr : (N,) float/complex.
    eps_re/im_arr            : (N, n_interp).
    deps_re/im_arr           : (N, n_interp)  [TM only; pass zeros for TE].
    pol                      : 'S' (TE) or 'P' (TM) — static.
    n_trans                  : ODE step-size hint — static.
    return_field             : bool (static).
        False  →  R = |r|² of shape (N,).
        True   →  normalized complex field of shape (N, n_trans), ordered z1→z2.
    """
    dt0   = -(z2 - z1) / n_trans
    saveat = (SaveAt(ts=jnp.linspace(z2, z1, n_trans))   # full profile
              if return_field else SaveAt(t1=True))        # final state only

    def _solve(ode_rhs, args_fn, kx, kz1, kz2, eps_re, eps_im, deps_re, deps_im):
        y0  = jnp.stack([jnp.ones(()), jnp.zeros(()), -jnp.imag(kz2), -jnp.real(kz2)])
        sol = diffeqsolve(
            ODETerm(ode_rhs), Dopri5(),
            t0=z2, t1=z1, dt0=dt0, y0=y0,
            args=args_fn(kx, eps_re, eps_im, deps_re, deps_im),
            saveat=saveat,
            stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
            max_steps=50000000,
        )
        if return_field:
            ys    = sol.ys                              # (n_trans, 4)
            F_z1  = ys[-1, 0] + 1j * ys[-1, 1]
            dF_z1 = ys[-1, 2] + 1j * ys[-1, 3]
            A     = 0.5 * (F_z1 + 1j * dF_z1 / kz1)
            return jnp.flip(ys[:, 0] + 1j * ys[:, 1]) / A   # (n_trans,) z1→z2
        else:
            ys    = sol.ys[0]                           # (1,4) → (4,) final state
            F_z1  = ys[0] + 1j * ys[1]
            dF_z1 = ys[2] + 1j * ys[3]
            A     = 0.5 * (F_z1 + 1j * dF_z1 / kz1)
            B     = 0.5 * (F_z1 - 1j * dF_z1 / kz1)
            return jnp.abs(B / A) ** 2                  # scalar R

    if pol == 'S':
        def single(kx, kz1, kz2, eps_re, eps_im, deps_re, deps_im):
            return _solve(ode_rhs_te,
                          lambda kx, er, ei, dr, di: (z_grid, er, ei, kx, k0),
                          kx, kz1, kz2, eps_re, eps_im, deps_re, deps_im)
    else:
        def single(kx, kz1, kz2, eps_re, eps_im, deps_re, deps_im):
            return _solve(ode_rhs_tm,
                          lambda kx, er, ei, dr, di: (z_grid, er, ei, dr, di, kx, k0),
                          kx, kz1, kz2, eps_re, eps_im, deps_re, deps_im)

    return vmap(single)(kx_arr, kz1_arr, kz2_arr,
                        eps_re_arr, eps_im_arr, deps_re_arr, deps_im_arr)


@partial(jit, static_argnames=('pol', 'order', 'return_field'))
def sweep_fem(
    kx_arr, kz1_arr, kz2_arr,
    eps_vals_arr,
    nodes, k0, eps1, eps2, n_ext, pol, order=1,
    return_field=False,
):
    """
    Generic FEM reflectance sweep via vmap — all N cases in parallel on GPU.

    All first-4 arguments must have leading batch dimension N.
    Use ``jnp.tile`` / ``jnp.full`` to broadcast non-swept parameters.

    kx_arr, kz1_arr, kz2_arr : (N,) float/complex.
    eps_vals_arr             : (N, n_nodes) complex.
    nodes                    : (n_nodes,) shared mesh.
    pol                      : 'S' or 'P' — static.
    order                    : 1 (P1) or 2 (P2) — static.
    return_field             : bool (static).
        False  →  R = |r|²    of shape (N,).
        True   →  complex r   of shape (N,).
    """
    r_idx = 2 * n_ext if order == 2 else n_ext

    def single(kx, kz1, kz2, eps_vals):
        F = assemble_and_solve_fem(
            nodes, eps_vals, k0, kx, kz1, eps1, kz2, eps2, n_ext, pol, order
        )
        r = F[r_idx] - 1.0
        if return_field:
            return r
        return jnp.abs(r) ** 2

    return vmap(single)(kx_arr, kz1_arr, kz2_arr, eps_vals_arr)
