"""
Analytical Airy-function solution for a piecewise-linear permittivity layer (TE).

Physics convention  (matches main.py):
  time factor : exp(−jωt)
  right-going wave in +z : exp(−j kz z)

For a single linear segment  εr(z) = g·(z − z₀), the TE Helmholtz equation
  d²E/dz² + [k₀²·εr(z) − kx²]·E = 0
transforms to the Airy equation  d²E/dξ² − ξ·E = 0  via
  ξ(z) = −α·(z − z₀_eff),   α = (k₀²g)^{1/3},   z₀_eff = z₀ + kx²/(k₀²g).

The general solution is  E(z) = c₁·Ai(ξ) + c₂·Bi(ξ).

Public API
----------
solve_and_field(z0, g, z1, z2, k0, z_arr, kx=0)
    Single linear segment: returns (field, r, t).

solve_v_profile(eps0, delta, k0, kx=0)
    V-shaped profile εr(z) = eps0 + (1−eps0)·|z/δ|, z ∈ [−δ, δ].
    Returns dict with r, t and Airy coefficients for each half.

full_field_v_profile(z_arr, eps0, delta, k0, kx=0, sol=None)
    Full complex field on an arbitrary z grid including plane-wave exteriors.
"""

import numpy as np
from scipy.special import airy as _scipy_airy

__all__ = [
    "transfer_matrix",
    "compute_airy_coefficients",
    "airy_field",
    "solve_and_field",
    "solve_v_profile",
    "full_field_v_profile",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xi_alpha(z, z0, g, k0, kx=0.0):
    """Return (ξ_array, α) using the principal complex cube root of k₀²g."""
    alpha  = np.power(k0**2 * g + 0j, 1.0 / 3.0)
    z0_eff = z0 + kx**2 / (k0**2 * g)
    xi = -alpha * (np.asarray(z, dtype=complex) - z0_eff)
    return xi, alpha


def _K_mat(z_scalar, z0, g, k0, kx=0.0):
    """
    2×2 matrix K at a single point such that [E, dE/dz]ᵀ = K · [c₁, c₂]ᵀ.
    K = [[ Ai(ξ),      Bi(ξ)     ],
         [−α Ai'(ξ),  −α Bi'(ξ) ]]
    Note: dE/dz = dE/dξ · dξ/dz = −α · dE/dξ.
    """
    xi, alpha = _xi_alpha(z_scalar, z0, g, k0, kx)
    Ai, Aip, Bi, Bip = _scipy_airy(complex(xi))
    return np.array([[Ai,        Bi       ],
                     [-alpha*Aip, -alpha*Bip]], dtype=complex)


# ─────────────────────────────────────────────────────────────────────────────
# Public core functions
# ─────────────────────────────────────────────────────────────────────────────

def transfer_matrix(z_a, z_b, z0, g, k0, kx=0.0):
    """
    Transfer matrix M such that
      [E(z_a), dE/dz(z_a)]ᵀ = M · [E(z_b), dE/dz(z_b)]ᵀ
    for εr(z) = g·(z − z₀), TE polarisation (or kx = 0).
    Works for any real or complex g (including g < 0).
    """
    Ka = _K_mat(z_a, z0, g, k0, kx)
    Kb = _K_mat(z_b, z0, g, k0, kx)
    return Ka @ np.linalg.inv(Kb)


def compute_airy_coefficients(z0, g, z1, z2, k0, kz1, kz2, kx=0.0):
    """
    Solve for r, t, c₁, c₂ for a single linear segment εr(z) = g·(z − z₀).

    Boundary conditions (right-going wave ∝ exp(−j kz z)):
      z = z₁ :  E = 1 + r,     dE/dz = −j·kz1·(1 − r)
      z = z₂ :  E = t,          dE/dz = −j·conj(kz2)·t

    Parameters
    ----------
    z0, g   : linear profile  εr(z) = g·(z − z₀)
    z1, z2  : segment boundaries
    k0      : free-space wavenumber
    kz1, kz2: z-wavenumbers in regions 1 and 2 (Im(kz) ≥ 0 convention)
    kx      : conserved tangential wavenumber (0 for normal incidence)

    Returns
    -------
    r, t  : complex reflection / transmission coefficients
    c1, c2: Airy superposition weights in E(z) = c₁ Ai(ξ) + c₂ Bi(ξ)
    """
    M = transfer_matrix(z1, z2, z0, g, k0, kx)
    m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

    # BCs →  1+r = P·t  (A),    −j·kz1·(1−r) = Q·t  (B)
    # dE/dz(z2) = −j·conj(kz2)·t  (decaying evanescent/propagating transmitted wave)
    P = m11 - 1j * np.conj(kz2) * m12
    Q = m21 - 1j * np.conj(kz2) * m22
    # (A) + (−1/j·kz1)·(B)  →  2 = (P + j·Q/kz1)·t
    t = complex(2.0 / (P + 1j * Q / kz1))
    r = complex(P * t - 1.0)

    # c₁, c₂ from  K(z₂)·[c₁,c₂]ᵀ = [t, −j·conj(kz2)·t]ᵀ  (decaying evanescent BC)
    K2 = _K_mat(z2, z0, g, k0, kx)
    c  = np.linalg.solve(K2, np.array([t, -1j * np.conj(kz2) * t]))
    return r, t, complex(c[0]), complex(c[1])


def airy_field(z0, g, k0, c1, c2, z_arr, kx=0.0):
    """
    E(z) = c₁·Ai(ξ(z)) + c₂·Bi(ξ(z))  evaluated at every point in z_arr.

    Parameters
    ----------
    z0, g : define εr(z) = g·(z − z₀)
    k0    : free-space wavenumber
    c1, c2: Airy superposition coefficients
    z_arr : array of z positions
    kx    : conserved tangential wavenumber

    Returns
    -------
    complex ndarray of the same shape as z_arr
    """
    xi, _ = _xi_alpha(np.asarray(z_arr, dtype=float), z0, g, k0, kx)
    Ai_v, _, Bi_v, _ = _scipy_airy(xi)
    return c1 * Ai_v + c2 * Bi_v


def solve_and_field(z0, g, z1, z2, k0, z_arr, kx=0.0):
    """
    Solve a single linear segment and return the field and r, t.

    The permittivity profile is  εr(z) = g·(z − z₀)  over [z1, z2].
    ε at the boundaries is read directly from the profile:
      ε₁ = g·(z₁ − z₀),  ε₂ = g·(z₂ − z₀).

    Parameters
    ----------
    z0, g  : linear profile (εr = g·(z − z₀))
    z1, z2 : transition layer boundaries
    k0     : free-space wavenumber  2π/λ₀
    z_arr  : z positions for field output (should lie in [z1, z2])
    kx     : conserved tangential wavenumber

    Returns
    -------
    field : complex ndarray
    r, t  : complex reflection / transmission coefficients
    """
    def _kz(eps):
        val = np.sqrt(k0**2 * complex(eps) - kx**2 + 0j)
        return val if val.imag >= 0 else -val

    kz1 = _kz(g * (z1 - z0))
    kz2 = _kz(g * (z2 - z0))
    r, t, c1, c2 = compute_airy_coefficients(z0, g, z1, z2, k0, kz1, kz2, kx)
    field = airy_field(z0, g, k0, c1, c2, np.asarray(z_arr, dtype=float), kx)
    return field, r, t


# ─────────────────────────────────────────────────────────────────────────────
# V-shaped profile  εr(z) = eps0 + (1−eps0)·|z/δ|  on  |z| ≤ δ
# ─────────────────────────────────────────────────────────────────────────────

def solve_v_profile(eps0, delta, k0, kx=0.0):
    """
    Analytical solution for the symmetric V-shaped permittivity profile:
      εr(z) = eps0 + (1−eps0)·|z/δ|,   z ∈ [−δ, δ]
      εr(z) = 1,                          |z| > δ

    The profile is piecewise linear:
      Left  half [−δ, 0]: εr(z) = g_L·(z − z0_L),   g_L = (eps0−1)/δ
      Right half [ 0, δ]: εr(z) = g_R·(z − z0_R),   g_R = (1−eps0)/δ

    The outer media both have ε = 1 (same on both sides by construction).

    Parameters
    ----------
    eps0  : permittivity at the centre z = 0
    delta : half-width of the transition layer
    k0    : free-space wavenumber  2π/λ₀
    kx    : conserved tangential wavenumber (0 for normal incidence)

    Returns
    -------
    dict with keys:
      r, t        : complex reflection / transmission coefficients
      kz1, kz2    : z-wavenumbers in outer media (equal here)
      c1_L, c2_L  : Airy coefficients for left half
      g_L, z0_L   : linear profile of left half
      c1_R, c2_R  : Airy coefficients for right half
      g_R, z0_R   : linear profile of right half
    """
    # Piecewise linear parameters
    g_L  = (eps0 - 1.0) / delta       # dεr/dz in left half
    z0_L = -eps0 / g_L                # zero of εr (turning point) on left
    g_R  = (1.0 - eps0) / delta       # dεr/dz in right half
    z0_R = -eps0 / g_R                # zero of εr on right

    def _kz(eps):
        val = np.sqrt(k0**2 * complex(eps) - kx**2 + 0j)
        return val if val.imag >= 0 else -val

    kz = _kz(1.0)   # same medium (ε=1) on both sides

    # Compose transfer matrices:  M = M_L @ M_R  (propagate right → left)
    M_R = transfer_matrix( 0.0,  delta, z0_R, g_R, k0, kx)  # 0 ← δ
    M_L = transfer_matrix(-delta, 0.0,  z0_L, g_L, k0, kx)  # −δ ← 0
    M   = M_L @ M_R

    m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    P = m11 - 1j * kz * m12
    Q = m21 - 1j * kz * m22
    t = complex(2.0 / (P + 1j * Q / kz))
    r = complex(P * t - 1.0)

    # Right-half Airy coefficients from BC at z = +δ
    K_R_z2 = _K_mat(delta, z0_R, g_R, k0, kx)
    c_R = np.linalg.solve(K_R_z2, np.array([t, -1j * kz * t]))

    # Left-half coefficients from continuity at z = 0
    state_0 = _K_mat(0.0, z0_R, g_R, k0, kx) @ c_R   # [E(0), dE/dz(0)]
    c_L = np.linalg.solve(_K_mat(0.0, z0_L, g_L, k0, kx), state_0)

    return dict(
        r=r, t=t, kz1=kz, kz2=kz,
        c1_L=complex(c_L[0]), c2_L=complex(c_L[1]), g_L=g_L, z0_L=z0_L,
        c1_R=complex(c_R[0]), c2_R=complex(c_R[1]), g_R=g_R, z0_R=z0_R,
    )


def full_field_v_profile(z_arr, eps0, delta, k0, kx=0.0, sol=None):
    """
    Full complex field for the V-shaped profile on an arbitrary z grid,
    including analytical plane-wave expressions in both exterior regions.

    Regions:
      z < −δ  :  exp(−j·kz1·(z+δ)) + r·exp(+j·kz1·(z+δ))   (normalised at z=−δ)
      −δ≤z≤0  :  c1_L·Ai(ξ_L(z)) + c2_L·Bi(ξ_L(z))
       0<z≤ δ  :  c1_R·Ai(ξ_R(z)) + c2_R·Bi(ξ_R(z))
      z > δ   :  t·exp(−j·conj(kz2)·(z−δ))

    Parameters
    ----------
    z_arr  : 1-D array of z positions
    eps0   : permittivity at z = 0
    delta  : half-width of the transition layer
    k0     : free-space wavenumber
    kx     : conserved tangential wavenumber
    sol    : dict returned by solve_v_profile (computed if None)

    Returns
    -------
    complex ndarray of shape z_arr.shape
    """
    if sol is None:
        sol = solve_v_profile(eps0, delta, k0, kx)

    r, t       = sol['r'], sol['t']
    kz1, kz2   = sol['kz1'], sol['kz2']
    z_arr      = np.asarray(z_arr, dtype=float)
    field      = np.empty(z_arr.shape, dtype=complex)

    mL  = z_arr < -delta
    mTL = (z_arr >= -delta) & (z_arr <= 0.0)
    mTR = (z_arr >  0.0)   & (z_arr <= delta)
    mR  = z_arr > delta

    if mL.any():
        dz = z_arr[mL] - (-delta)
        field[mL] = np.exp(-1j * kz1 * dz) + r * np.exp(1j * kz1 * dz)

    if mTL.any():
        field[mTL] = airy_field(
            sol['z0_L'], sol['g_L'], k0, sol['c1_L'], sol['c2_L'],
            z_arr[mTL], kx)

    if mTR.any():
        field[mTR] = airy_field(
            sol['z0_R'], sol['g_R'], k0, sol['c1_R'], sol['c2_R'],
            z_arr[mTR], kx)

    if mR.any():
        field[mR] = t * np.exp(-1j * np.conj(kz2) * (z_arr[mR] - delta))

    return field
