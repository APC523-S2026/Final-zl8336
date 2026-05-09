# 1D Helmholtz Solver for Electromagnetic Wave Propagation Through a Dielectric Transition Layer

## 1. Background and Motivation

When a monochromatic electromagnetic (EM) plane wave propagates along the $z$-axis and encounters a medium whose relative permittivity $\varepsilon_r(z)$ varies continuously, part of the wave is reflected back and the remainder is transmitted forward.
Understanding this scattering problem is fundamental in photonics, optics, and antenna design — for example, anti-reflection coatings, gradient-index lenses, and plasma sheaths all involve such transition layers.

This project implements a self-contained 1D solver for this problem.
The computational domain is divided into three zones:

| Zone | Range | Description |
|------|-------|-------------|
| Region 1 | $z \le z_1$ | Semi-infinite homogeneous medium, permittivity $\varepsilon_1$ |
| Transition layer | $z_1 \le z \le z_2$ | Inhomogeneous medium, $\varepsilon_r(z)$ varies continuously |
| Region 2 | $z \ge z_2$ | Semi-infinite homogeneous medium, permittivity $\varepsilon_2$ |

The solver takes as input the permittivity function $\varepsilon_r(z)$, the layer boundaries $z_1$ and $z_2$, the angle of incidence $\theta$, the polarisation (TE or TM), and the free-space wavelength $\lambda_0$.
It returns the complex reflection coefficient $r$, the complex transmission coefficient $t$, and the full field profile inside and outside the layer.

---

## 2. Physical Setup and Sign Conventions

Throughout this project we adopt the following sign conventions:

- **Time dependence:** $e^{-j\omega t}$, where $\omega = 2\pi c / \lambda_0$ is the angular frequency and $c$ is the speed of light.
- **Right-going plane wave in $+z$:** $e^{-j k_z z}$, i.e.\ spatial phase *decreases* in the direction of propagation.
- **Free-space wavenumber:** $k_0 = 2\pi / \lambda_0$.
- **Transverse wavenumber:** $k_x = k_0 \sqrt{\varepsilon_1}\,\sin\theta$, conserved by Snell's law.
- **$z$-wavenumber in a uniform medium with permittivity $\varepsilon$:**

$$
k_z(\varepsilon) = \sqrt{k_0^2\,\varepsilon - k_x^2}, \quad \operatorname{Im}(k_z) \ge 0.
$$

The branch cut is chosen so that evanescent waves (which arise when $k_0^2\varepsilon - k_x^2 < 0$) decay in the $+z$ direction.

With a unit-amplitude incident wave normalised at $z = z_1$, the total field in each exterior region is

$$
E(z) = e^{-jk_{z1}(z-z_1)} + r\,e^{+jk_{z1}(z-z_1)}, \quad z < z_1,
$$
$$
E(z) = t\,e^{-jk_{z2}(z-z_2)}, \quad z > z_2,
$$

where $r$ and $t$ are the reflection and transmission coefficients.
Energy conservation for lossless media reads

$$
R + T = 1, \quad R = |r|^2, \quad T = |t|^2\,\frac{\operatorname{Re}(k_{z2})}{\operatorname{Re}(k_{z1})}\ \ \text{(TE)}, \quad
T = |t|^2\,\frac{\operatorname{Re}(k_{z2}/\varepsilon_2)}{\operatorname{Re}(k_{z1}/\varepsilon_1)}\ \ \text{(TM)}.
$$

---

## 3. Governing Equations

The 3D Maxwell equations reduce to a 1D scalar ODE for the field component that is tangential to the interface.

### 3.1 TE Polarisation (S-polarisation, $\mathbf{E} \parallel \hat{y}$)

The $y$-component of the electric field $E_y(z)$ satisfies the **standard Helmholtz equation**:

$$
\boxed{\frac{d^2 E_y}{dz^2} + \left[k_0^2\,\varepsilon_r(z) - k_x^2\right] E_y = 0.}
$$

This is the equation of a 1D harmonic oscillator with a position-dependent "spring constant" $k_{\rm eff}^2(z) = k_0^2\varepsilon_r(z) - k_x^2$.

### 3.2 TM Polarisation (P-polarisation, $\mathbf{H} \parallel \hat{y}$)

The $y$-component of the magnetic field $H_y(z)$ satisfies a **modified Helmholtz equation** with an additional first-derivative (drift) term that arises from the spatial variation of $\varepsilon_r$:

$$
\boxed{\frac{d^2 H_y}{dz^2} - \frac{1}{\varepsilon_r(z)}\frac{d\varepsilon_r}{dz}\frac{d H_y}{dz} + \left[k_0^2\,\varepsilon_r(z) - k_x^2\right] H_y = 0.}
$$

The drift coefficient $\varepsilon_r^{-1}\,d\varepsilon_r/dz$ vanishes wherever $\varepsilon_r$ is constant (homogeneous regions) and reduces the TM equation to the same form as the TE equation in those regions.

In both cases the field $F(z)$ (either $E_y$ or $H_y$) can be written as a second-order linear ODE:

$$
F'' = f(z,\, F,\, F').
$$

---

## 4. Method 1 — ODE Shooting

### 4.1 Principle

The ODE shooting method converts the boundary-value problem into an initial-value problem by exploiting the fact that the field in Region 2 has the known form $F(z) = t\,e^{-jk_{z2}(z-z_2)}$.
Differentiating: $F'(z_2) = -jk_{z2}\,t$.

We therefore fix **initial conditions at $z = z_2$** with unit transmitted amplitude ($t = 1$ before normalisation):

$$
F(z_2) = 1, \quad F'(z_2) = -jk_{z2},
$$

and integrate the ODE **backward** from $z_2$ to $z_1$.
This backward shooting approach is numerically preferred because forward propagation can amplify exponentially growing evanescent modes, whereas backward propagation keeps them decaying.

After integration we arrive at $(F(z_1),\, F'(z_1))$ with unknown normalization.
The reflection and transmission coefficients are extracted by decomposing the field at $z_1$ into left- and right-going plane waves:

$$
A = \tfrac{1}{2}\!\left(F(z_1) + \frac{j\,F'(z_1)}{k_{z1}}\right), \quad
B = \tfrac{1}{2}\!\left(F(z_1) - \frac{j\,F'(z_1)}{k_{z1}}\right),
$$

$$
r = \frac{B}{A}, \qquad t = \frac{1}{A}.
$$

Here $A$ is the amplitude of the incident (right-going) wave and $B$ is the amplitude of the reflected (left-going) wave at $z_1$.
The entire field is then rescaled by $1/A$ so that the incident amplitude is unity.

### 4.2 Implementation with diffrax (JAX)

The second-order ODE is rewritten as a first-order system by splitting into real and imaginary parts.
Defining the state vector $\mathbf{y} = [\operatorname{Re}F,\, \operatorname{Im}F,\, \operatorname{Re}F',\, \operatorname{Im}F'] \in \mathbb{R}^4$, the right-hand side is linear in $\mathbf{y}$ and can be evaluated using pre-computed interpolation tables of $\varepsilon_r(z)$ and $d\varepsilon_r/dz$ on the interval $[z_1, z_2]$.

The integration is performed with the **diffrax** library (JAX-based), which supports just-in-time (JIT) compilation and GPU execution.
The following time-stepping schemes are available:

| Solver name | Method | Order | Step control |
|-------------|--------|-------|--------------|
| `Dopri5` | Dormand–Prince | 5th | Adaptive (PID controller) |
| `Tsit5` | Tsitouras | 5th | Adaptive (PID controller) |
| `Heun` | Heun (explicit RK2) | 2nd | Adaptive or fixed |
| `KenCarp4` | Kennedy–Carpenter ARK (IMEX) | 4th | Adaptive |
| `SemiImplicitEuler` | Störmer–Verlet split | 1st | Fixed |

For adaptive solvers the user specifies relative (`rtol`) and absolute (`atol`) error tolerances.
For fixed-step operation (useful in convergence studies) the step size is set by the number of grid points `n_trans` inside the transition layer: $h = (z_2 - z_1)/n_{\rm trans}$.

---

## 5. Method 2 — Finite Element Method (FEM)

### 5.1 Weak Formulation

The FEM is based on the Galerkin weighted-residual approach.
Multiplying the TE equation $F'' + k_{\rm eff}^2(z)\,F = 0$ by a test function $v(z)$, integrating over the domain $[z_{\rm left},\, z_{\rm right}]$, and integrating the $F''$ term by parts gives the **weak form**:

$$
\int_{z_{\rm left}}^{z_{\rm right}} F'\,v'\,dz - \int_{z_{\rm left}}^{z_{\rm right}} k_{\rm eff}^2(z)\,F\,v\,dz = \left[F' v\right]_{z_{\rm left}}^{z_{\rm right}}.
$$

The boundary terms on the right-hand side are replaced by the physical radiation (Robin) boundary conditions described below.

For TM polarisation the drift term introduces an additional contribution:

$$
\int \frac{1}{\varepsilon_r} F'\,v'\,dz - \int k_{\rm TM}^2(z)\,F\,v\,dz = \text{boundary terms},
$$

where $k_{\rm TM}^2(z) = k_0^2 - k_x^2/\varepsilon_r(z)$ and the stiffness integrand carries the $1/\varepsilon_r$ weight.

### 5.2 Mesh and Element Types

The computational domain $[z_{\rm left}, z_{\rm right}] = [z_1 - L_{\rm ext},\, z_2 + L_{\rm ext}]$ is discretised into elements.
Two element types are supported:

#### P1 — Linear Elements (first order)

Each element $[z_e,\, z_{e+1}]$ of width $h_e = z_{e+1} - z_e$ carries two linear shape functions:

$$
\varphi_1(\xi) = 1 - \xi, \quad \varphi_2(\xi) = \xi, \quad \xi = \frac{z - z_e}{h_e} \in [0,1].
$$

The element stiffness matrix (from the $\int F'v'\,dz$ term) and mass matrix (from the $\int k_{\rm eff}^2 F\,v\,dz$ term) are:

$$
K_{\rm el}^{(1)} = \frac{1}{h_e}\begin{pmatrix}1 & -1 \\ -1 & 1\end{pmatrix}, \qquad
M_{\rm el}^{(1)} = \frac{(k_0^2\bar\varepsilon_e - k_x^2)\,h_e}{6}\begin{pmatrix}2 & 1 \\ 1 & 2\end{pmatrix},
$$

where $\bar\varepsilon_e = \tfrac{1}{2}(\varepsilon_r(z_e)+\varepsilon_r(z_{e+1}))$ is the midpoint permittivity.
The element matrix is $A_{\rm el}^{(1)} = K_{\rm el}^{(1)} - M_{\rm el}^{(1)}$.

For TM polarisation the stiffness carries the $1/\bar\varepsilon_e$ weight and the mass term uses $k_0^2 - k_x^2/\bar\varepsilon_e$:

$$
K_{\rm el,TM}^{(1)} = \frac{1}{\bar\varepsilon_e\,h_e}\begin{pmatrix}1 & -1 \\ -1 & 1\end{pmatrix}, \qquad
M_{\rm el,TM}^{(1)} = \frac{(k_0^2 - k_x^2/\bar\varepsilon_e)\,h_e}{6}\begin{pmatrix}2 & 1 \\ 1 & 2\end{pmatrix}.
$$

#### P2 — Quadratic Elements (second order)

A midpoint node is inserted at the centre of each P1 element, giving three nodes per element at $z_e$, $z_e + h_e/2$, $z_{e+1}$.
The three Lagrange shape functions on the reference element $\xi\in[0,1]$ are:

$$
\varphi_1(\xi) = (2\xi-1)(\xi-1), \quad
\varphi_2(\xi) = 4\xi(1-\xi), \quad
\varphi_3(\xi) = \xi(2\xi-1).
$$

The reference stiffness and mass matrices are:

$$
\hat K^{(2)} = \frac{1}{h_e}\begin{pmatrix} 7/3 & -8/3 & 1/3 \\ -8/3 & 16/3 & -8/3 \\ 1/3 & -8/3 & 7/3 \end{pmatrix}, \qquad
\hat M^{(2)} = \frac{(k_0^2\bar\varepsilon_e - k_x^2)\,h_e}{30}\begin{pmatrix} 4 & 2 & -1 \\ 2 & 16 & 2 \\ -1 & 2 & 4 \end{pmatrix}.
$$

Note the off-diagonal $-1$ entries in $\hat M^{(2)}$, which are characteristic of quadratic Lagrange elements.
The element matrix is $A_{\rm el}^{(2)} = \hat K^{(2)} - \hat M^{(2)}$.

If the mesh has $N$ P1 nodes, the P2 discretisation has $2N-1$ degrees of freedom (original nodes at even indices, midpoints at odd indices).

**Why P2?**
P1 elements introduce a numerical dispersion error of order $O(h^2)$: the discrete wavenumber $k_h \approx k(1 - k^2h^2/24)$ differs from the physical wavenumber $k$.
This mismatch between the propagating field inside the domain and the exact plane-wave boundary conditions creates a partial reflection at the boundaries, producing standing-wave oscillations in the exterior region and a convergence plateau.
P2 elements reduce the dispersion error to $O(h^4)$, eliminating the oscillations and restoring the expected convergence slope.

### 5.3 Radiation Boundary Conditions (Robin / Port BCs)

The exterior regions contain propagating plane waves.
Imposing that no spurious reflection enters from the computational boundaries leads to the **Robin (first-order absorbing) boundary conditions**:

- **Left boundary** $z = z_{\rm left}$: the field contains an incident wave $F_{\rm inc}(z) = e^{-jk_{z1}(z-z_1)}$ plus an outgoing reflected wave.
  The BC term added to the global system is:

$$
  F'(z_{\rm left}) = jk_{z1}\,F(z_{\rm left}) - 2jk_{z1}\,F_{\rm inc}(z_{\rm left}),
$$

  which in the assembled matrix reads: $A[0,0] \mathrel{+}= jk_{z1}$, $b[0] \mathrel{+}= 2jk_{z1}\,F_{\rm inc}(z_{\rm left})$.

- **Right boundary** $z = z_{\rm right}$: only the outgoing transmitted wave is present ($F' = -jk_{z2}\,F$).
  The BC term is: $A[-1,-1] \mathrel{+}= jk_{z2}$.

For TM polarisation the impedance $k_{z}/\varepsilon$ replaces $k_z$ in all boundary terms.

### 5.4 Global Assembly and Solution

All element matrices are assembled into a global $N\times N$ (P1) or $(2N-1)\times(2N-1)$ (P2) complex matrix $\mathbf{A}$ and right-hand side vector $\mathbf{b}$ by scatter-add operations.
The linear system

$$
\mathbf{A}\,\mathbf{F} = \mathbf{b}
$$

is solved by **dense LU factorisation** using `jnp.linalg.solve` (JAX, GPU-accelerated and fully differentiable via the implicit-function theorem).

After solving, the reflection and transmission coefficients are read off directly from the field vector:

$$
r = F[i_{z_1}] - 1, \qquad t = F[i_{z_2}],
$$

where $i_{z_1}$ and $i_{z_2}$ are the indices corresponding to $z_1$ and $z_2$ respectively.

---

## 6. Solver Interface

All functionality is wrapped in the class `HelmholtzSolver1D` (defined in `Final/main.py`).

```python
solver = HelmholtzSolver1D(
    eps_func,          # callable z → complex ε_r(z)
    z1, z2,            # transition layer boundaries
    theta_deg=0.0,     # angle of incidence in degrees
    pol='S',           # 'S' = TE,  'P' = TM
    lambda0=1.0,       # free-space wavelength
)

# ODE method (backward shooting, diffrax)
result = solver.solve_ode(
    res_trans=100,         # cells per λ₀ inside [z1, z2]
    res_ext=20,            # cells per λ₀ in exterior plot region
    solver_name='Dopri5',  # ODE integrator (see table above)
    fixed_step=False,      # True → constant step size for convergence studies
    rtol=1e-8, atol=1e-10,
)

# FEM method
result = solver.solve_fem(
    res_trans=100,   # cells per λ₀ inside [z1, z2]
    res_ext=20,
    order=2,         # 1 = P1 linear,  2 = P2 quadratic (recommended)
)

# result dict: {'r', 't', 'z', 'field', 'method', ...}
solver.energy_check(result['r'], result['t'])   # prints R, T, R+T
solver.plot_field(result)
```

Resolution is specified in **cells per $\lambda_0$**: `res_trans` gives the number of grid intervals per free-space wavelength inside the transition layer, and `res_ext` gives the same for the exterior plot region.
Defaults are chosen to resolve the local wavenumber $k_{\rm eff}(z) \propto \sqrt{|\varepsilon_r(z)|}$:

| Parameter | Default |
|-----------|---------|
| `res_trans` (ODE) | $50 \times \max_{z\in[z_1,z_2]}\sqrt{|\varepsilon_r(z)|}$ |
| `res_trans` (FEM) | $100 \times \max_{z\in[z_1,z_2]}\sqrt{|\varepsilon_r(z)|}$ |
| `res_ext` | $\texttt{res\_trans} / 2$ |

The FEM factor is doubled relative to ODE because FEM discretisation error is $O(h^2)$ (P1) or $O(h^4)$ (P2) and requires a finer base grid to match the ODE shooting accuracy at moderate $|\varepsilon_r|$.

---

## 7. Summary

| Feature | ODE shooting | FEM P1 | FEM P2 |
|---------|-------------|--------|--------|
| Spatial order | 5th (Dopri5) | 2nd | 4th |
| Exterior oscillations | None (analytical exterior) | Present at coarse mesh | Eliminated |
| Supports general $\varepsilon(z)$ | Yes | Yes | Yes |
| Supports TM polarisation | Yes | Yes | Yes |
| GPU / JIT | Yes (diffrax) | Yes (JAX) | Yes (JAX) |
| Differentiable | No (ODE solver) | Yes (jnp.linalg.solve) | Yes |

The ODE shooting method is the most accurate at moderate resolution (5th-order convergence with Dopri5) and naturally avoids exterior standing-wave artefacts.
The FEM P2 method is fully differentiable with respect to the permittivity profile, making it suitable for inverse-design and gradient-based optimisation workflows.

---

## 8. Evanescent Transmitted Wave: Branch Cut Subtlety and Sign Fix

### 8.1 The Problem

The $z$-wavenumber in a uniform medium is defined as

$$
k_z(\varepsilon) = \sqrt{k_0^2\,\varepsilon - k_x^2}, \qquad \operatorname{Im}(k_z) \ge 0.
$$

This branch choice is natural for the **incident and reflected waves** in Region 1.
For the **reflected** evanescent component ($\varepsilon_1 = 1$, propagating), there is no ambiguity.

However, for the **transmitted wave** in Region 2, the same branch creates a subtle sign error when $\varepsilon_2 < 0$ (e.g.\ a plasma with $\varepsilon_2 = -1$).

**Concrete example.** With $\varepsilon_2 = -1$ and $\theta = 45°$, $k_x = k_0/\sqrt{2}$:

$$
k_0^2\varepsilon_2 - k_x^2 = -k_0^2 - \tfrac{1}{2}k_0^2 = -\tfrac{3}{2}k_0^2 < 0.
$$

The branch cut gives $k_{z2} = +j\alpha$ (with $\alpha = k_0\sqrt{3/2} > 0$, $\operatorname{Im}(k_{z2}) > 0$).

Now substitute into the transmitted-wave formula:

$$
E_{\rm trans}(z) = t\,e^{-j k_{z2}(z-z_2)} = t\,e^{-j(+j\alpha)(z-z_2)} = t\,e^{+\alpha(z-z_2)}.
$$

This **grows exponentially** as $z \to +\infty$ — a non-physical result for an evanescent wave that should decay away from the interface.

### 8.2 The Two Independent Solutions

The Helmholtz equation in the uniform Region 2 ($\varepsilon_2 = \text{const}$) has two independent solutions:

$$
\psi_+(z) = e^{-jk_{z2}z} = e^{+\alpha z} \quad \text{(growing, non-physical)},
$$
$$
\psi_-(z) = e^{+jk_{z2}z} = e^{-\alpha z} \quad \text{(decaying, \textbf{physical})}.
$$

Note that $\psi_-(z) = e^{-j(-j\alpha)z} = e^{-j\,\overline{k_{z2}}\,z}$, where $\overline{k_{z2}} = \operatorname{conj}(k_{z2}) = -j\alpha$ is the complex conjugate of $k_{z2}$.

The correct transmitted wave is therefore $\psi_-(z)$, which corresponds to replacing $k_{z2}$ by $\overline{k_{z2}}$ in every formula involving the transmitted field.

For **propagating** transmitted waves ($k_{z2} \in \mathbb{R}$), $\overline{k_{z2}} = k_{z2}$, so the substitution has no effect — the fix is backward-compatible.

### 8.3 Mathematical Fix

Wherever $k_{z2}$ appears as an outgoing boundary condition or transmitted-field expression, it must be replaced by $\overline{k_{z2}}$:

| Location | Wrong formula | Correct formula |
|----------|---------------|-----------------|
| Right exterior field | $t\,e^{-jk_{z2}(z-z_2)}$ | $t\,e^{-j\overline{k_{z2}}(z-z_2)}$ |
| ODE initial condition at $z_2$ | $F'(z_2) = -jk_{z2}$ | $F'(z_2) = -j\overline{k_{z2}}$ |
| Airy transfer-matrix BC at $z_2$ | $[t,\,-jk_{z2}t]^\top$ | $[t,\,-j\overline{k_{z2}}t]^\top$ |
| FEM right Robin BC | $A[-1,-1] \mathrel{+}= jk_{z2}$ | $A[-1,-1] \mathrel{+}= j\overline{k_{z2}}$ |

Note that the **left** Robin BC uses $k_{z1}$, which is always real (propagating) in our setup ($\varepsilon_1 = 1$), so it requires no correction.

### 8.4 Physical Interpretation of the ODE Fix

The ODE integrates backward from $z_2$ to $z_1$.
The initial condition at $z_2$ sets which of the two independent solutions of the Helmholtz equation is selected throughout the transition layer.

**Before the fix:** the IC $F'(z_2) = -jk_{z2} = \alpha > 0$ seeds the growing solution $\psi_+$, which is non-physical. Although the integration remains mathematically well-posed (it tracks a perfectly valid linear combination of Airy functions), the resulting field profile in $z > z_2$ grows exponentially and the extracted phase of $r$ is wrong.

**After the fix:** the IC $F'(z_2) = -j\overline{k_{z2}} = -\alpha < 0$ seeds the decaying solution $\psi_-$. The integration now tracks the physically correct state, and the field profile decays in $z > z_2$ as required.

### 8.5 Why $|r| = 1$ Is Preserved Regardless

For a lossless evanescent transmitted medium ($\varepsilon_2 < 0$, $k_{z2}$ purely imaginary), no real power is transported in the $z$-direction: $\operatorname{Re}(k_{z2}) = 0$ implies $T = 0$ and hence $R = |r|^2 = 1$ by energy conservation, independently of which Airy-function branch is selected.

This means the wrong IC still produces $|r| = 1$ — the energy check passes even with the bug present.
However, the **phase** of $r$ (and the entire field profile inside the transition layer) is computed for the unphysical growing-mode problem and differs from the correct physical value.
The bug is therefore invisible to amplitude-only diagnostics and only manifests in phase plots or field visualisations that extend into $z > z_2$.

### 8.6 Files Modified

| File | Change |
|------|--------|
| `main.py` | ODE IC: `[1, 0, kz2_im, -kz2_re]` → `[1, 0, -kz2_im, -kz2_re]` |
| `main.py` | `sweep_ode` IC: `jnp.imag(kz2)` → `-jnp.imag(kz2)` |
| `main.py` | Right exterior field: `exp(-1j*kz2*dz)` → `exp(-1j*conj(kz2)*dz)` |
| `main.py` | FEM Robin BC (P1 & P2, TE & TM): `1j*kz2` → `1j*conj(kz2)` |
| `airy_solution.py` | Transfer-matrix P, Q coefficients: `kz2` → `conj(kz2)` |
| `airy_solution.py` | Airy BC vector at $z_2$: `-1j*kz2*t` → `-1j*conj(kz2)*t` |
| `airy_solution.py` | `full_field_v_profile` transmitted field: same substitution |
| `plasma_reflection.ipynb` | `full_airy_field_plasma` transmitted field: same substitution |
