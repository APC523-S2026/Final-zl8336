Below is the complete, formal theoretical document in LaTeX format. This version updates **Section 4** to explicitly include the algebraic derivation of the coefficients $c_1, c_2, r$, and $t$ using the four boundary condition equations and the Wronskian property.

---

# Theoretical Analysis of Electromagnetic Wave Propagation in Non-Homogeneous Transition Layers

## 1. Problem Statement
Consider a one-dimensional non-homogeneous medium where the relative permittivity $\epsilon_r(z)$ varies continuously from $\epsilon_{r1}$ to $\epsilon_{r2}$ within the interval $z \in [z_1, z_2]$. The regions $z < z_1$ (Region 1) and $z > z_2$ (Region 2) are semi-infinite homogeneous media.

A plane electromagnetic wave is incident from Region 1 at an angle $\theta_i$. The system exhibits translational symmetry along the $y$-axis ($\partial/\partial y = 0$). We assume a time-harmonic field of the form $e^{-j\omega_0 t}$ and a spatial phase convention of $e^{j(\mathbf{k} \cdot \mathbf{r} - \omega_0 t)}$.

## 2. Derivation of Wave Equations
From Maxwell’s equations in a source-free, non-magnetic ($\mu_r = 1$) gradient medium:
$$\nabla \times \mathbf{E} = j\omega_0\mu_0 \mathbf{H}, \quad \nabla \times \mathbf{H} = -j\omega_0\epsilon_0 \epsilon_r(z) \mathbf{E}$$
The tangential wavenumber $k_x = k_0\sqrt{\epsilon_{r1}}\sin\theta_i$ is conserved.

### 2.1 TE Polarization (s-polarization)
For $\mathbf{E} = (0, E_y, 0)$, the wave equation is:
$$\frac{d^2 E_y}{dz^2} + \left[ k_0^2 \epsilon_r(z) - k_x^2 \right] E_y = 0$$

### 2.2 TM Polarization (p-polarization)
For $\mathbf{H} = (0, H_y, 0)$, the wave equation is:
$$\frac{d^2 H_y}{dz^2} - \frac{1}{\epsilon_r(z)} \frac{d\epsilon_r}{dz} \frac{dH_y}{dz} + \left[ k_0^2 \epsilon_r(z) - k_x^2 \right] H_y = 0$$

## 3. Analytical Solution for Linear Gradients
For normal incidence ($k_x = 0$) and a linear gradient $\epsilon_r(z) = g(z - z_0)$, where $g = \frac{\epsilon_{r2} - \epsilon_{r1}}{z_2 - z_1}$, we introduce the dimensionless variable:
$$\xi(z) = -(k_0^2 g)^{1/3} (z - z_0)$$
The Helmholtz equation transforms into the **Airy Equation**: $\frac{d^2 E}{d\xi^2} - \xi E = 0$. The general solution is:
$$E(z) = c_1 \text{Ai}(\xi(z)) + c_2 \text{Bi}(\xi(z))$$

## 4. Resolution of Coefficients and Transfer Matrix
To determine the unknown coefficients $\{c_1, c_2, r, t\}$, we apply boundary conditions at $z_1$ and $z_2$. Let $\alpha = (k_0^2 g)^{1/3}$. The derivative transformation is $\frac{dE}{dz} = -\alpha \frac{dE}{d\xi}$.

### 4.1 The System of Equations
Matching the field and its derivative at the interfaces yields:
1. **At $z_1$ ($\xi = \xi_1$):**
   $$c_1 \text{Ai}(\xi_1) + c_2 \text{Bi}(\xi_1) = 1 + r$$
   $$c_1 \text{Ai}'(\xi_1) + c_2 \text{Bi}'(\xi_1) = \frac{jk_1}{\alpha}(1 - r)$$
2. **At $z_2$ ($\xi = \xi_2$):**
   $$c_1 \text{Ai}(\xi_2) + c_2 \text{Bi}(\xi_2) = t$$
   $$c_1 \text{Ai}'(\xi_2) + c_2 \text{Bi}'(\xi_2) = \frac{jk_2}{\alpha}t$$

### 4.2 Solving for $c_1$ and $c_2$
Using the equations at $z_2$ in matrix form:
$$\begin{pmatrix} \text{Ai}(\xi_2) & \text{Bi}(\xi_2) \\ \text{Ai}'(\xi_2) & \text{Bi}'(\xi_2) \end{pmatrix} \begin{pmatrix} c_1 \\ c_2 \end{pmatrix} = \begin{pmatrix} 1 \\ \frac{jk_2}{\alpha} \end{pmatrix} t$$
Applying the Wronskian property $W\{\text{Ai}, \text{Bi}\} = \text{Ai}\text{Bi}' - \text{Ai}'\text{Bi} = 1/\pi$:
$$\begin{pmatrix} c_1 \\ c_2 \end{pmatrix} = \pi \begin{pmatrix} \text{Bi}'(\xi_2) & -\text{Bi}(\xi_2) \\ -\text{Ai}'(\xi_2) & \text{Ai}(\xi_2) \end{pmatrix} \begin{pmatrix} 1 \\ \frac{jk_2}{\alpha} \end{pmatrix} t$$
Which gives the coefficients in terms of $t$:
$$c_1 = \pi \left[ \text{Bi}'(\xi_2) - \frac{jk_2}{\alpha} \text{Bi}(\xi_2) \right] t, \quad c_2 = -\pi \left[ \text{Ai}'(\xi_2) - \frac{jk_2}{\alpha} \text{Ai}(\xi_2) \right] t$$

### 4.3 Transfer Matrix Elements
The transfer matrix $\mathbf{M}$ relates the states at $\xi_1$ and $\xi_2$:
$$\begin{pmatrix} E(\xi_1) \\ E'(\xi_1) \end{pmatrix} = \underbrace{K(\xi_1) K(\xi_2)^{-1}}_{\mathbf{M}} \begin{pmatrix} E(\xi_2) \\ E'(\xi_2) \end{pmatrix}$$
The elements are:
* $m_{11} = \pi \left[ \text{Ai}(\xi_1)\text{Bi}'(\xi_2) - \text{Bi}(\xi_1)\text{Ai}'(\xi_2) \right]$
* $m_{12} = \pi \left[ \text{Bi}(\xi_1)\text{Ai}(\xi_2) - \text{Ai}(\xi_1)\text{Bi}(\xi_2) \right]$
* $m_{21} = \pi \left[ \text{Ai}'(\xi_1)\text{Bi}'(\xi_2) - \text{Bi}'(\xi_1)\text{Ai}'(\xi_2) \right]$
* $m_{22} = \pi \left[ \text{Bi}'(\xi_1)\text{Ai}(\xi_2) - \text{Ai}'(\xi_1)\text{Bi}(\xi_2) \right]$

### 4.4 Derivation of $t$ and $r$

Substituting $c_1$ and $c_2$ from Section 4.2 into the two boundary equations at $z_1$ (equations 1 and 2 in Section 4.1), we use the transfer matrix elements of Section 4.3 to simplify. Specifically:

$$c_1 \text{Ai}(\xi_1) + c_2 \text{Bi}(\xi_1) = \pi t \left[ \text{Bi}'(\xi_2)\text{Ai}(\xi_1) - \text{Ai}'(\xi_2)\text{Bi}(\xi_1) - \frac{jk_2}{\alpha}\left(\text{Bi}(\xi_2)\text{Ai}(\xi_1) - \text{Ai}(\xi_2)\text{Bi}(\xi_1)\right) \right]$$

Recognizing the transfer matrix elements:
$$= \left( m_{11} + \frac{jk_2}{\alpha} m_{12} \right) t$$

Similarly for the derivative equation at $z_1$:
$$c_1 \text{Ai}'(\xi_1) + c_2 \text{Bi}'(\xi_1) = \left( m_{21} + \frac{jk_2}{\alpha} m_{22} \right) t$$

The boundary conditions at $z_1$ thus reduce to the $2\times 2$ linear system in $\{r, t\}$:

$$\text{(A):} \quad 1 + r = \left( m_{11} + \frac{jk_2}{\alpha} m_{12} \right) t$$
$$\text{(B):} \quad \frac{jk_1}{\alpha}(1 - r) = \left( m_{21} + \frac{jk_2}{\alpha} m_{22} \right) t$$

Adding $\frac{jk_1}{\alpha} \times \text{(A)}$ and $\text{(B)}$:
$$\frac{2jk_1}{\alpha} = \left[ \frac{jk_1}{\alpha}\left( m_{11} + \frac{jk_2}{\alpha} m_{12} \right) + \left( m_{21} + \frac{jk_2}{\alpha} m_{22} \right) \right] t$$

Multiplying through by $\alpha$ and defining the common denominator:
$$D \equiv \alpha m_{21} - \frac{k_1 k_2}{\alpha} m_{12} + j\left(k_1 m_{11} + k_2 m_{22}\right)$$

we obtain:
$$t = \frac{2jk_1}{D}$$

Substituting back into (A) and rearranging $r = (m_{11} + jk_2 m_{12}/\alpha)\,t - 1$:
$$r = \frac{-\alpha m_{21} - \dfrac{k_1 k_2}{\alpha} m_{12} + j\left(k_1 m_{11} - k_2 m_{22}\right)}{D}$$

### 4.5 Complete Explicit Formulas

Expanding $D$ and the transfer matrix elements fully in terms of Airy function values at $\xi_1$ and $\xi_2$:

$$\boxed{D = \pi\alpha\!\left[\text{Ai}'(\xi_1)\text{Bi}'(\xi_2) - \text{Bi}'(\xi_1)\text{Ai}'(\xi_2)\right] - \frac{\pi k_1 k_2}{\alpha}\!\left[\text{Bi}(\xi_1)\text{Ai}(\xi_2) - \text{Ai}(\xi_1)\text{Bi}(\xi_2)\right]}$$
$$\hspace{2.2cm} +\; j\pi k_1\!\left[\text{Ai}(\xi_1)\text{Bi}'(\xi_2) - \text{Bi}(\xi_1)\text{Ai}'(\xi_2)\right] + j\pi k_2\!\left[\text{Bi}'(\xi_1)\text{Ai}(\xi_2) - \text{Ai}'(\xi_1)\text{Bi}(\xi_2)\right]$$

The four unknowns are then given explicitly by:

$$\boxed{t = \frac{2jk_1}{D}}$$

$$\boxed{r = \frac{-\alpha m_{21} - \dfrac{k_1 k_2}{\alpha} m_{12} + j\!\left(k_1 m_{11} - k_2 m_{22}\right)}{D}}$$

$$\boxed{c_1 = \frac{2\pi jk_1}{D}\!\left[\text{Bi}'(\xi_2) - \frac{jk_2}{\alpha}\text{Bi}(\xi_2)\right]}$$

$$\boxed{c_2 = -\frac{2\pi jk_1}{D}\!\left[\text{Ai}'(\xi_2) - \frac{jk_2}{\alpha}\text{Ai}(\xi_2)\right]}$$

where $k_1 = k_0\sqrt{\epsilon_{r1}}$, $k_2 = k_0\sqrt{\epsilon_{r2}}$, $\alpha = (k_0^2 g)^{1/3}$, and the transfer matrix elements $m_{ij}$ are defined in Section 4.3.