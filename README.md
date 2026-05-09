# 1D Helmholtz EM Solver

**APC 523 Final Project** — Zhiping Li, Xuyang Xu

Numerical solver for 1D Helmholtz EM scattering at a dielectric transition layer.
Two independent methods: ODE shooting (diffrax/JAX) and FEM (JAX + scipy.sparse).
Both support GPU acceleration via JAX.

---

## Repository layout

```
Final/
  main.py                   ← all solver logic (HelmholtzSolver1D class)
  requirements.txt          ← pinned Python dependencies
  helmholtz_test.ipynb      ← main demo notebook (all test cases)
  example/
    airy_comparison.ipynb   ← convergence & stability study vs Airy solution
    solver_comparison.ipynb ← five ODE solvers compared
    gradient_effect.ipynb   ← GPU sweep: R(θ) vs layer width δ
    plasma_reflection.ipynb ← plasma density diagnostics application
    airy_solution.py        ← Airy analytical reference implementation
  doc/
    solver_report.md        ← full written report
    slides.tex / slides.pdf ← presentation slides
```

---

## Environment setup (Princeton cluster with A100 GPU)

The code requires JAX with CUDA support. On the Princeton cluster the
pre-built `GPU_Python` conda environment already contains all dependencies.

### Activate the environment

```bash
module load anaconda3/2024.10
conda activate GPU_Python
```

### Verify GPU is visible to JAX

```python
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]  (or similar A100 entry)
```

### Install from scratch (if the conda env is unavailable)

```bash
# 1. Create a new env
conda create -n helmholtz_env python=3.11 -y
conda activate helmholtz_env

# 2. Install JAX with CUDA 12 support
pip install "jax[cuda12]"

# 3. Install remaining dependencies
pip install -r requirements.txt
```

Pinned versions used during development (see `requirements.txt`):

| Package   | Version |
|-----------|---------|
| jax       | 0.9.2   |
| jaxlib    | 0.9.2   |
| diffrax   | 0.7.2   |
| equinox   | 0.13.7  |
| numpy     | 2.4.4   |
| scipy     | 1.17.1  |
| matplotlib| 3.10.8  |

---

## Running on the cluster

### Interactive GPU session

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G \
       --gres=gpu:1 --time=01:00:00 --partition=gpu
module load anaconda3/2024.10
conda activate GPU_Python
jupyter notebook --no-browser --port=8888
```

Then forward the port from your laptop:

```bash
ssh -N -L 8888:localhost:8888 <netid>@<cluster-login-node>
```

and open `http://localhost:8888` in your browser.

### Batch job (Slurm)

Save the following as `run.slurm` and submit with `sbatch run.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=helmholtz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=helmholtz_%j.out

module load anaconda3/2024.10
conda activate GPU_Python

# Run a notebook non-interactively
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 \
    example/airy_comparison.ipynb \
    --output example/airy_comparison_out.ipynb
```

---

## Quick-start Python example

```python
from main import HelmholtzSolver1D
import numpy as np

# Define a smooth tanh permittivity profile
def eps_tanh(z, eps1=1.0, eps2=4.0, delta=0.5):
    return (eps1 + eps2) / 2 + (eps2 - eps1) / 2 * np.tanh(z / delta)

# Create solver (TE polarisation, 45° incidence)
solver = HelmholtzSolver1D(eps_tanh, z1=-2.0, z2=2.0,
                           theta_deg=45.0, pol='S', lambda0=1.0)

# ODE solve (GPU-accelerated via JAX/diffrax)
result = solver.solve_ode()
solver.energy_check(result['r'], result['t'])   # R + T ≈ 1
solver.plot_field(result)

# GPU-parallel angle sweep (jax.vmap over θ)
from main import sweep_ode
thetas = np.linspace(0, 85, 200)
R, T = sweep_ode(solver, thetas)
```

### Resolution defaults

| Parameter        | Default                                              |
|-----------------|------------------------------------------------------|
| `res_trans` ODE | $50 \times \max\sqrt{|\varepsilon_r(z)|}$ cells/λ₀  |
| `res_trans` FEM | $100 \times \max\sqrt{|\varepsilon_r(z)|}$ cells/λ₀ |
| `res_ext`       | `res_trans / 2`                                      |

---

## Notes on GPU performance

- JAX JIT-compiles the ODE integration on the first call; subsequent calls are fast.
- For parameter sweeps, `sweep_ode` uses `jax.vmap` to execute all angles in a
  single GPU kernel, achieving **10–50× speedup** over a Python loop.
- 64-bit complex arithmetic (`jax_enable_x64 = True`) is enabled automatically
  by `main.py`; ensure your GPU supports FP64 (A100 does).
- If no GPU is found, JAX falls back to CPU automatically with a printed warning.
