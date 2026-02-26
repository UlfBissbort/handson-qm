# Hands-On Quantum Mechanics

Numerical quantum mechanics from scratch. Each notebook sets up a physical system, solves the Schrödinger equation on a grid, and visualizes the results — no black-box libraries, no hand-waving.

## Notebooks

| # | Topic | Key ideas |
|---|-------|-----------|
| 01 | [Harmonic oscillator dynamics](01_harmonic_oscillator_dyanmics.py) | Grid discretization, finite-difference Hamiltonian, ODE time evolution, expectation values, probability current, Husimi Q function |

## Prerequisites

- Python 3.10+
- NumPy, SciPy, Matplotlib

```
pip install numpy scipy matplotlib
```

## Running

Each notebook exists in two formats:
- **`.py`** — the source of truth. Uses `#%%` cell delimiters. Open in VS Code and run cells with **Shift+Enter** (requires the Python extension). Also runs as a plain script: `python 01_harmonic_oscillator_dyanmics.py`
- **`.ipynb`** — exported Jupyter notebook with the same content. Plots and animations are stripped (matplotlib/Jupyter uses inefficient encodings that bloat file size). Re-run all cells to regenerate output.

## Approach

Every concept is introduced with prose, then implemented in code, then visualized.

No prior quantum mechanics experience required — just comfort with Python and basic linear algebra.
