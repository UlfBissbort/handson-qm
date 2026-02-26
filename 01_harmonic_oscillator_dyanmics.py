#%%
"""
# Dynamics of a Quantum Harmonic Oscillator

A particle trapped in a harmonic potential is one of the most important systems
in quantum mechanics — it shows up everywhere from molecular vibrations to
quantum optics. In this notebook we'll simulate one from scratch: set up the
Schrödinger equation on a spatial grid, evolve it forward in time, and watch
the probability density slosh back and forth.

The time-dependent Schrödinger equation is:

$$
i\hbar \frac{\partial}{\partial t}\psi(x,t) = \hat{H}\psi(x,t)
$$

where the Hamiltonian $\hat{H} = \hat{T} + \hat{V}$ splits into kinetic and
potential energy. Our job is to turn this into something a computer can solve.
"""

#%%
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import matplotlib.pyplot as plt

#%%
"""
## The Setup

We're simulating a single particle of mass $m$ in a harmonic potential:

$$
V(x) = \tfrac{1}{2} m \omega^2 x^2
$$

The parameter $\omega$ controls how tight the trap is — higher $\omega$ means
a steeper potential well and faster oscillations.

We work in **natural units** where $\hbar = 1$ and $m = 1$. This isn't just
laziness: it means lengths are measured in units of
$\sqrt{\hbar / (m\omega)}$ and times in units of $1/\omega$, which keeps all
our numbers close to 1.

Below are the physical parameters. The `packet_width` scales the initial
Gaussian relative to the ground state width $\sigma_0 = \sqrt{\hbar/(m\omega)}$.
When `packet_width = 1.0`, the wave packet has exactly the ground state shape
— if you displace it from the center it will oscillate back and forth without
changing shape (try it!). Values other than 1 cause the packet to "breathe" as
it moves.
"""

#%%
# Physical parameters
hbar = 1.0
m = 1.0
omega = 1.0                # oscillator frequency

packet_width = 1.0          # 1.0 = ground state width (shape-preserving)
x0_displacement = 5.0       # initial displacement from center

# Derived
sigma0 = np.sqrt(hbar / (m * omega))   # ground state width
sigma = packet_width * sigma0           # actual packet width

print(f"Ground state width σ₀ = {sigma0:.4f}")
print(f"Initial packet width σ = {sigma:.4f}")
print(f"Initial displacement   = {x0_displacement}")

#%%
"""
## Discretizing Space

We can't represent a continuous function $\psi(x)$ on a computer — we need to
pick a finite set of grid points $x_0, x_1, \ldots, x_{N-1}$ and store the
values of $\psi$ at those points. The wave function becomes a vector:

$$
\vec{\psi} = \begin{pmatrix} \psi(x_0) \\ \psi(x_1) \\ \vdots \\ \psi(x_{N-1}) \end{pmatrix}
$$

The grid spacing $\Delta x = x_1 - x_0$ controls the resolution. Too coarse
and we miss the fine structure of $\psi$; too fine and computations get slow.
For our harmonic oscillator, the wave function stays localized near the center,
so we need the grid wide enough to contain it but not absurdly large.

A good rule of thumb: the classical turning point (where $V(x) = E$) for our
displaced packet is roughly at $x = x_0$. We set the grid a few widths beyond
that — enough to safely contain the packet at all times, but not so wide that
the potential at the grid edges creates unnecessarily large eigenvalues
(which would slow down our ODE solver).
"""

#%%
# Spatial grid
Nx = 512
L = abs(x0_displacement) + 8 * sigma           # just wide enough
x = np.linspace(-L, L, Nx)
dx = x[1] - x[0]

print(f"Grid: {Nx} points from {x[0]:.1f} to {x[-1]:.1f}, dx = {dx:.4f}")

#%%
"""
## The Harmonic Potential

Let's define and plot the potential. Nothing fancy here — just the parabola
$V(x) = \tfrac{1}{2} m \omega^2 x^2$.
"""

#%%
V = 0.5 * m * omega**2 * x**2

plt.figure(figsize=(8, 3))
plt.plot(x, V, 'k-', linewidth=1.5)
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Harmonic oscillator potential')
plt.ylim(0, 30)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
"""
## Turning Derivatives into Matrices

Here's the central trick of numerical quantum mechanics: **derivatives become
matrices**.

When you learned calculus, the derivative of $f$ at a point was a limit. On our
discrete grid, we replace that limit with a finite difference. The simplest
approximation for the *second* derivative at grid point $x_i$ is:

$$
\frac{d^2\psi}{dx^2}\bigg|_{x_i}
\approx \frac{\psi_{i+1} - 2\psi_i + \psi_{i-1}}{\Delta x^2}
$$

(You can derive this yourself by adding the Taylor expansions for
$\psi(x_i + \Delta x)$ and $\psi(x_i - \Delta x)$ — the first-derivative
terms cancel and you're left with the second derivative.)

Now look at what this formula does: it takes three entries of the vector
$\vec\psi$ and combines them into one number. That's a matrix-vector product!
Written out for all grid points:

$$
\frac{d^2}{dx^2}\vec\psi \ \approx\
\frac{1}{\Delta x^2}
\begin{pmatrix}
-2 &  1 &    &        &    \\\\
 1 & -2 &  1 &        &    \\\\
   &  1 & -2 & \ddots &    \\\\
   &    & \ddots & -2 &  1 \\\\
   &    &    &  1 & -2
\end{pmatrix}
\begin{pmatrix} \psi_0 \\\\ \psi_1 \\\\ \vdots \\\\ \\\\ \psi_{N-1} \end{pmatrix}
$$

This **tridiagonal matrix** is sparse — most entries are zero. We use
`scipy.sparse.diags` to build it efficiently. The kinetic energy operator is:

$$
\hat{T} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2}
$$

so we just multiply our matrix by $-\hbar^2 / (2m)$.
"""

#%%
# Kinetic energy operator (sparse tridiagonal matrix)
main_diag = -2.0 / dx**2 * np.ones(Nx)
off_diag  =  1.0 / dx**2 * np.ones(Nx - 1)

T_kinetic = -(hbar**2 / (2 * m)) * diags(
    [off_diag, main_diag, off_diag],
    [-1, 0, 1],
    shape=(Nx, Nx),
    dtype=complex
)

# Potential energy operator (diagonal matrix)
V_operator = diags(V, 0, shape=(Nx, Nx), dtype=complex)

# Full Hamiltonian
H = T_kinetic + V_operator

print(f"Hamiltonian: {H.shape[0]}×{H.shape[1]} sparse matrix, {H.nnz} non-zero entries")

#%%
"""
## The Initial Wave Packet

We start with a Gaussian wave packet centered at position $x_0$, with zero
initial momentum:

$$
\psi(x, 0) = \left(\frac{1}{2\pi\sigma^2}\right)^{1/4}
\exp\!\left(-\frac{(x - x_0)^2}{4\sigma^2}\right)
$$

The width $\sigma$ determines how spread out the packet is. For the harmonic
oscillator, the ground state has a specific width
$\sigma_0 = \sqrt{\hbar/(m\omega)}$. When we use exactly this width (i.e.
`packet_width = 1.0`), something special happens: the displaced packet
oscillates back and forth *without changing its shape*. For any other width, the
packet will "breathe" — alternately squeezing and stretching as it oscillates.

Let's create our initial state and make sure it's properly normalized (the
total probability $\int|\psi|^2 dx$ must equal 1):
"""

#%%
# Initial Gaussian wave packet (displaced, zero momentum)
# Important: must be complex — solve_ivp needs complex input to evolve complex-valued psi
psi_0 = ((1 / (2 * np.pi * sigma**2))**0.25
         * np.exp(-(x - x0_displacement)**2 / (4 * sigma**2))
         + 0j)  # make it complex

# Normalize numerically (belt-and-suspenders)
norm = np.sum(np.abs(psi_0)**2) * dx
psi_0 /= np.sqrt(norm)

# Verify
print(f"Initial norm: {np.sum(np.abs(psi_0)**2) * dx:.10f}")

# Plot the initial state inside the potential
fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(x, V, 'k-', linewidth=1, label='V(x)')
ax1.set_ylabel('V(x)', color='k')
ax1.set_ylim(0, 30)

ax2 = ax1.twinx()
ax2.fill_between(x, np.abs(psi_0)**2, alpha=0.5, color='steelblue', label=r'$|\psi|^2$')
ax2.plot(x, np.abs(psi_0)**2, color='steelblue', linewidth=1.5)
ax2.set_ylabel(r'$|\psi(x,0)|^2$', color='steelblue')

ax1.set_xlabel('x')
ax1.set_title('Initial wave packet in the harmonic potential')
ax1.set_xlim(-L, L)
plt.tight_layout()
plt.show()

#%%
"""
## Solving the Schrödinger Equation

The Schrödinger equation $i\hbar \partial_t\psi = \hat{H}\psi$ is a
first-order ODE in time. We can rewrite it as:

$$
\frac{d\vec\psi}{dt} = -\frac{i}{\hbar} H \vec\psi
$$

This is just a matrix-vector ODE — exactly the kind of thing `scipy.integrate.solve_ivp`
is built for. We hand it the right-hand side function, the initial state, and
a time span, and it returns $\psi(t)$ at the requested times.

We'll simulate for a few full oscillation periods ($T_{\mathrm{osc}} = 2\pi / \omega$)
so we can see the packet go back and forth multiple times.

One subtlety: the Schrödinger equation is oscillatory (the eigenvalues of $-iH/\hbar$
are purely imaginary), which means the RK45 solver's adaptive step size control can
be fooled into taking steps that are too large. We set `max_step` explicitly to keep
the solver honest.
"""

#%%
# Time propagation
T_osc = 2 * np.pi / omega          # one oscillation period
T_total = 3 * T_osc                # simulate 3 full periods
Nt = 300                           # number of output snapshots
t_eval = np.linspace(0, T_total, Nt)

def schrodinger_rhs(t, psi):
    return (-1j / hbar) * (H @ psi)

# max_step is essential: the Schrödinger equation has large imaginary eigenvalues
# and RK45's error estimator doesn't detect the resulting instability
max_eigenvalue_estimate = 2 * hbar**2 / (m * dx**2) + np.max(V)
max_dt = 2.0 / max_eigenvalue_estimate  # conservative stability limit
print(f"Estimated max stable dt: {max_dt:.5f}")

print(f"Propagating for {T_total:.2f} time units ({T_total/T_osc:.0f} periods)...")
print(f"Using {Nt} snapshots, dt_output = {t_eval[1]-t_eval[0]:.4f}")

solution = solve_ivp(
    schrodinger_rhs,
    [0, T_total],
    psi_0,
    t_eval=t_eval,
    method='RK45',
    max_step=max_dt,
    rtol=1e-8,
    atol=1e-10,
)

print(f"Solver status: {solution.message}")
print(f"Number of RHS evaluations: {solution.nfev}")

#%%
"""
## Norm Conservation Check

A correct time evolution must conserve the norm of the wave function — total
probability can't appear or disappear. Let's check how well our ODE solver did:
"""

#%%
# Check norm at start and end
norm_start = np.sum(np.abs(solution.y[:, 0])**2) * dx
norm_end   = np.sum(np.abs(solution.y[:, -1])**2) * dx

print(f"Norm at t=0:     {norm_start:.10f}")
print(f"Norm at t=T:     {norm_end:.10f}")
print(f"Relative change: {abs(norm_end - norm_start) / norm_start:.2e}")

# Quick diagnostic: track the expectation value of x over time
x_expect = np.array([np.sum(x * np.abs(solution.y[:, i])**2) * dx for i in range(Nt)])
print(f"\n<x> at t=0: {x_expect[0]:.3f}")
print(f"<x> at t=T/4: {x_expect[Nt//4]:.3f}")
print(f"<x> at t=T/2: {x_expect[Nt//2]:.3f}")
print(f"<x> range: [{x_expect.min():.3f}, {x_expect.max():.3f}]")

#%%
"""
## Animating the Wave Packet

Now for the fun part — let's watch the probability density $|\psi(x,t)|^2$
evolve in time. We overlay the harmonic potential so you can see the packet
oscillating inside the well.
"""

#%%
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

pdf = np.abs(solution.y)**2

fig, ax = plt.subplots(figsize=(9, 4))

# Static: potential (rescaled to fit on the same axis)
V_scale = np.max(pdf) / 20  # scale factor so potential is visible but not dominant
ax.plot(x, V * V_scale, 'k-', linewidth=0.8, alpha=0.4, label='V(x) (scaled)')
ax.fill_between(x, V * V_scale, alpha=0.05, color='k')

# Dynamic: probability density (line only — fill_between breaks jshtml)
line, = ax.plot(x, pdf[:, 0], color='steelblue', linewidth=1.5, label=r'$|\psi|^2$')

ax.set_xlim(-L, L)
ax.set_ylim(0, np.max(pdf) * 1.1)
ax.set_xlabel('x')
ax.set_ylabel(r'$|\psi(x,t)|^2$')
title = ax.set_title(f't = 0.00  (period 0.00)')
ax.grid(True, alpha=0.2)
ax.legend(loc='upper right')

def update(i):
    line.set_ydata(pdf[:, i])
    title.set_text(f't = {t_eval[i]:.2f}  (period {t_eval[i]/T_osc:.2f})')
    return line, title

anim = FuncAnimation(fig, update, frames=Nt, interval=33, blit=True)
plt.close()

HTML(anim.to_jshtml())

