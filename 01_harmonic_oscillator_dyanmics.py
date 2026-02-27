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
\frac{d^2}{dx^2}\vec\psi \approx
\frac{1}{\Delta x^2}
\begin{pmatrix}
-2 &  1 &    &        &    \\
 1 & -2 &  1 &        &    \\
   &  1 & -2 & \ddots &    \\
   &    & \ddots & -2 &  1 \\
   &    &    &  1 & -2
\end{pmatrix}
\begin{pmatrix} \psi_0 \\ \psi_1 \\ \vdots \\ \psi_{N-1} \end{pmatrix}
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
\exp\left(-\frac{(x - x_0)^2}{4\sigma^2}\right)
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

#%%
"""
## Expectation Values

An animation is great for building intuition, but to do quantitative physics we
need numbers we can track over time. The central tool for this is the
**expectation value**.

For any observable $\hat{A}$ (position, momentum, energy, ...), the expectation
value at time $t$ is:

$$
\langle A \rangle(t) = \int_{-\infty}^{\infty} \psi^*(x,t) \hat{A} \psi(x,t) dx
$$

This is the average you'd get if you prepared the same quantum state many times
and measured $A$ each time. It's not the result of a single measurement — it's
a statistical average over many identical experiments.

On our discrete grid, the integral becomes a sum:

$$
\langle A \rangle \approx \sum_{i=0}^{N-1} \psi_i^* (A\vec\psi)_i \Delta x
$$

where $A$ is the matrix representation of $\hat A$ acting on the vector $\vec\psi$.

The simplest case is **position**. Since $\hat{x}$ just multiplies by $x$, the
matrix $A$ is diagonal with the grid values $x_i$ on the diagonal. The
expectation value simplifies to:

$$
\langle x \rangle(t) = \sum_i x_i |\psi_i(t)|^2 \Delta x
$$

This is literally the center of mass of the probability distribution. Let's
compute it at every snapshot and plot the result:
"""

#%%
# Compute <x>(t) at each snapshot
x_expect = np.array([
    np.sum(x * np.abs(solution.y[:, i])**2) * dx
    for i in range(Nt)
])

# Classical prediction: x(t) = x0 * cos(omega * t)
x_classical = x0_displacement * np.cos(omega * t_eval)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_eval, x_expect, 'steelblue', linewidth=2, label=r'$\langle x \rangle$ (quantum)')
ax.plot(t_eval, x_classical, 'k--', linewidth=1, alpha=0.6, label=r'$x_0 \cos(\omega t)$ (classical)')
ax.set_xlabel('Time')
ax.set_ylabel(r'$\langle x \rangle$')
ax.set_title('Position expectation value vs. time')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()

print(f"Max deviation from classical: {np.max(np.abs(x_expect - x_classical)):.4f}")

#%%
"""
## Momentum Expectation Value

Position was easy because $\hat{x}$ is just "multiply by $x$" — a diagonal
matrix. Momentum is more interesting. The momentum operator is:

$$
\hat{p} = -i\hbar \frac{d}{dx}
$$

We already know how to turn a second derivative into a matrix. The
*first* derivative works the same way, using the **central difference**:

$$
\frac{d\psi}{dx}\bigg|_{x_i} \approx \frac{\psi_{i+1} - \psi_{i-1}}{2 \Delta x}
$$

As a matrix, this is antisymmetric: $+1/(2\Delta x)$ on the upper off-diagonal
and $-1/(2\Delta x)$ on the lower, with zeros on the main diagonal.

Once we have the derivative matrix $D$, the momentum operator is $P = -i\hbar D$,
and the expectation value is $\langle p \rangle = \vec\psi^\dagger P \vec\psi \Delta x$.
"""

#%%
# First derivative matrix (central differences)
off_upper = 1.0 / (2 * dx) * np.ones(Nx - 1)
off_lower = -1.0 / (2 * dx) * np.ones(Nx - 1)
D = diags([off_lower, off_upper], [-1, 1], shape=(Nx, Nx), dtype=complex)

# Momentum operator P = -i*hbar*D
P = -1j * hbar * D

# Compute <p>(t) at each snapshot
p_expect = np.array([
    np.real(np.sum(np.conj(solution.y[:, i]) * (P @ solution.y[:, i])) * dx)
    for i in range(Nt)
])

# Classical prediction: p(t) = -m*omega*x0*sin(omega*t)
p_classical = -m * omega * x0_displacement * np.sin(omega * t_eval)

print(f"<p> at t=0: {p_expect[0]:.4f} (expected 0.0)")
print(f"Max |<p>|:  {np.max(np.abs(p_expect)):.4f} (expected {m*omega*x0_displacement:.1f})")

#%%
"""
## Wave Packet Width

The position expectation value tells us *where* the packet is on average, but
not how *spread out* it is. For that we need the **standard deviation** of the
position distribution:

$$
\sigma_x(t) = \sqrt{\langle x^2 \rangle - \langle x \rangle^2}
$$

For our coherent state (`packet_width = 1.0`), a remarkable thing happens:
this width stays constant as the packet oscillates. The wave packet doesn't
spread at all — it moves rigidly, like a classical ball. If you change
`packet_width` to something else (try 0.5 or 2.0), you'll see the width
oscillate: the packet "breathes."
"""

#%%
# Compute <x^2>(t) and sigma_x(t)
x2_expect = np.array([
    np.sum(x**2 * np.abs(solution.y[:, i])**2) * dx
    for i in range(Nt)
])
sigma_x = np.sqrt(x2_expect - x_expect**2)

print(f"sigma_x at t=0:   {sigma_x[0]:.4f} (expected {sigma:.4f})")
print(f"sigma_x at t=T/2: {sigma_x[Nt//2]:.4f}")
print(f"sigma_x range:    [{sigma_x.min():.4f}, {sigma_x.max():.4f}]")

#%%
"""
## Energy Expectation Value

Finally, the total energy. Since $\hat{H}$ is the Hamiltonian itself, the
expectation value is:

$$
\langle E \rangle(t) = \vec\psi^\dagger H \vec\psi \Delta x
$$

For a closed quantum system the energy is exactly conserved — it
shouldn't change at all. Any drift in $\langle E \rangle$ is a direct measure
of our numerical error. This makes it a useful diagnostic beyond the norm
check.
"""

#%%
# Compute <E>(t) at each snapshot
E_expect = np.array([
    np.real(np.sum(np.conj(solution.y[:, i]) * (H @ solution.y[:, i])) * dx)
    for i in range(Nt)
])

E0 = E_expect[0]
print(f"<E> at t=0:       {E0:.6f}")
print(f"<E> at t=T:       {E_expect[-1]:.6f}")
print(f"Relative drift:   {abs(E_expect[-1] - E0) / E0:.2e}")

# Analytical energy for a coherent state: E = (n + 1/2)*hbar*omega where
# n_mean = x0^2 * m*omega / (2*hbar) for a displaced ground state
E_analytical = 0.5 * m * omega**2 * x0_displacement**2 + 0.5 * hbar * omega
print(f"Analytical E:     {E_analytical:.6f}")

#%%
"""
## The Full Picture

Let's put everything together in one plot. Four quantities, four panels: position,
momentum, width, and energy — each telling a different part of the story.
"""

#%%
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

# Position
ax = axes[0, 0]
ax.plot(t_eval, x_expect, 'steelblue', linewidth=1.5)
ax.plot(t_eval, x_classical, 'k--', linewidth=0.8, alpha=0.5)
ax.set_ylabel(r'$\langle x \rangle$')
ax.set_title('Position')
ax.grid(True, alpha=0.2)
ax.axhline(0, color='k', linewidth=0.3)

# Momentum
ax = axes[0, 1]
ax.plot(t_eval, p_expect, 'coral', linewidth=1.5)
ax.plot(t_eval, p_classical, 'k--', linewidth=0.8, alpha=0.5)
ax.set_ylabel(r'$\langle p \rangle$')
ax.set_title('Momentum')
ax.grid(True, alpha=0.2)
ax.axhline(0, color='k', linewidth=0.3)

# Width
ax = axes[1, 0]
ax.plot(t_eval, sigma_x, 'seagreen', linewidth=1.5)
ax.axhline(sigma, color='k', linestyle='--', linewidth=0.8, alpha=0.5, label=r'$\sigma_0$')
ax.set_xlabel('Time')
ax.set_ylabel(r'$\sigma_x$')
ax.set_title('Wave packet width')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Energy
ax = axes[1, 1]
ax.plot(t_eval, E_expect, 'goldenrod', linewidth=1.5)
ax.axhline(E_analytical, color='k', linestyle='--', linewidth=0.8, alpha=0.5, label='analytical')
ax.set_xlabel('Time')
ax.set_ylabel(r'$\langle E \rangle$')
ax.set_title('Total energy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

fig.suptitle('Harmonic oscillator dynamics — expectation values', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

#%%
"""
## How Motion is Encoded: The Phase of $\psi$

In classical mechanics, position and velocity are independent quantities — you
specify both to define the state. In quantum mechanics, the wave function
$\psi(x,t)$ encodes *everything*: both where the particle is likely to be
found *and* how it's moving.

The secret is in the **complex phase**. We can always write:

$$
\psi(x,t) = |\psi(x,t)| e^{i\phi(x,t)}
$$

The amplitude $|\psi|$ determines the probability density (where the particle
is). The phase $\phi(x)$ determines the motion. Specifically, the local
wavenumber $k(x) = \partial\phi/\partial x$ is related to the local momentum
by $p = \hbar k$. When the wave packet moves to the right, $\phi(x)$ increases
with $x$ — the real and imaginary parts of $\psi$ oscillate rapidly in space.
When the packet is momentarily at rest (at the classical turning points), the
phase is nearly flat and $\psi$ is approximately real.

Let's see this directly. We'll look at the real and imaginary parts of $\psi$
at four key moments in one oscillation cycle:
"""

#%%
# Snapshots at four key times: t = 0, T/4, T/2, 3T/4
# Use time-based index lookup (not Nt fractions) so the correct times are
# sampled regardless of how many periods T_total covers.
snapshot_times = [0, T_osc/4, T_osc/2, 3*T_osc/4]
snapshot_indices = [np.argmin(np.abs(t_eval - t)) for t in snapshot_times]
snapshot_labels = ['t = 0 (at rest, x = +5)', 't = T/4 (max speed, x = 0)',
                   't = T/2 (at rest, x = -5)', 't = 3T/4 (max speed, x = 0)']

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

for ax, idx, label in zip(axes, snapshot_indices, snapshot_labels):
    psi_t = solution.y[:, idx]
    ax.fill_between(x, np.abs(psi_t)**2, alpha=0.15, color='gray')
    ax.plot(x, np.real(psi_t), 'steelblue', linewidth=1.2, label=r'Re($\psi$)')
    ax.plot(x, np.imag(psi_t), 'coral', linewidth=1.2, label=r'Im($\psi$)')
    ax.plot(x, np.abs(psi_t), 'k-', linewidth=0.5, alpha=0.4, label=r'$|\psi|$')
    ax.set_ylabel(r'$\psi(x)$')
    ax.set_title(label, fontsize=10)
    ax.set_xlim(-L, L)
    ax.set_ylim(-0.8, 0.8)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.grid(True, alpha=0.15)

axes[0].legend(loc='upper left', fontsize=8, ncol=3)
axes[-1].set_xlabel('x')
fig.suptitle(r'Real and imaginary parts of $\psi$ during one period', fontsize=12, y=1.01)
plt.tight_layout()
plt.show()

#%%
"""
Notice the pattern:
- At $t = 0$ and $t = T/2$, the packet is at the turning points (maximum
  displacement, zero velocity). $\psi$ is nearly real — the imaginary part is
  negligible and there's no spatial oscillation. The phase is flat.
- At $t = T/4$ and $t = 3T/4$, the packet is at the center with maximum speed.
  $\psi$ oscillates rapidly in space — Re and Im form a wave pattern. The wave
  moves to the left at $T/4$ and to the right at $3T/4$, matching the direction
  of motion.

This spatial oscillation *is* the momentum. Faster oscillation = higher momentum.

## Probability Current Density

We've seen that motion is encoded in the phase. There's a precise quantity that
captures this: the **probability current density** $j(x,t)$. It measures the
rate at which probability flows past a point, just like electrical current
measures charge flow:

$$
j(x,t) = \frac{\hbar}{m} \operatorname{Im}\left(\psi^* \frac{\partial\psi}{\partial x}\right)
$$

This definition comes directly from the continuity equation for probability:

$$
\frac{\partial |\psi|^2}{\partial t} + \frac{\partial j}{\partial x} = 0
$$

which says probability is conserved locally — if $|\psi|^2$ decreases
somewhere, probability must be flowing away through $j$.

On our grid, we compute $j$ using the same derivative matrix $D$ from before:
$j_i = (\hbar / m) \operatorname{Im}(\psi_i^* (D\psi)_i)$.

Positive $j$ means probability flows to the right; negative means to the left.
"""

#%%
# Compute current density at the same four snapshots
# Pre-compute j at all snapshots to find global max for consistent y-limits
j_snapshots = []
for idx in snapshot_indices:
    psi_t = solution.y[:, idx]
    dpsi_dx = D @ psi_t
    j_snapshots.append((hbar / m) * np.imag(np.conj(psi_t) * dpsi_dx))
j_ylim = max(np.max(np.abs(j)) for j in j_snapshots) * 1.1

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

for ax, idx, label, j_current in zip(axes, snapshot_indices, snapshot_labels, j_snapshots):
    psi_t = solution.y[:, idx]
    pdf_t = np.abs(psi_t)**2

    ax.fill_between(x, pdf_t / np.max(pdf_t) * j_ylim,
                     alpha=0.1, color='gray', label=r'$|\psi|^2$ (scaled)')
    ax.plot(x, j_current, 'darkorchid', linewidth=1.5, label='j(x)')
    ax.set_ylabel('j(x)')
    ax.set_title(label, fontsize=10)
    ax.set_xlim(-L, L)
    ax.set_ylim(-j_ylim, j_ylim)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.grid(True, alpha=0.15)

axes[0].legend(loc='upper left', fontsize=8, ncol=2)
axes[-1].set_xlabel('x')
fig.suptitle('Probability current density at four snapshots', fontsize=12, y=1.01)
plt.tight_layout()
plt.show()

#%%
"""
The current density confirms the picture: $j$ is zero at the turning
points (no flow) and maximum at the center crossing (maximum speed). The sign
flips between $t = T/4$ (moving left) and $t = 3T/4$ (moving right).

Let's watch $j(x,t)$ evolve continuously:
"""

#%%
# Animated current density
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Precompute j(x) for all snapshots
j_all = np.zeros((Nx, Nt))
for i in range(Nt):
    psi_t = solution.y[:, i]
    dpsi = D @ psi_t
    j_all[:, i] = (hbar / m) * np.imag(np.conj(psi_t) * dpsi)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

# Top: probability density
line_pdf, = ax1.plot(x, pdf[:, 0], 'steelblue', linewidth=1.5)
V_scale = np.max(pdf) / 20
ax1.plot(x, V * V_scale, 'k-', linewidth=0.5, alpha=0.3)
ax1.set_xlim(-L, L)
ax1.set_ylim(0, np.max(pdf) * 1.1)
ax1.set_ylabel(r'$|\psi|^2$')
ax1.grid(True, alpha=0.2)
title1 = ax1.set_title(f't = 0.00  (period 0.00)')

# Bottom: current density
j_max = np.max(np.abs(j_all)) * 1.1
line_j, = ax2.plot(x, j_all[:, 0], 'darkorchid', linewidth=1.5)
ax2.set_xlim(-L, L)
ax2.set_ylim(-j_max, j_max)
ax2.set_xlabel('x')
ax2.set_ylabel('j(x)')
ax2.axhline(0, color='k', linewidth=0.3)
ax2.grid(True, alpha=0.2)

def update_j(i):
    line_pdf.set_ydata(pdf[:, i])
    line_j.set_ydata(j_all[:, i])
    title1.set_text(f't = {t_eval[i]:.2f}  (period {t_eval[i]/T_osc:.2f})')
    return line_pdf, line_j, title1

anim_j = FuncAnimation(fig, update_j, frames=Nt, interval=33, blit=True)
plt.close()

HTML(anim_j.to_jshtml())

#%%
"""
## Phase Space and the Husimi Q Function

So far we've looked at the state from one angle at a time: $|\psi(x)|^2$
shows us where the particle might be, and $j(x)$ shows how probability flows.
But a quantum state encodes *both* position and momentum simultaneously. Is
there a way to visualize the full picture?

In classical mechanics, you'd plot the state as a point in **phase space** — a
2D plane with position on one axis and momentum on the other. A harmonic
oscillator traces out a circle (or ellipse) in this space.

For a quantum state, we can't pinpoint both $x$ and $p$ exactly (Heisenberg's
uncertainty principle forbids it). But we can ask a softer question: "If I had
a detector shaped like a minimum-uncertainty wave packet centered at position
$x_0$ with momentum $p_0$, how much of $\psi$ would it pick up?" The answer is
the **Husimi Q function**:

$$
Q(x_0, p_0) = \frac{1}{\pi\hbar} |\langle \alpha_{x_0, p_0} | \psi \rangle|^2
$$

where $|\alpha_{x_0, p_0}\rangle$ is a Gaussian centered at $(x_0, p_0)$:

$$
\alpha(x) = \left(\frac{1}{2\pi\sigma_0^2}\right)^{1/4}
\exp\left(-\frac{(x-x_0)^2}{4\sigma_0^2}\right)
\exp\left(\frac{i p_0 x}{\hbar}\right)
$$

Think of $Q$ as a **blurred photograph** of the quantum state in phase space.
The blur has a fixed size (set by the uncertainty principle), but the overall
shape tells you where the state lives in the position-momentum plane.

For our displaced coherent state, $Q$ is a single Gaussian blob — and as time
passes, this blob traces out a circle in phase space, just like the classical
trajectory.
"""

#%%
# Phase space grid for the Q function
N_xq, N_pq = 50, 50
p_max_q = m * omega * abs(x0_displacement) * 1.8   # covers the classical orbit
xq = np.linspace(-abs(x0_displacement) * 1.8, abs(x0_displacement) * 1.8, N_xq)
pq = np.linspace(-p_max_q, p_max_q, N_pq)
XQ, PQ = np.meshgrid(xq, pq, indexing='xy')  # XQ shape (N_pq, N_xq)

# Precompute overlap building blocks (independent of time)
prefactor = (1 / (2 * np.pi * sigma0**2))**0.25
# Gaussian envelope: g[j, i] = prefactor * exp(-(x_i - xq_j)^2 / (4*sigma0^2))
g_weights = prefactor * np.exp(
    -(x[np.newaxis, :] - xq[:, np.newaxis])**2 / (4 * sigma0**2)
)  # shape (N_xq, Nx)

# Momentum phase factors: phase[k, i] = exp(-i * pq_k * x_i / hbar)
phase_factors = np.exp(-1j * pq[:, np.newaxis] * x[np.newaxis, :] / hbar)  # (N_pq, Nx)

# Compute Q at selected frames
n_q_frames = 80
q_frame_idx = np.linspace(0, Nt - 1, n_q_frames, dtype=int)
Q_frames = np.zeros((n_q_frames, N_pq, N_xq))

print(f"Computing Q function on {N_xq}x{N_pq} phase space grid, {n_q_frames} frames...")
for fi, ti in enumerate(q_frame_idx):
    psi_t = solution.y[:, ti]
    g_psi = g_weights * psi_t[np.newaxis, :]     # (N_xq, Nx)
    overlap = (g_psi * dx) @ phase_factors.conj().T  # (N_xq, N_pq)
    Q_frames[fi] = np.abs(overlap.T)**2 / (np.pi * hbar)  # (N_pq, N_xq)

print(f"Done. Q max = {Q_frames.max():.4f}")

#%%
"""
Let's first look at a static 3D view of the initial Q function — you should see
a single Gaussian peak sitting at $(x_0, 0)$ in phase space:
"""

#%%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XQ, PQ, Q_frames[0], cmap='viridis', alpha=0.9,
                rstride=1, cstride=1, linewidth=0, antialiased=True)
ax.set_xlabel('x (position)')
ax.set_ylabel('p (momentum)')
ax.set_zlabel('Q(x, p)')
ax.set_title('Husimi Q function at t = 0')
ax.view_init(elev=30, azim=-60)
plt.tight_layout()
plt.show()

#%%
"""
Now let's animate this. The Q blob should orbit in a circle — the quantum
version of the classical phase space trajectory. The animation renders each
frame as a 3D surface, so it takes a moment to generate.
"""

#%%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

Q_max = Q_frames.max() * 1.05
t_q = t_eval[q_frame_idx]

def update_Q(fi):
    ax.clear()
    ax.plot_surface(XQ, PQ, Q_frames[fi], cmap='viridis', alpha=0.9,
                    rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_zlabel('Q')
    ax.set_zlim(0, Q_max)
    ax.set_title(f'Q function  t = {t_q[fi]:.2f}  (period {t_q[fi]/T_osc:.2f})')
    ax.view_init(elev=30, azim=-60)

print(f"Generating 3D animation ({n_q_frames} frames)... this takes a moment.")
anim_Q = FuncAnimation(fig, update_Q, frames=n_q_frames, interval=50, blit=False)
plt.close()

HTML(anim_Q.to_jshtml())
# %%
