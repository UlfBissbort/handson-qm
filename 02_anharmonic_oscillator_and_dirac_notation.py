#%%
"""
# Beyond the Harmonic Oscillator: Anharmonicity and Dirac Notation

In the first notebook we simulated a quantum particle in a harmonic potential
and discovered something remarkable: a displaced Gaussian wave packet oscillated
back and forth *forever without changing its shape*. It stayed a perfect Gaussian
at every instant — no spreading, no distortion, no surprises. If you only ever
saw the probability density $|\psi(x,t)|^2$, you might mistake it for a
classical ball on a spring.

That perfect behavior is not generic. It's a very special property of the
harmonic potential $V(x) = \frac{1}{2} m \omega^2 x^2$. The reason is deep:
the harmonic oscillator's energy levels are *exactly evenly spaced*
($E_n = \hbar\omega(n + \frac{1}{2})$), so all the frequency components in
the wave packet stay perfectly in phase as time passes. Mess with that equal
spacing — even slightly — and the wave packet will eventually distort, spread,
and develop structure that looks nothing like a Gaussian.

In this notebook we'll do exactly that: add a small quartic ($x^4$) term to
the potential and watch what happens. But first, we need a more powerful
language for talking about quantum states. The $\psi(x)$ notation from
Notebook 1 is tied to a specific representation — the position basis. We need
something more abstract.
"""

#%%
"""
## Dirac Notation: States Without Coordinates

### The Problem with $\psi(x)$

In Notebook 1, a quantum state was a function $\psi(x)$ — a complex number at
every point in space. But this is like describing a vector by listing its
components: $\\vec{v} = (v_x, v_y, v_z)$. The components depend on your
coordinate system, but the vector itself doesn't. If someone hands you a
velocity vector, its *physical meaning* (speed and direction of an object)
is the same whether you write it in Cartesian or polar coordinates. The
components change; the physics doesn't.

Quantum states work the same way. Paul Dirac invented a notation that captures
this. A quantum state is written as a **ket**:

$$
|\\psi\\rangle
$$

This is the abstract state — no coordinates, no basis, just "the state of the
system." It lives in a vector space called **Hilbert space**, which is
(very roughly) an infinite-dimensional version of the vector spaces you know
from linear algebra.

### Getting Numbers Out: The Inner Product

To get actual numbers from $|\\psi\\rangle$, you project it onto a **basis**.
If you choose the **position basis**, where $|x\\rangle$ represents "the particle
is exactly at position $x$", then:

$$
\\psi(x) = \\langle x | \\psi \\rangle
$$

The left side is the wave function we've been using all along. The right side
says: "take the inner product of the position basis state $|x\\rangle$ with the
abstract state $|\\psi\\rangle$." The **bra** $\\langle x|$ is the dual
(conjugate transpose) of the ket $|x\\rangle$.

The notation is deliberately suggestive: a **bra**cket $\\langle x | \\psi \\rangle$
is an inner product. A ket $|\\psi\\rangle$ is like a column vector. A bra
$\\langle\\phi|$ is like a row vector (the conjugate transpose). Their product is
a number.

### Basis Independence

This might seem like unnecessary formalism, but it buys us something real:
**the same state can be expressed in different bases**. You could equally well
project onto momentum eigenstates:

$$
\\tilde\\psi(p) = \\langle p | \\psi \\rangle
$$

Same state $|\\psi\\rangle$, different representation. The physics doesn't care
which basis you use — just like a force vector doesn't care whether you describe
it in Cartesian or spherical coordinates.

The completeness relation ties everything together:

$$
\\int |x\\rangle \\langle x| dx = \\hat{I}
$$

This says: if you sum up all position projectors $|x\\rangle\\langle x|$,
you get the identity operator. It's the quantum version of "every vector can
be expanded in a complete basis." Insert this resolution of the identity into
any bra-ket expression and you recover the position representation.
"""

#%%
"""
## From Continuous to Discrete: What Our Grid Really Means

In Notebook 1, we discretized space onto a grid $x_0, x_1, \\ldots, x_{N-1}$
and stored $\\psi$ as a vector of $N$ complex numbers. In Dirac notation, what
we actually computed was:

$$
\\psi_i = \\langle x_i | \\psi \\rangle
$$

Our NumPy vector `psi` is the collection of these overlaps — the state
$|\\psi\\rangle$ expressed in the position basis $\\{|x_i\\rangle\\}$.

When we compute the norm as `np.sum(np.abs(psi)**2) * dx`, we're evaluating:

$$
\\langle \\psi | \\psi \\rangle = \\sum_i |\\langle x_i|\\psi\\rangle|^2 \\Delta x
\\approx \\int |\\langle x|\\psi\\rangle|^2 dx = \\int |\\psi(x)|^2 dx
$$

When we compute $\\langle x \\rangle =$ `np.sum(x * np.abs(psi)**2) * dx`,
we're evaluating $\\langle \\psi | \\hat{x} | \\psi \\rangle$ — the bra-ket
sandwich of the position operator.

### Matrix Representations Are Always Relative to a Basis

**Every matrix we built in Notebook 1 was secretly a representation of an
abstract operator in the position basis.** The tridiagonal kinetic energy
matrix $T$ represents $\\langle x_i | \\hat{T} | x_j \\rangle$. The diagonal
potential matrix represents $\\langle x_i | \\hat{V} | x_j \\rangle = V(x_i)
\\delta_{ij}$. The Hamiltonian matrix $H$ represents
$\\langle x_i | \\hat{H} | x_j \\rangle$.

This is exactly like how a rotation matrix $R$ is a representation of an
abstract rotation in a particular coordinate system. Change the basis
(coordinate system) and you get a different matrix, but the rotation itself
is the same.

The power of Dirac notation is this separation: $|\\psi\\rangle$ is the state,
$\\hat{H}$ is the operator, and the matrix elements
$\\langle x_i | \\hat{H} | x_j \\rangle$ are what we actually put in the
computer. The physics lives in the abstract objects; the numerics live in a
particular basis.

### Energy Eigenstates as a Basis

There's another natural basis: the **energy eigenstates** $|n\\rangle$ satisfying
$\\hat{H}|n\\rangle = E_n|n\\rangle$. Any state can be expanded:

$$
|\\psi\\rangle = \\sum_n c_n |n\\rangle \\quad \\text{where} \\quad c_n = \\langle n | \\psi \\rangle
$$

Time evolution is trivially simple in this basis:

$$
|\\psi(t)\\rangle = \\sum_n c_n e^{-iE_n t/\\hbar} |n\\rangle
$$

Each energy eigenstate just acquires a phase. The wave function in position
space is then $\\psi(x,t) = \\sum_n c_n e^{-iE_n t/\\hbar} \\langle x|n\\rangle$.
Whether the wave packet stays together or falls apart depends entirely on how
the phases $e^{-iE_n t/\\hbar}$ relate to each other — which depends on the
energy spectrum $\\{E_n\\}$.
"""

#%%
"""
## The Anharmonic Potential

Now let's break the harmonic oscillator's perfect symmetry. We add a
quartic term:

$$
V(x) = \\frac{1}{2} m \\omega^2 x^2 + \\lambda x^4
$$

The parameter $\\lambda$ controls the strength of the anharmonicity. When
$\\lambda = 0$ we recover the harmonic oscillator. For positive $\\lambda$, the
potential rises faster than a parabola at large $|x|$, which:
- pushes higher energy levels *further apart* than $\\hbar\\omega$
- breaks the equal spacing that kept the harmonic wave packet coherent
- causes the wave packet to gradually lose its shape

We choose $\\lambda = 0.01$ — small enough that the potential looks nearly
parabolic near the center, but large enough to produce visible effects
over several oscillation periods:
"""

#%%
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# Physical parameters (natural units: hbar = m = omega = 1)
hbar = 1.0
m = 1.0
omega = 1.0

# Anharmonicity
lam = 0.01

# Initial wave packet: a coherent state displaced from center
x0_displacement = 3.0

# Coherent state width: the ground state of the HO has rms width
# sigma = sqrt(hbar / (2*m*omega)). A displaced Gaussian with this exact
# width is a coherent state — it oscillates without spreading in the
# harmonic potential.
sigma = np.sqrt(hbar / (2 * m * omega))

print(f"Coherent state width sigma = {sigma:.4f}")
print(f"Anharmonic coupling lambda = {lam}")
print(f"Initial displacement x0 = {x0_displacement}")

# How significant is the quartic term at the displacement?
V_harm_at_x0 = 0.5 * m * omega**2 * x0_displacement**2
V_quart_at_x0 = lam * x0_displacement**4
print(f"\nAt x = {x0_displacement}:")
print(f"  Harmonic V  = {V_harm_at_x0:.2f}")
print(f"  Quartic V   = {V_quart_at_x0:.2f}")
print(f"  Ratio       = {V_quart_at_x0/V_harm_at_x0:.1%}")

#%%
"""
## Comparing Potentials

At the initial displacement $x_0 = 3$, the quartic correction is about 18%
of the harmonic term. That's a meaningful perturbation — not a tiny correction,
but not so large that the potential looks qualitatively different from a
parabola near the center.
"""

#%%
# Spatial grid
Nx = 512
L = abs(x0_displacement) + 8 * sigma
x = np.linspace(-L, L, Nx)
dx = x[1] - x[0]

# Potentials
V_harm = 0.5 * m * omega**2 * x**2
V_anharm = V_harm + lam * x**4

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, V_harm, 'k--', linewidth=1, alpha=0.6, label='Harmonic')
ax.plot(x, V_anharm, 'steelblue', linewidth=1.5,
        label=f'Anharmonic ($\\lambda = {lam}$)')
ax.axvline(x0_displacement, color='coral', linestyle=':', alpha=0.5,
           label=f'$x_0 = {x0_displacement}$')
ax.axvline(-x0_displacement, color='coral', linestyle=':', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('V(x)')
ax.set_title('Harmonic vs. anharmonic potential')
ax.set_ylim(0, 20)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Grid: {Nx} points, dx = {dx:.4f}")

#%%
"""
## The Energy Spectrum: Why Equal Spacing Matters

Before we run the time evolution, let's look at the energy eigenvalues of
both potentials. This is where the Dirac notation pays off: a state
$|\\psi\\rangle = \\sum_n c_n |n\\rangle$ evolves as
$|\\psi(t)\\rangle = \\sum_n c_n e^{-iE_n t/\\hbar} |n\\rangle$. If the
energy gaps $E_{n+1} - E_n$ are all equal (as in the harmonic oscillator),
then all the phases realign after one period $T = 2\\pi/\\omega$ and the
wave packet returns to its exact initial shape. If the gaps vary, the phases
drift apart and the wave packet deforms.

We can compute eigenvalues numerically using `scipy.sparse.linalg.eigsh`:
"""

#%%
# Build Hamiltonians
main_diag = -2.0 / dx**2 * np.ones(Nx)
off_diag  =  1.0 / dx**2 * np.ones(Nx - 1)

T_kinetic = -(hbar**2 / (2 * m)) * diags(
    [off_diag, main_diag, off_diag],
    [-1, 0, 1], shape=(Nx, Nx), dtype=float
)

H_harm = T_kinetic + diags(V_harm, 0, shape=(Nx, Nx), dtype=float)
H_anharm = T_kinetic + diags(V_anharm, 0, shape=(Nx, Nx), dtype=float)

# Compute first 15 eigenvalues
n_eigs = 15
E_harm, _ = eigsh(H_harm, k=n_eigs, which='SM')
E_anharm, _ = eigsh(H_anharm, k=n_eigs, which='SM')
E_harm = np.sort(E_harm)
E_anharm = np.sort(E_anharm)

# Energy level gaps
gaps_harm = np.diff(E_harm)
gaps_anharm = np.diff(E_anharm)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: energy levels
ax = axes[0]
for i in range(n_eigs):
    ax.plot([0, 1], [E_harm[i], E_harm[i]], 'k-', linewidth=1.5, alpha=0.6)
    ax.plot([2, 3], [E_anharm[i], E_anharm[i]], 'steelblue', linewidth=1.5)
ax.set_xticks([0.5, 2.5])
ax.set_xticklabels(['Harmonic', 'Anharmonic'])
ax.set_ylabel('Energy')
ax.set_title('Energy levels')
ax.grid(True, alpha=0.2, axis='y')

# Right: gaps between consecutive levels
ax = axes[1]
ns = np.arange(len(gaps_harm))
ax.plot(ns, gaps_harm, 'ko-', markersize=5, alpha=0.6, label='Harmonic')
ax.plot(ns, gaps_anharm, 'o-', color='steelblue', markersize=5, label='Anharmonic')
ax.axhline(1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Level index n')
ax.set_ylabel('$E_{n+1} - E_n$')
ax.set_title('Energy level spacing')
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print("Harmonic gaps (should all be ~1.0):")
print(f"  range: [{gaps_harm.min():.4f}, {gaps_harm.max():.4f}]")
print(f"\nAnharmonic gaps (should increase with n):")
print(f"  range: [{gaps_anharm.min():.4f}, {gaps_anharm.max():.4f}]")
print(f"  spread: {gaps_anharm.max() - gaps_anharm.min():.4f}")

#%%
"""
## Time Evolution: Harmonic vs. Anharmonic

Now the main event. We evolve the same initial coherent state under both
Hamiltonians for 10 oscillation periods. The harmonic case is our control:
the wave packet should remain a perfect Gaussian throughout. The anharmonic
case is where the interesting physics happens.
"""

#%%
# Convert Hamiltonians to complex for the ODE solver
H = T_kinetic.astype(complex) + diags(V_anharm + 0j, 0, shape=(Nx, Nx))
H_h = T_kinetic.astype(complex) + diags(V_harm + 0j, 0, shape=(Nx, Nx))

# Initial coherent state
psi_0 = ((1 / (2 * np.pi * sigma**2))**0.25
         * np.exp(-(x - x0_displacement)**2 / (4 * sigma**2))
         + 0j)
psi_0 /= np.sqrt(np.sum(np.abs(psi_0)**2) * dx)

print(f"Initial norm: {np.sum(np.abs(psi_0)**2) * dx:.10f}")

# Time parameters
T_osc = 2 * np.pi / omega
T_total = 10 * T_osc
Nt = 600
t_eval = np.linspace(0, T_total, Nt)

# Stability limit
max_eigenvalue_estimate = 2 * hbar**2 / (m * dx**2) + np.max(V_anharm)
max_dt = 2.0 / max_eigenvalue_estimate

def make_rhs(hamiltonian):
    def rhs(t, psi):
        return (-1j / hbar) * (hamiltonian @ psi)
    return rhs

print(f"Propagating for {T_total/T_osc:.0f} periods...")

print("  Anharmonic...", end=" ", flush=True)
sol_a = solve_ivp(
    make_rhs(H), [0, T_total], psi_0,
    t_eval=t_eval, method='RK45',
    max_step=max_dt, rtol=1e-8, atol=1e-10,
)
print(f"done ({sol_a.nfev} evaluations)")

print("  Harmonic...", end=" ", flush=True)
sol_h = solve_ivp(
    make_rhs(H_h), [0, T_total], psi_0,
    t_eval=t_eval, method='RK45',
    max_step=max_dt, rtol=1e-8, atol=1e-10,
)
print(f"done ({sol_h.nfev} evaluations)")

# Norm check
norm_a = np.sum(np.abs(sol_a.y[:, -1])**2) * dx
norm_h = np.sum(np.abs(sol_h.y[:, -1])**2) * dx
print(f"\nFinal norms — harmonic: {norm_h:.10f}, anharmonic: {norm_a:.10f}")

#%%
"""
## Comparing Position and Width

Two diagnostics tell us the essential story:
- $\\langle x \\rangle(t)$: does the center still oscillate like a cosine?
- $\\sigma_x(t)$: does the width stay constant?

For the harmonic case, Ehrenfest's theorem guarantees
$\\langle x \\rangle = x_0 \\cos(\\omega t)$ exactly, and the coherent state
width is constant. For the anharmonic case, both should deviate.
"""

#%%
# Compute <x>(t) and sigma_x(t) for both
x_a = np.array([np.sum(x * np.abs(sol_a.y[:, i])**2) * dx for i in range(Nt)])
x_h = np.array([np.sum(x * np.abs(sol_h.y[:, i])**2) * dx for i in range(Nt)])
x2_a = np.array([np.sum(x**2 * np.abs(sol_a.y[:, i])**2) * dx for i in range(Nt)])
x2_h = np.array([np.sum(x**2 * np.abs(sol_h.y[:, i])**2) * dx for i in range(Nt)])
sig_a = np.sqrt(x2_a - x_a**2)
sig_h = np.sqrt(x2_h - x_h**2)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# Position
ax = axes[0]
ax.plot(t_eval / T_osc, x_h, 'k-', linewidth=1, alpha=0.4, label='Harmonic')
ax.plot(t_eval / T_osc, x_a, 'steelblue', linewidth=1.5, label='Anharmonic')
ax.set_ylabel(r'$\langle x \rangle$')
ax.set_title('Position expectation value')
ax.legend()
ax.grid(True, alpha=0.2)
ax.axhline(0, color='k', linewidth=0.3)

# Width
ax = axes[1]
ax.plot(t_eval / T_osc, sig_h, 'k-', linewidth=1, alpha=0.4, label='Harmonic')
ax.plot(t_eval / T_osc, sig_a, 'steelblue', linewidth=1.5, label='Anharmonic')
ax.axhline(sigma, color='k', linestyle='--', linewidth=0.8, alpha=0.3,
           label=f'Initial $\\sigma = {sigma:.3f}$')
ax.set_ylabel(r'$\sigma_x$')
ax.set_title('Wave packet width')
ax.legend()
ax.grid(True, alpha=0.2)

# Difference in position
ax = axes[2]
ax.plot(t_eval / T_osc, x_a - x_h, 'coral', linewidth=1.5)
ax.set_xlabel('Time (oscillation periods)')
ax.set_ylabel(r'$\Delta\langle x \rangle$')
ax.set_title('Position difference (anharmonic - harmonic)')
ax.grid(True, alpha=0.2)
ax.axhline(0, color='k', linewidth=0.3)

plt.tight_layout()
plt.show()

print(f"Harmonic width range:   [{sig_h.min():.4f}, {sig_h.max():.4f}] (expected constant ~{sigma:.4f})")
print(f"Anharmonic width range: [{sig_a.min():.4f}, {sig_a.max():.4f}]")
print(f"Max |position diff|:    {np.max(np.abs(x_a - x_h)):.3f}")

#%%
"""
## Watching the Wave Packet Break Apart

Numbers confirm the effect; now let's *see* it. Below are snapshots of
$|\\psi(x,t)|^2$ at four times during the evolution. The gray Gaussian is the
harmonic case (which always looks the same); the blue curve is the anharmonic
case.

In the Dirac picture, what's happening is that the coefficients $c_n$ in
$|\\psi(t)\\rangle = \\sum_n c_n e^{-iE_n t/\\hbar}|n\\rangle$ are acquiring
phases that no longer cancel. The interference between energy eigenstates
creates the multi-peaked structure you see.
"""

#%%
snapshot_periods = [0, 2, 5, 10]
snapshot_indices = [int(p / 10 * (Nt - 1)) for p in snapshot_periods]

fig, axes = plt.subplots(len(snapshot_periods), 1, figsize=(10, 10), sharex=True)

for ax, idx, period in zip(axes, snapshot_indices, snapshot_periods):
    pdf_h = np.abs(sol_h.y[:, idx])**2
    pdf_a = np.abs(sol_a.y[:, idx])**2

    ax.fill_between(x, pdf_h, alpha=0.2, color='gray', label='Harmonic')
    ax.plot(x, pdf_h, 'k-', linewidth=0.8, alpha=0.5)
    ax.fill_between(x, pdf_a, alpha=0.4, color='steelblue', label='Anharmonic')
    ax.plot(x, pdf_a, 'steelblue', linewidth=1.5)
    ax.set_ylabel(r'$|\psi|^2$')
    ax.set_title(f't = {period} periods', fontsize=10)
    ax.set_xlim(-L, L)
    ax.grid(True, alpha=0.15)

axes[0].legend(fontsize=9)
axes[-1].set_xlabel('x')
fig.suptitle('Probability density snapshots: harmonic vs. anharmonic',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.show()

#%%
"""
## The Autocorrelation Function: Tracking Coherence

The snapshots show the wave packet fragmenting, but is this a one-way process?
Will the wave packet *ever* reassemble? In the harmonic oscillator, it
reassembles perfectly every period. In the anharmonic case, the answer depends
on the energy spectrum.

A powerful tool for this question is the **autocorrelation function**:

$$
C(t) = |\\langle \\psi(0) | \\psi(t) \\rangle|^2
$$

This measures the overlap between the current state and the initial state. When
$C(t) = 1$, the state has returned to its starting point (up to a global phase).
When $C(t) \\approx 0$, the state has spread out so much that it has essentially
no overlap with the original localized wave packet.

In the energy basis, the autocorrelation has a clean form:

$$
\\langle\\psi(0)|\\psi(t)\\rangle = \\sum_n |c_n|^2 e^{-iE_n t/\\hbar}
$$

This is a sum of oscillating terms. For the harmonic oscillator, all
frequencies are multiples of $\\omega$, so $C(t)$ is exactly periodic with
period $T = 2\\pi/\\omega$. For the anharmonic oscillator, the frequencies
are incommensurate — but they may *approximately* realign at certain times,
producing **quantum revivals**.
"""

#%%
# Autocorrelation function: |<psi(0)|psi(t)>|^2
autocorr_a = np.array([
    np.abs(np.sum(np.conj(psi_0) * sol_a.y[:, i]) * dx)**2
    for i in range(Nt)
])
autocorr_h = np.array([
    np.abs(np.sum(np.conj(psi_0) * sol_h.y[:, i]) * dx)**2
    for i in range(Nt)
])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax = axes[0]
ax.plot(t_eval / T_osc, autocorr_h, 'k-', linewidth=1, alpha=0.6)
ax.set_ylabel(r'$|C(t)|^2$')
ax.set_title('Autocorrelation: harmonic (returns to 1 every period)')
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.plot(t_eval / T_osc, autocorr_a, 'steelblue', linewidth=1.5)
ax.set_xlabel('Time (oscillation periods)')
ax.set_ylabel(r'$|C(t)|^2$')
ax.set_title('Autocorrelation: anharmonic (decays as wave packet spreads)')
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print(f"Harmonic autocorrelation at T_osc: {autocorr_h[Nt//10]:.4f} (expected ~1.0)")
print(f"Anharmonic autocorrelation at T_osc: {autocorr_a[Nt//10]:.4f}")
print(f"Anharmonic min autocorrelation: {autocorr_a.min():.4f}")

#%%
"""
## Hunting for Quantum Revivals

The autocorrelation is decaying, but will it come back? The theory of quantum
revivals tells us that for a system with energy levels $E_n$ that can be
expanded to second order around the mean quantum number $\\bar n$:

$$
E_n \\approx E_{\\bar n} + E'(\\bar n)(n - \\bar n) + \\tfrac{1}{2} E''(\\bar n)(n - \\bar n)^2
$$

the wave packet reassembles at the **revival time**:

$$
T_{\\text{rev}} = \\frac{2\\pi}{|E''(\\bar n)| / \\hbar}
$$

For our coherent state at $x_0 = 3$ with $\\omega = 1$, the mean quantum number
is $\\bar n = x_0^2 m\\omega / (2\\hbar) \\approx 4.5$. Let's estimate the revival
time from the numerically computed eigenvalues:
"""

#%%
# Estimate revival time from the energy spectrum
n_mean = 4  # closest integer to 4.5
d2E = E_anharm[n_mean + 1] - 2 * E_anharm[n_mean] + E_anharm[n_mean - 1]
T_rev = 2 * np.pi * hbar / abs(d2E)

print(f"Mean quantum number: ~{x0_displacement**2 * m * omega / (2*hbar):.1f}")
print(f"E''(n_mean) from eigenvalues: {d2E:.6f}")
print(f"Estimated revival time: {T_rev:.1f} time units = {T_rev/T_osc:.1f} periods")
print(f"\nThat's a long time! Let's simulate for {int(T_rev/T_osc * 1.2)} periods to see it...")

#%%
"""
## Long-Time Evolution: Seeing the Revival

The revival time is about 45 periods. Let's run a longer simulation to
see whether the wave packet actually reconstitutes itself. We'll track the
autocorrelation function — a spike in $|C(t)|^2$ signals a revival.
"""

#%%
# Longer simulation to capture revival
T_long = T_rev * 1.2
Nt_long = 1200
t_long = np.linspace(0, T_long, Nt_long)

print(f"Long simulation: {T_long/T_osc:.0f} periods, {Nt_long} snapshots...")
sol_long = solve_ivp(
    make_rhs(H), [0, T_long], psi_0,
    t_eval=t_long, method='RK45',
    max_step=max_dt, rtol=1e-8, atol=1e-10,
)
print(f"Done. nfev = {sol_long.nfev}")
print(f"Final norm: {np.sum(np.abs(sol_long.y[:, -1])**2) * dx:.10f}")

# Autocorrelation over the long run
autocorr_long = np.array([
    np.abs(np.sum(np.conj(psi_0) * sol_long.y[:, i]) * dx)**2
    for i in range(Nt_long)
])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_long / T_osc, autocorr_long, 'steelblue', linewidth=1)
ax.axvline(T_rev / T_osc, color='coral', linestyle='--', linewidth=1,
           alpha=0.7, label=f'Predicted revival ({T_rev/T_osc:.1f} periods)')
ax.set_xlabel('Time (oscillation periods)')
ax.set_ylabel(r'$|C(t)|^2$')
ax.set_title('Autocorrelation over long time — hunting for revivals')
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# Find peaks in autocorrelation
from scipy.signal import find_peaks
peaks, props = find_peaks(autocorr_long, height=0.3, distance=10)
if len(peaks) > 0:
    print("\nAutocorrelation peaks above 0.3:")
    for p in peaks[:10]:
        print(f"  t = {t_long[p]/T_osc:.2f} periods, |C|^2 = {autocorr_long[p]:.4f}")
else:
    print("\nNo strong revivals found — the wave packet has fully dispersed.")
    print("Highest autocorrelation after initial decay:", autocorr_long[100:].max())

#%%
"""
## Snapshots at Special Times

Let's look at the probability density at several moments during the long
evolution: the initial state, after the wave packet has spread, near the
predicted revival time, and near any fractional revivals (at $T_{\\text{rev}}/2$,
$T_{\\text{rev}}/3$, etc.) where the wave packet might split into multiple
copies of itself:
"""

#%%
# Snapshots at interesting times
times_of_interest = {
    't = 0 (initial)': 0,
    't = 5 periods (spreading)': 5 * T_osc,
    f't = {T_rev/(2*T_osc):.0f} periods (half revival)': T_rev / 2,
    f't = {T_rev/T_osc:.0f} periods (full revival)': T_rev,
}

fig, axes = plt.subplots(len(times_of_interest), 1, figsize=(10, 10), sharex=True)

for ax, (label, t_target) in zip(axes, times_of_interest.items()):
    idx = np.argmin(np.abs(t_long - t_target))
    pdf = np.abs(sol_long.y[:, idx])**2

    ax.fill_between(x, pdf, alpha=0.4, color='steelblue')
    ax.plot(x, pdf, 'steelblue', linewidth=1.5)

    # Overlay initial state for reference
    pdf_0 = np.abs(psi_0)**2
    ax.plot(x, pdf_0, 'k--', linewidth=0.8, alpha=0.3, label='Initial')

    autocorr_val = np.abs(np.sum(np.conj(psi_0) * sol_long.y[:, idx]) * dx)**2
    ax.set_ylabel(r'$|\psi|^2$')
    ax.set_title(f'{label}  ($|C|^2 = {autocorr_val:.3f}$)', fontsize=10)
    ax.set_xlim(-L, L)
    ax.grid(True, alpha=0.15)

axes[0].legend(fontsize=9, loc='upper left')
axes[-1].set_xlabel('x')
fig.suptitle('Wave packet at special times during long evolution', fontsize=12, y=1.01)
plt.tight_layout()
plt.show()

#%%
"""
## The Full Picture: Four Diagnostics

Let's put position, width, energy, and autocorrelation together in one
summary plot over the full long evolution:
"""

#%%
# Compute all diagnostics for the long run
x_long = np.array([np.sum(x * np.abs(sol_long.y[:, i])**2) * dx for i in range(Nt_long)])
x2_long = np.array([np.sum(x**2 * np.abs(sol_long.y[:, i])**2) * dx for i in range(Nt_long)])
sig_long = np.sqrt(x2_long - x_long**2)

E_long = np.array([
    np.real(np.sum(np.conj(sol_long.y[:, i]) * (H @ sol_long.y[:, i])) * dx)
    for i in range(Nt_long)
])

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Position
ax = axes[0]
ax.plot(t_long / T_osc, x_long, 'steelblue', linewidth=0.8)
ax.axvline(T_rev / T_osc, color='coral', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel(r'$\langle x \rangle$')
ax.set_title('Position')
ax.grid(True, alpha=0.2)
ax.axhline(0, color='k', linewidth=0.3)

# Width
ax = axes[1]
ax.plot(t_long / T_osc, sig_long, 'steelblue', linewidth=0.8)
ax.axhline(sigma, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
ax.axvline(T_rev / T_osc, color='coral', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel(r'$\sigma_x$')
ax.set_title('Wave packet width')
ax.grid(True, alpha=0.2)

# Energy
ax = axes[2]
ax.plot(t_long / T_osc, E_long, 'goldenrod', linewidth=0.8)
ax.axvline(T_rev / T_osc, color='coral', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel(r'$\langle E \rangle$')
ax.set_title('Total energy (should be conserved)')
ax.grid(True, alpha=0.2)

# Autocorrelation
ax = axes[3]
ax.plot(t_long / T_osc, autocorr_long, 'steelblue', linewidth=0.8)
ax.axvline(T_rev / T_osc, color='coral', linestyle='--', linewidth=0.8, alpha=0.5,
           label=f'$T_{{rev}} \\approx {T_rev/T_osc:.0f}$ periods')
ax.set_xlabel('Time (oscillation periods)')
ax.set_ylabel(r'$|C(t)|^2$')
ax.set_title('Autocorrelation')
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(True, alpha=0.2)

fig.suptitle(f'Anharmonic oscillator ($\\lambda = {lam}$) — long-time dynamics',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

print(f"Energy conservation: E(0) = {E_long[0]:.6f}, E(end) = {E_long[-1]:.6f}, "
      f"drift = {abs(E_long[-1]-E_long[0])/E_long[0]:.2e}")

#%%
"""
## What We've Learned

**The harmonic oscillator is special.** Its evenly-spaced energy levels mean
that a coherent state (displaced Gaussian) oscillates forever without
spreading — the phases $e^{-iE_n t/\\hbar}$ all realign perfectly every period.
This is why it behaves so classically.

**Anharmonicity breaks this.** Even a small quartic perturbation makes the
energy levels unequally spaced. The phases drift apart, causing the wave packet
to spread and develop interference fringes. In Dirac language:
$|\\psi(t)\\rangle = \\sum_n c_n e^{-iE_n t/\\hbar}|n\\rangle$ loses coherence
because the $e^{-iE_n t}$ factors no longer periodically realign.

**But quantum mechanics is unitary.** The spreading is reversible — information
about the initial state isn't lost, just scrambled across many energy
eigenstates. The autocorrelation function $|\\langle\\psi(0)|\\psi(t)\\rangle|^2$
tracks this: it drops when the wave packet disperses but can partially recover
at the revival time $T_{\\text{rev}} = 2\\pi\\hbar / |E''(\\bar n)|$.

**Dirac notation clarifies the mechanism.** Without it, you'd see a function
$\\psi(x,t)$ doing complicated things and have no clean way to explain *why*.
With $|\\psi\\rangle = \\sum_n c_n |n\\rangle$, the entire story reduces to
phase accumulation in the energy basis. The position representation
$\\psi(x,t) = \\langle x|\\psi(t)\\rangle$ is just one way to visualize a state
whose essential dynamics are captured by the eigenvalue structure of $\\hat{H}$.
"""
