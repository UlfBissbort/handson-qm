#%%
r"""
# Finding Energy Eigenstates: The Shooting Method

In the previous notebooks we solved the time-dependent Schrodinger equation:

$$
i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle
$$

We built the Hamiltonian as a matrix, handed it to an ODE solver, and watched wave packets slosh back and forth in harmonic and anharmonic potentials. The state $|\psi(t)\rangle$ was always changing — moving, spreading, developing interference fringes.

But what if we asked a different question: **are there states that don't change at all?** A state $|\psi(t)\rangle$ whose probability density $|\langle x|\psi(t)\rangle|^2$ looks the same at every instant — no sloshing, no spreading, nothing. Such a state would be the quantum equivalent of a standing wave: vibrating in place, but with a shape that never moves.

## Stationary States

For $|\langle x|\psi(t)\rangle|^2$ to be constant in time, all the time dependence in $|\psi(t)\rangle$ must be a global phase — a factor $e^{-iEt/\hbar}$ that multiplies the entire state and cancels out when we take the squared modulus. So we're looking for states of the form:

$$
|\psi(t)\rangle = e^{-iEt/\hbar}|\phi\rangle
$$

Substituting this into the Schrodinger equation, the time derivative pulls down a factor of $E$ and the exponential cancels on both sides, leaving:

$$
\hat{H}|\phi\rangle = E|\phi\rangle
$$

This is an **eigenvalue equation**. The special states $|\phi\rangle$ are called **energy eigenstates**, and the allowed values $E$ are the energy **eigenvalues**. Finding them is one of the central problems in quantum mechanics — once you know all the eigenstates and eigenvalues of a system, you essentially know everything about it.

## Turning It Into a Spatial ODE

To compute anything, we project onto position: $\phi(x) = \langle x|\phi\rangle$. The Hamiltonian $\hat{H} = \hat{T} + \hat{V}$ becomes a differential operator, and the eigenvalue equation reads:

$$
-\frac{\hbar^2}{2m}\frac{d^2\phi}{dx^2} + V(x)\phi(x) = E\phi(x)
$$

Rearranging:

$$
\frac{d^2\phi}{dx^2} = \frac{2m}{\hbar^2}\bigl[V(x) - E\bigr]\phi(x)
$$

This is a second-order ODE in *space*. The energy $E$ enters as a parameter. Given any value of $E$, we can pick initial conditions and integrate across $x$ — exactly the way `solve_ivp` integrated the time-dependent equation across $t$ in Notebook 1.

Let's just try it. We'll take the anharmonic potential from Notebook 2 — $V(x) = \frac{1}{2}m\omega^2 x^2 + \lambda x^4$ — pick some energy, and integrate the ODE from left to right to see what $\phi(x)$ looks like.
"""

#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Natural units: hbar = m = omega = 1
hbar = 1.0
m = 1.0
omega = 1.0
lam = 0.01          # quartic coupling (same as Notebook 2)

def V(x):
    return 0.5 * m * omega**2 * x**2 + lam * x**4

# The ODE: phi'' = (2m/hbar^2) * (V(x) - E) * phi
# As a first-order system: y = [phi, phi']
def schrodinger_ode(x, y, E):
    phi, dphi = y
    d2phi = (2 * m / hbar**2) * (V(x) - E) * phi
    return [dphi, d2phi]

# Try an energy between the ground state (~0.51) and first excited state (~1.54)
E_try = 4.2

x_start, x_end = -6.0, 6.0
x_eval = np.linspace(x_start, x_end, 2000)

# Start with a tiny value deep in the forbidden region (left side).
# The exact numbers don't matter — the ODE is linear, so any nonzero
# start will give the same shape up to an overall scale.
sol = solve_ivp(
    lambda x, y: schrodinger_ode(x, y, E_try),
    [x_start, x_end],
    [1e-5, 1e-3],
    t_eval=x_eval,
    rtol=1e-10, atol=1e-12,
)

phi = sol.y[0]

# Normalize so the peak in the classically allowed region equals 1.
# This way the wave-like shape is clearly visible before the divergence.
allowed = np.abs(x_eval) < 2.5
phi /= np.max(np.abs(phi[allowed]))

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2]})

ax1.plot(x_eval, V(x_eval), 'k-', linewidth=1.5, label='V(x)')
ax1.axhline(E_try, color='coral', linestyle='--', linewidth=1,
            label=f'E = {E_try}')
ax1.set_ylabel('Energy')
ax1.set_ylim(0, 10)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title(f'Shooting with E = {E_try} (not an eigenvalue)')

ax2.plot(x_eval, phi, 'steelblue', linewidth=1.5)
ax2.axhline(0, color='k', linewidth=0.3)
ax2.set_xlabel('x')
ax2.set_ylabel(r'$\phi(x)$')
ax2.set_ylim(-5, 5)
ax2.grid(True, alpha=0.3)

# Shade classically forbidden regions
forbidden = V(x_eval) > E_try
ax2.fill_between(x_eval, -5, 5, where=forbidden,
                 alpha=0.06, color='red', label='Classically forbidden')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"phi at x = {x_end}: {phi[-1]:.2e}  (should be ~0 for an eigenstate)")

#%%
r"""
## Why Does It Blow Up?

The solution oscillates nicely in the classically allowed region (where $E > V(x)$, between the turning points), but once it crosses into the forbidden region on the right it explodes. To understand why, consider what the ODE does in a region where the potential is roughly constant at some value $V_0 > E$:

$$
\frac{d^2\phi}{dx^2} = \frac{2m}{\hbar^2}(V_0 - E)\phi = \kappa^2 \phi
$$

where $\kappa = \sqrt{2m(V_0 - E)}/\hbar > 0$. This is easy to solve by inspection — the general solution is a sum of two exponentials:

$$
\phi(x) = A e^{+\kappa x} + B e^{-\kappa x}
$$

Both are perfectly valid solutions of the ODE. The decaying exponential $e^{-\kappa x}$ is what we'd expect of a quantum particle in a forbidden region: the probability density $|\phi|^2 \propto e^{-2\kappa x}$ drops off rapidly — the particle is exponentially unlikely to be found far from the allowed region. The growing exponential $e^{+\kappa x}$, on the other hand, says the particle becomes *more* likely to be found the further you go into the forbidden region, without limit. Such a solution can't be normalized — $\int|\phi|^2 dx$ diverges. Physicists call these **unphysical solutions** and discard them: meaningful quantum states must live in $L^2$ space, meaning they are square-integrable.

That's exactly what we're seeing above. Our numerical integration starts from the left with some arbitrary initial conditions. As it propagates rightward through the forbidden region, it inevitably picks up a component of the growing exponential — even if only through numerical rounding — and that component takes over, sending $\phi$ to $\pm\infty$.

This is the key insight behind the **shooting method**: only for special values of $E$ — the eigenvalues — does the growing exponential have exactly zero amplitude, leaving a normalizable solution that decays on both sides. To find those special energies, we can monitor $\phi$ at some fixed point in the forbidden region and vary $E$. Let's evaluate $\phi$ at $x = 5$ (well past the right turning point) for a range of energies and see what happens:
"""

#%%
# Scan E and record phi at a fixed point in the forbidden region
x_probe = 5.0
E_scan = np.linspace(-2, 14.0, 400)
phi_at_probe = np.zeros(len(E_scan))

for i, E in enumerate(E_scan):
    sol = solve_ivp(
        lambda x, y, E=E: schrodinger_ode(x, y, E),
        [x_start, x_end],
        [1e-5, 1e-3],
        t_eval=[x_probe],
        rtol=1e-10, atol=1e-12,
    )
    phi_at_probe[i] = sol.y[0, 0]

# The raw values span many orders of magnitude, so we use a sign-preserving
# log scale: sign(phi) * log10(1 + |phi|). This compresses the huge
# divergences while keeping the zero crossings clearly visible.
phi_signed_log = np.sign(phi_at_probe) * np.log10(1 + np.abs(phi_at_probe))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(E_scan, phi_signed_log, 'steelblue', linewidth=1)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('Energy E')
ax.set_ylabel(r'sign($\phi$) $\times$ log$_{10}$(1 + |$\phi$|)  at  x = ' + f'{x_probe}')
ax.set_title(f'Solution at x = {x_probe} as a function of energy')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find and print the zero crossings
sign_changes = np.where(np.diff(np.sign(phi_at_probe)))[0]
print("Zero crossings (eigenvalue estimates):")
for idx in sign_changes:
    E_cross = 0.5 * (E_scan[idx] + E_scan[idx + 1])
    print(f"  E ~ {E_cross:.2f}")

#%%
r"""
## Reading the Plot

This is *not* a wave function — it's the value of the wave function at a single fixed point $x = 5$, deep in the classically forbidden region, plotted as a function of the energy parameter $E$ that we feed into the ODE. For each $E$, we integrated the full ODE from left to right and recorded where $\phi$ ended up at $x = 5$. The signed-log scale compresses the enormous range of values so we can see what's going on.

The pattern is striking. For most energies, the solution has diverged wildly by the time it reaches $x = 5$ — either to hugely positive or hugely negative values. But at certain special energies, the curve crosses zero. At those energies, the solution happens to pass through zero at $x = 5$ instead of diverging.

Now imagine pushing the probe point further out — to $x = 6$, $x = 10$, $x = 100$. The divergences get exponentially worse (the growing exponential $e^{+\kappa x}$ has more room to grow), but the zero crossings stay in the same place. That's the crucial point: at those special energies, the solution doesn't just happen to be small at one particular $x$ — it actually *decays* in the forbidden region. No matter how far out you look, it stays well-behaved.

These are the energies where the ODE has a unique solution (up to normalization) that doesn't blow up — a solution that is square-integrable, that lives in $L^2$. In other words: a normalizable wave function, a physically meaningful quantum state. These are the **energy eigenstates** $|\phi_n\rangle$ with eigenvalues $E_n$, and we've just located them by watching where the divergence changes sign.

## A Ground State, but No Ceiling

Look at the left side of the plot: below the first zero crossing ($E \approx 0.51$), $\phi(x=5)$ stays positive and enormous — it never crosses zero again. No matter how low you make $E$, the solution always diverges the same way. There is a **lowest eigenvalue** (the ground state energy), and no eigenstate below it.

On the right side, the crossings keep coming. As $E$ increases the classically allowed region gets wider, allowing more oscillations in $\phi(x)$ before it hits the forbidden region. There is no upper bound — the spectrum extends to infinity. This is a general feature of confining potentials: a discrete, infinite ladder of energy levels with a bottom rung but no top.
"""

#%%
r"""
## Pinning Down the Eigenvalues

We know the eigenvalues live at the zero crossings. To find them precisely, we can use a **root-finding algorithm**. Given a bracket $[E_a, E_b]$ where $\phi(x_\text{probe})$ changes sign, find the $E$ where it equals zero.

We'll use **Brent's method** (`brentq` from `scipy.optimize`) — it combines the safety of bisection with the speed of inverse quadratic interpolation. Since the root is bracketed, convergence is guaranteed; the only question is how many iterations it takes (typically around 10).

The strategy: scan upward in energy with a coarse step of $\Delta E = 0.5$, which is safely below the minimum eigenvalue spacing of any confining potential with $\omega = 1$. Each time $\phi$ changes sign, we've bracketed an eigenvalue — hand it to Brent's method and move on. Stop once we've collected $M$ eigenvalues.

We also switch from `solve_ivp` to `odeint` for the ODE integration — for this kind of smooth, non-stiff problem it's roughly 40x faster.
"""

#%%
from scipy.integrate import odeint
from scipy.optimize import brentq

def find_eigenvalues(lam, M=10, dE=0.5):
    """Find the lowest M eigenvalues of V(x) = 0.5*x^2 + lam*x^4."""
    x_probe = 7.0
    x_span = np.linspace(-x_probe, x_probe, 2000)

    def phi_at_probe(E):
        def ode(y, x):
            return [y[1], (2 * m / hbar**2) * (0.5 * m * omega**2 * x**2 + lam * x**4 - E) * y[0]]
        sol = odeint(ode, [1e-5, 1e-3], x_span)
        return sol[-1, 0]

    eigenvalues = []
    E = 0.0
    phi_prev = phi_at_probe(E)

    while len(eigenvalues) < M:
        E += dE
        phi_curr = phi_at_probe(E)
        if np.sign(phi_curr) != np.sign(phi_prev) and np.sign(phi_prev) != 0:
            E_root = brentq(phi_at_probe, E - dE, E, xtol=1e-10)
            eigenvalues.append(E_root)
        phi_prev = phi_curr

    return np.array(eigenvalues)

# Find the first 10 eigenvalues
M = 10
E_n = find_eigenvalues(lam, M=M)

print(f"First {M} eigenvalues (lam = {lam}):")
for n, E in enumerate(E_n):
    print(f"  E_{n} = {E:.8f}")

# Compare to harmonic oscillator
E_harmonic = find_eigenvalues(0.0, M=M)
print(f"\nEffect of anharmonicity (lam = {lam}):")
print(f"  {'n':>3}  {'Harmonic':>12}  {'Anharmonic':>12}  {'Shift':>10}")
for n in range(M):
    shift = E_n[n] - E_harmonic[n]
    print(f"  {n:3d}  {E_harmonic[n]:12.6f}  {E_n[n]:12.6f}  {shift:+10.6f}")

#%%
r"""
## The Eigenstates Themselves

We've found the eigenvalues — now let's look at the wave functions. For each $E_n$, we integrate the ODE one more time and keep the full solution $\phi_n(x) = \langle x | \phi_n \rangle$. Since the ODE is linear, the overall amplitude is arbitrary; we normalize each eigenstate so that $\int |\phi_n(x)|^2 dx = 1$.
"""

#%%
# Compute all 10 eigenstates (we'll plot the first 4, use all 10 later)
x_wf_span = np.linspace(x_start, x_end, 2000)
dx_wf = x_wf_span[1] - x_wf_span[0]

eigenstates = []
for n in range(M):
    def ode(y, x, E=E_n[n]):
        return [y[1], (2 * m / hbar**2) * (V(x) - E) * y[0]]
    sol = odeint(ode, [1e-5, 1e-3], x_wf_span)
    phi_n = sol[:, 0]
    phi_n /= np.sqrt(np.sum(phi_n**2) * dx_wf)
    eigenstates.append(phi_n)

n_show = 4

# Plot in the same style as the first shooting plot
colors = ['steelblue', 'coral', 'seagreen', 'goldenrod']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2]})

# Top: potential with energy levels
ax1.plot(x_wf_span, V(x_wf_span), 'k-', linewidth=1.5, label='V(x)')
for n in range(n_show):
    ax1.axhline(E_n[n], color=colors[n], linestyle='--', linewidth=1,
                label=f'$E_{n}$ = {E_n[n]:.4f}')
ax1.set_ylabel('Energy')
ax1.set_ylim(0, 10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_title(f'First {n_show} eigenstates of the anharmonic oscillator ($\\lambda$ = {lam})')

# Bottom: wave functions
for n in range(n_show):
    ax2.plot(x_wf_span, eigenstates[n], color=colors[n], linewidth=1.5,
             label=f'$\\phi_{n}(x)$')
ax2.axhline(0, color='k', linewidth=0.3)
ax2.set_xlabel('x')
ax2.set_ylabel(r'$\phi_n(x)$')
ax2.legend(fontsize=8)
ax2.set_ylim(-1, 1)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(x_start, x_end)

plt.tight_layout()
plt.show()

#%%
r"""
You can see the residual divergence creeping in at the edges — our eigenvalues are accurate to $\sim 10^{-10}$, not mathematically exact, so a tiny admixture of the growing exponential eventually takes over.

The wave functions reveal a beautiful pattern. Each successive eigenstate has exactly one more node (zero crossing) than the last: $\phi_0$ has none, $\phi_1$ has one, $\phi_2$ has two, $\phi_3$ has three. This is a general theorem — the $n$-th eigenstate of a one-dimensional potential always has exactly $n$ nodes.

Notice also how the oscillations get shorter in wavelength as $n$ increases. This makes physical sense: a higher eigenvalue means more total energy, and since $E = T + V$, the kinetic energy $T = E - V(x)$ is larger in the allowed region. In the position representation, kinetic energy is encoded in curvature — faster spatial oscillation means higher momentum, and higher momentum means higher kinetic energy. The de Broglie relation $p = \hbar k = h/\lambda$ makes this precise: shorter wavelength $\lambda$ corresponds to higher momentum $p$, and thus higher kinetic energy $p^2/2m$.
"""

#%%
r"""
## Orthonormality

Eigenstates of a Hermitian operator with distinct eigenvalues are guaranteed to be orthogonal: $\langle \phi_m | \phi_n \rangle = 0$ for $m \neq n$. Combined with our normalization $\langle \phi_n | \phi_n \rangle = 1$, this gives the **orthonormality** condition:

$$
\langle \phi_m | \phi_n \rangle = \int_{-\infty}^{\infty} \phi_m^*(x) \phi_n(x) dx = \delta_{mn}
$$

This is not something we imposed — it's a consequence of the eigenvalue equation and the Hermiticity of $\hat{H}$. Let's check how well our numerically computed eigenstates satisfy it by computing the full $10 \times 10$ overlap matrix.

One subtlety: our shooting solutions diverge at the domain edges (we saw this earlier — the eigenvalues are accurate to $\sim 10^{-10}$, not exact). That divergence pollutes the overlap integral if we integrate over the full domain. We avoid this by truncating the integration at $|x| \leq 5$, well into the forbidden region where the physical wave function has decayed to negligible values but before the numerical divergence kicks in.
"""

#%%
# Compute the overlap matrix, truncating at |x| <= 5 to avoid edge divergence
x_cut = 5.0
mask = np.abs(x_wf_span) <= x_cut

# Renormalize on the truncated domain
eigenstates_trunc = []
for phi in eigenstates:
    phi_t = phi[mask]
    phi_t = phi_t / np.sqrt(np.sum(phi_t**2) * dx_wf)
    eigenstates_trunc.append(phi_t)

overlap = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        overlap[i, j] = np.sum(eigenstates_trunc[i] * eigenstates_trunc[j]) * dx_wf

# Print it
print("Overlap matrix <phi_m | phi_n>:\n")
print("     ", "".join(f"   n={j}" for j in range(M)))
for i in range(M):
    row = "".join(f" {overlap[i,j]:+7.4f}" for j in range(M))
    print(f"m={i}  {row}")

print(f"\nLargest off-diagonal element: {np.max(np.abs(overlap - np.eye(M))):.2e}")

#%%
from matplotlib.colors import LogNorm

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(np.abs(overlap), cmap='viridis', norm=LogNorm(vmin=1e-6, vmax=1),
               origin='upper')
ax.set_xlabel('n')
ax.set_ylabel('m')
ax.set_xticks(range(M))
ax.set_yticks(range(M))
ax.set_title(r'$|\langle \phi_m | \phi_n \rangle|$')
plt.colorbar(im, ax=ax, label='Overlap (log scale)')
plt.tight_layout()
plt.show()

#%%
r"""
## Why Are These Wave Functions Real?

You may have noticed something: our entire shooting calculation used only real numbers. In Notebooks 1 and 2, the wave function $\psi(x,t)$ was complex — we needed both real and imaginary parts to encode motion. But the eigenstates $\phi_n(x)$ we just found are purely real. What changed?

Look at the two equations side by side. The time-dependent Schrodinger equation has an $i$ out front:

$$
i\hbar \frac{\partial \psi}{\partial t} = \hat{H}\psi
$$

That $i$ is what forces $\psi$ to be complex — even if you start with a real wave function, the time derivative immediately generates an imaginary part. Remember Notebook 1: we saw that the complex phase of $\psi(x,t)$ is what encodes momentum. When the wave packet moved to the right, the real and imaginary parts oscillated rapidly in space; when it sat at a turning point with zero velocity, $\psi$ was nearly real. The complex structure of $\psi(x,t)$ is inseparable from the dynamics.

But an eigenstate *doesn't move*. It's stationary — the probability density $|\phi(x)|^2$ is constant in time. There is no momentum to encode, no direction of motion, so the wave function doesn't need an imaginary part. And indeed, the time-independent equation we've been solving has no $i$ anywhere:

$$
-\frac{\hbar^2}{2m}\frac{d^2\phi}{dx^2} + V(x)\phi = E\phi
$$

When $V(x)$ is real, every coefficient in this equation is real. If $\phi(x)$ is a real function, the left side is real and the right side is real — there's nothing to generate an imaginary part. Real solutions exist.

### The Linear Algebra Perspective

This has a clean analogue in matrix language. In Notebook 1, we represented $\hat{H}$ as a matrix on our spatial grid. The finite-difference Laplacian is a real symmetric tridiagonal matrix. The potential $V(x)$ contributes a real diagonal matrix. So the full Hamiltonian matrix $H$ is **real symmetric** — not just Hermitian, but actually real.

This is the key distinction from linear algebra:
- **Hermitian** matrices ($H = H^\dagger$) are guaranteed to have real *eigenvalues*. But their eigenvectors may have complex entries.
- **Real symmetric** matrices ($H = H^T$, with all real entries) are a special case of Hermitian. They also have real eigenvalues — but additionally, the eigenvectors can always be chosen to be purely real.

Our Hamiltonian falls in the second, more restrictive category. That's why we could work with real numbers throughout.

### The Physics: Time-Reversal Symmetry

The deeper reason is **time-reversal symmetry**. For a particle in a real potential with no magnetic field, the physics looks the same whether time runs forward or backward. Mathematically, if $\phi(x)$ is an eigenstate with energy $E$, then its complex conjugate $\phi^*(x)$ is also an eigenstate with the same energy (you can verify this by conjugating both sides of the eigenvalue equation — nothing changes when all coefficients are real). If the eigenvalue $E$ is non-degenerate, $\phi$ and $\phi^*$ must be the same state up to an overall constant — which forces $\phi$ to be real (up to a global phase we can absorb into the normalization).

### When Does This Break Down?

Put a charged particle in a magnetic field. The vector potential $\vec{A}$ enters the Hamiltonian through the kinetic energy $(p - eA)^2 / 2m$, which in position representation introduces terms proportional to $i\hbar A(x) \frac{d}{dx}$. The Hamiltonian matrix is still **Hermitian** — eigenvalues are still real, as they must be for any observable. But it is no longer real symmetric: it has imaginary off-diagonal elements. Time-reversal symmetry is broken, $\phi$ and $\phi^*$ are genuinely different states, and the eigenfunctions are irreducibly complex.

The takeaway: eigenvalues are always real (that's the Hermitian guarantee). Eigenstates are real only when the Hamiltonian has time-reversal symmetry — which for us means a real potential and no magnetic field.

### A Note on Linearity

The eigenvalue equation $\hat{H}|\phi\rangle = E|\phi\rangle$ is linear: if $|\phi\rangle$ is a solution, so is $c|\phi\rangle$ for any scalar $c$. This is why normalizing after solving is allowed — the rescaled solution automatically satisfies the same equation. But $c$ can be any *complex* number, not just a real one. Multiplying a real eigenstate by $e^{i\theta}$ rotates it into the complex plane, giving an equally valid — but complex — eigenstate. So when we say eigenstates "are real," we really mean they *can be chosen* real. The physics doesn't change if you multiply by a phase; it's a convention, and a convenient one.
"""

#%%
