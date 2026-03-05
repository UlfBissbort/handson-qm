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
E_scan = np.linspace(0.1, 12.0, 500)
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

# Normalize for display: divide by the envelope to see the sign changes
# The raw values span many orders of magnitude, so we plot the sign-preserving log
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

#%%
