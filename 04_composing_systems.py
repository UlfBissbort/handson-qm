#%%
r"""
# Composing Systems: From Classical Mechanics to Quantum States

So far every notebook has described a single particle. Before we add a second one, it's worth asking a question that sounds almost too simple: **what is a subsystem?** In classical mechanics the answer seems obvious: this ball and that ball, this magnet and that magnet. Each has its own position and orientation, and we can always tell which is which. Quantum mechanics makes this murkier. Two electrons aren't just similar, they are indistinguishable. Neither carries a label, so what does it even mean to talk about "electron 1"? And in quantum field theory, particles are excitations of an underlying field: the field is the system, and the particles are patterns in it. The way out is to notice that "subsystem" was never handed to us by nature. It's a choice: we decide which degrees of freedom describe the system and how to group them. In this notebook we'll work through several concrete choices of subsystems, first classically and then quantum mechanically, and compare them: where the two descriptions agree, where they part ways, and why quantum mechanics needs a fundamentally different way of combining systems.
"""

#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%%
r"""
## Two Particles in Three Dimensions

Let's start with the most familiar case. The state of a classical particle is its position and its momentum, six numbers that we can stack into one vector:

$$
\vec z_1 = \begin{pmatrix} \vec r_1 \\ \vec p_1 \end{pmatrix} \in \mathbb{R}^6, \qquad \vec z_2 = \begin{pmatrix} \vec r_2 \\ \vec p_2 \end{pmatrix} \in \mathbb{R}^6
$$

To describe both particles at once, we simply stack the two vectors. Dimensions add: $6 + 6 = 12$.

$$
\vec z = \begin{pmatrix} \vec z_1 \\ \vec z_2 \end{pmatrix} \in \mathbb{R}^{12} = \mathbb{R}^6 \oplus \mathbb{R}^6
$$

Without an interaction, the equations of motion split into two independent blocks, one per particle. With an interaction, the force on particle 1 depends on the coordinates of particle 2, and vice versa, so the blocks couple. But the state is still the same 12 numbers: **interactions change the dynamics, not the state space**.
"""

#%%
r"""
## Two Pendula Connected by a Spring

A pendulum is not a point particle. The bob and the rod are made of something like $10^{23}$ atoms, each with its own position and momentum. Yet nobody describes a swinging pendulum that way. At the gentle energies of a swing, the rod doesn't stretch and the atoms don't rearrange; the only motion that actually happens is the swing itself. So we describe each pendulum by a single **effective degree of freedom**, its angle $\theta$, and the pair by $(\theta_1, \dot\theta_1, \theta_2, \dot\theta_2)$. This description is valid at sufficiently low energy; hit the pendulum hard enough and the hidden degrees of freedom come back.

Once chosen, the angles combine like any other coordinates: stacked side by side. A weak spring between the bobs couples $\theta_1$ and $\theta_2$ in the equations of motion, and the swing passes from one pendulum to the other and back. Even here the split is not unique: the in-phase and out-of-phase **normal modes** are two independent oscillators, neither of which is "a pendulum".

This principle carries over to quantum mechanics: we choose the degrees of freedom that matter at the energies we care about, and ignore the rest. An atom becomes a single particle, a vibrating molecule a single oscillator.
"""

#%%
r"""
## Two Magnets

Our third example has no position at all. Picture two compass needles pinned a distance $d$ apart in a plane, free only to turn. Each needle is fully described by a single angle $\phi$ (and its angular velocity), and it has a moment of inertia $I$. This is an **internal degree of freedom**: it says how the object is oriented, not where it is, and it's the closest classical relative of quantum spin. Combined, the two angles are again just listed together.

A uniform external field gives each needle a preferred direction, like Earth's field for a compass. The needles also feel each other's magnetic field: this magnetic force couples $\phi_1$ and $\phi_2$ in the equations of motion, and it falls off as $1/d^3$. Let's tilt needle 1, release it, and watch both angles:
"""

#%%
inertia = 1.0               # moment of inertia of each needle
mu = 1.0                    # magnetic moment of each needle
B = 1.0                     # external field along x, the needles' preferred direction
d = 2.5                     # distance between the needles, along x
phi1_0 = np.radians(20)     # needle 1 starts tilted; needle 2 starts aligned

def magnets_rhs(t, y):
    phi1, phi2, w1, w2 = y
    c = mu**2 / d**3        # strength of the magnetic coupling between the needles
    torque1 = -mu * B * np.sin(phi1) - c * (3 * np.sin(phi1) * np.cos(phi2) - np.sin(phi1 - phi2))
    torque2 = -mu * B * np.sin(phi2) - c * (3 * np.cos(phi1) * np.sin(phi2) + np.sin(phi1 - phi2))
    return [w1, w2, torque1 / inertia, torque2 / inertia]

t = np.linspace(0, 150, 3000)
sol = solve_ivp(magnets_rhs, [0, t[-1]], [phi1_0, 0, 0, 0],
                t_eval=t, rtol=1e-10, atol=1e-12)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax1.plot(t, np.degrees(sol.y[0]), 'steelblue', linewidth=1)
ax1.set_ylabel(r'$\phi_1$ (degrees)')
ax1.set_title(f'Two coupled magnets, d = {d}')
ax1.grid(True, alpha=0.2)
ax2.plot(t, np.degrees(sol.y[1]), 'coral', linewidth=1)
ax2.set_ylabel(r'$\phi_2$ (degrees)')
ax2.set_xlabel('Time')
ax2.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

#%%
r"""
Notice that the swing passes from needle 1 to needle 2 and back, just like the pendula. Move the needles further apart and the exchange slows down.
"""

#%%
r"""
## What the Classical Examples Have in Common

- Combining systems = listing coordinates side by side. Dimensions add.
- Interactions only couple the coordinates in the equations of motion.
- This works because a classical state is a single point. In quantum mechanics this is what breaks.
"""

#%%
r"""
## Two Quantum Spins

Let's shrink our magnets down to single ions. In a magnetic insulator such as nickel oxide, the ions sit fixed on a crystal lattice and their electrons can't move from one ion to the next. All that remains of each ion is the orientation of its magnetic moment: an effective degree of freedom, just like the angle of a pendulum.

Classically, a spinning charged object carries angular momentum and, with it, a magnetic moment. Quantum particles carry angular momentum too, called **spin**, but nothing is actually rotating: it's an intrinsic property, like charge or mass, and measured along any axis it takes only a few discrete values. In a Ni²⁺ ion, two electron spins lock together into a single **spin 1**, whose component along the $z$ axis can only be $+1$, $0$ or $-1$ (in units of $\hbar$).

The state of one spin assigns a complex amplitude to each of these three outcomes:

$$
|\psi\rangle = a\,|{+1}\rangle + b\,|0\rangle + c\,|{-1}\rangle = \begin{pmatrix} a \\ b \\ c \end{pmatrix} \in \mathbb{C}^3
$$

Where the needle had one continuous angle, the spin has three amplitudes. Its dynamics follow the same recipe as in the earlier notebooks: a Hamiltonian, here the energy of the magnetic moment in a field, drives the time evolution of $|\psi\rangle$. The result looks familiar: the expectation value of the spin precesses around the field, just like a classical spinning magnet.

[Placeholder: a single spin-1 precessing in a field, ⟨S_x⟩, ⟨S_y⟩, ⟨S_z⟩ vs. time]

Now take two ions, each with its own spin. Each one on its own is described by a separate state in $\mathbb{C}^3$. The classical recipe would stack them into $3 + 3 = 6$ numbers. But when we measure both spins, we get one of $3 \times 3 = 9$ joint outcomes, from $(+1, +1)$ to $(-1, -1)$, and quantum mechanics assigns an amplitude to each of them. The state of the pair therefore lives in a 9-dimensional space, the **tensor product** $\mathbb{C}^3 \otimes \mathbb{C}^3$. Concretely, each joint basis state is a Kronecker product of single-spin vectors, for example:

$$
|{+1}\rangle \otimes |0\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} \otimes \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} = (0, 1, 0,\; 0, 0, 0,\; 0, 0, 0)^T
$$

[Placeholder: the 9 joint basis states built with `np.kron`]

If each spin has its own state, the pair is in a **product state** $|\psi_1\rangle \otimes |\psi_2\rangle$. That's the closest thing to the classical picture. But most states in the 9-dimensional space are not of this form, for example

$$
|\psi\rangle = \frac{1}{\sqrt 3}\bigl(|{+1}, {-1}\rangle - |0, 0\rangle + |{-1}, {+1}\rangle\bigr)
$$

which can't be split into one state per spin. Like any state of the pair, $|\psi\rangle$ is a vector in the abstract ket space of the combined system. The nine joint kets, built from the basis states of both spins, form a basis of that space. In the Kronecker ordering, the first spin picks the block and the second spin the position within it:

$$
|{+1},{+1}\rangle,\; |{+1},0\rangle,\; |{+1},{-1}\rangle,\; |0,{+1}\rangle,\; \ldots,\; |{-1},{-1}\rangle
$$

Decomposing a state in this basis, we can represent any state of the pair simply as a tuple of nine complex numbers, its amplitudes. For $|\psi\rangle$:

$$
\frac{1}{\sqrt 3}\,(0, 0, 1,\; 0, -1, 0,\; 1, 0, 0)^T
$$

The gap grows quickly: two spins need 9 amplitudes instead of 6, and $N$ spins need $3^N$ instead of $3N$.

[Placeholder: 3^N vs. 3N for N = 1 … 10]
"""

#%%
r"""
## Sum or Product?

Let's step back and compare the two ways of combining vector spaces. The **direct sum** $\mathbb{C}^3 \oplus \mathbb{C}^3$ stacks two vectors on top of each other: $3 + 3 = 6$ components, just like the classical coordinates of our particles, pendula and needles. The **tensor product** $\mathbb{C}^3 \otimes \mathbb{C}^3$ has one component for every pair of basis states: $3 \times 3 = 9$.

Quantum mechanics uses both, for different situations. Superposition combines alternatives, so a direct sum describes one system that is in part 1 *or* in part 2, for example a single electron that sits either on ion 1 or on ion 2. The tensor product describes two systems that both have their own properties at the same time: spin 1 *and* spin 2.

Could we still describe our two spins in the sum space? Without coupling, almost: stack $(\psi_1, \psi_2)$ and let each part follow its own Schrödinger equation. But three things go wrong:

- The sum space has a single norm, $\|\psi_1\|^2 + \|\psi_2\|^2 = 1$. The probabilities are shared between the parts, as if the pair were either spin 1 or spin 2.
- The relative phase between $\psi_1$ and $\psi_2$ becomes measurable, although each spin's own global phase should not be.
- If both spins sit in stationary states with energies $E_1$ and $E_2$, the pair should have energy $E_1 + E_2$. In the sum space it gets a weighted average instead.

The product space gets all three right: probabilities of independent spins multiply, each spin's phase stays global, and energies add. With coupling, the sum space fails completely. Keeping one state per spin then requires nonlinear equations, an approximation known as mean-field theory, and those can never produce a state like $|\psi\rangle$ above.

Using the product space is a postulate of quantum mechanics, for spins as well as for particles. We adopt it because it describes composite systems better than any alternative we know of, and quantum mechanics is among the most thoroughly tested theories we have.

Next, we'll couple the two spins and see how an interaction looks in this product space.
"""
