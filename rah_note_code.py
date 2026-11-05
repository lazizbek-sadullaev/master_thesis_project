# Reproduces all figures and numbers in
# L. Sadullaev, M. Bekimov, "Regression-based parameter identification for a
# reduced-order discrete model of rotary regenerative air preheaters" (2026).
# Requires numpy and matplotlib. Run:  python rah_note_code.py
# Model: z(n+1) = A z(n) + r,  outlets u = p + gamma*h*(x - p),  v = q + delta*h*(y - q)

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"

# ---- parameters (illustrative values from Azamov & Bekimov 2016) ----
alpha, beta, gamma, delta = 0.048, 0.068, 0.041, 0.029
h, p, q = 0.25, 32.0, 282.0

A = np.array([[0,             1 - beta * h],
              [1 - alpha * h, 0           ]])
r = h * np.array([beta * q, alpha * p])
mu = np.sqrt((1 - alpha * h) * (1 - beta * h))
z_bar = np.linalg.solve(np.eye(2) - A, r)
print("mu =", mu, " steady state =", z_bar)


def simulate(z0, N):
    Z = np.zeros((N + 1, 2))
    Z[0] = z0
    for n in range(N):
        Z[n + 1] = A @ Z[n] + r
    return Z


# ---- check Propositions 1-2 (closed-form solution) against direct iteration ----
D = alpha + beta - alpha * beta * h
z0 = np.array([60.0, 250.0])
Z = simulate(z0, 9)
worst = 0.0
for n in range(1, 10):
    if n % 2 == 0:
        x = mu**n * z0[0] + (1 - mu**n) * (alpha * p * (1 - beta * h) + beta * q) / D
        y = mu**n * z0[1] + (1 - mu**n) * (beta * q * (1 - alpha * h) + alpha * p) / D
    else:
        M = mu**(n - 1)
        x = M * (1 - beta * h) * z0[1] + (beta * q * (1 - mu**(n + 1)) + alpha * p * (1 - beta * h) * (1 - M)) / D
        y = M * (1 - alpha * h) * z0[0] + (alpha * p * (1 - mu**(n + 1)) + beta * q * (1 - alpha * h) * (1 - M)) / D
    worst = max(worst, np.max(np.abs(Z[n] - [x, y])))
print("max |closed form - iteration| over n=1..9:", worst)

# ---- Figure 1: convergence from a cold start ----
Z = simulate([p, p], 400)
n = np.arange(len(Z))
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.plot(n, Z[:, 0], color=BLUE, lw=1.5, label=r"$x(n)$")
ax.plot(n, Z[:, 1], color=ORANGE, lw=1.5, ls="--", label=r"$y(n)$")
ax.axhline(z_bar[0], color=GRAY, lw=0.8, ls=":")
ax.axhline(z_bar[1], color=GRAY, lw=0.8, ls=":")
ax.set_xlabel("half-turns $n$")
ax.set_ylabel("packing temperature, °C")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("fig_convergence.pdf")

# ---- Figure 2: phase plane, even and odd iterates ----
Z = simulate([60.0, 250.0], 300)
fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.plot(Z[0::2, 0], Z[0::2, 1], "o", ms=3, color=BLUE, label="even $n$")
ax.plot(Z[1::2, 0], Z[1::2, 1], "s", ms=3, color=ORANGE, mfc="none", label="odd $n$")
ax.plot(*z_bar, "k*", ms=9, label=r"$\bar z$")
ax.set_xlabel("$x$, °C")
ax.set_ylabel("$y$, °C")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("fig_phase.pdf")


# ---- identification by linear regression on outlet data (Proposition 4) ----
def identify(U, V):
    # U(n+1) = aU + bU V(n),   V(n+1) = aV + bV U(n)
    M = np.column_stack([np.ones(len(V) - 1), V[:-1]])
    aU, bU = np.linalg.lstsq(M, U[1:], rcond=None)[0]
    M = np.column_stack([np.ones(len(U) - 1), U[:-1]])
    aV, bV = np.linalg.lstsq(M, V[1:], rcond=None)[0]
    gh = aU / (q - p)
    dh = aV / (p - q)
    bh = 1 - bU * dh / gh
    ah = 1 - bV * gh / dh
    return np.array([ah, bh, gh, dh]) / h


Z = simulate([p, p], 100)
U = gamma * h * (Z[:, 0] - p)
V = delta * h * (Z[:, 1] - q)
true = np.array([alpha, beta, gamma, delta])
print("noise-free estimate:", identify(U, V))

rng = np.random.default_rng(1)
sigmas = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02])
med = []
for s in sigmas:
    est = np.array([identify(U + rng.normal(0, s, U.shape), V + rng.normal(0, s, V.shape))
                    for _ in range(500)])
    med.append(100 * np.median(np.abs(est - true) / true, axis=0))
med = np.array(med)
print("sigma, median relative error % (alpha, beta, gamma, delta):")
for s, m in zip(sigmas, med):
    print(s, np.round(m, 2))

# ---- Figure 3: estimation error vs noise (Section 5.3) ----
fig, ax = plt.subplots(figsize=(5.2, 2.8))
styles = [(r"$\alpha$", BLUE, "o", "-"), (r"$\beta$", ORANGE, "s", "--"),
          (r"$\gamma$", "#1baf7a", "^", "-."), (r"$\delta$", "#6250d6", "D", ":")]
for k, (lab, c, mk, ls) in enumerate(styles):
    ax.loglog(sigmas, med[:, k], color=c, marker=mk, ms=4, ls=ls, lw=1.5, label=lab)
ax.set_xlabel(r"noise level $\sigma$, °C")
ax.set_ylabel("median relative error, %")
ax.legend(frameon=False, ncol=4, loc="upper left")
fig.tight_layout()
fig.savefig("fig_identification.pdf")

# ---- Table 1: two-step method of Azamov & Bekimov (2016) on exact model data ----
for z0 in ([100.0, 150.0], [60.0, 250.0]):
    z1 = A @ np.array(z0) + r
    z2 = A @ z1 + r
    u1, u2 = gamma * h * (z1[0] - p), gamma * h * (z2[0] - p)
    v1, v2 = delta * h * (z1[1] - q), delta * h * (z2[1] - q)
    d = u1 * v1 - u2 * v2
    x1 = ((u1 - u2) * v2 * p + (v1 - v2) * u1 * q) / d
    y1 = ((u1 - u2) * v1 * p + (v1 - v2) * u2 * q) / d
    gh, dh = u1 / (x1 - p), v1 / (y1 - q)
    ah = (x1 - q) / (x1 - p) - (y1 - q) / (x1 - p) * v2 / v1
    bh = (y1 - p) / (y1 - q) - (x1 - p) / (y1 - q) * u2 / u1
    print("two-step, z0 =", z0, "->", np.round([ah / h, bh / h, gh / h, dh / h], 4))
