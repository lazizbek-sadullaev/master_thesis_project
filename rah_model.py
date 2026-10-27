# Rotary regenerative air preheater (RAH): reduced discrete model from my M.S. thesis
#   x(n+1) = (1 - beta*h) * y(n) + beta*h * q      (drum part leaving the gas side)
#   y(n+1) = (1 - alpha*h) * x(n) + alpha*h * p    (drum part leaving the air side)
# In matrix form: z(n+1) = A z(n) + r,  z = (x, y)

import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Parameters (thesis, Table 2.1) ----
alpha, beta = 0.048, 0.068   # heat-transfer coefficients
gamma, delta = 0.041, 0.029  # outlet coefficients
h = 0.25                     # half-turn time step
p = 32.0                     # incoming air temperature, deg C
q = 282.0                    # incoming gas temperature, deg C

# ---- 2. Matrix A and vector r ----
A = np.array([[0,             1 - beta * h],
              [1 - alpha * h, 0           ]])
r = np.array([beta * h * q, alpha * h * p])

# ---- 3. Stability check: all |eigenvalues| < 1 ? ----
eig = np.linalg.eigvals(A)
print("eigenvalues:", eig)
print("|eigenvalues|:", np.abs(eig))

# ---- 4. Steady state: solve (I - A) z_bar = r ----
z_bar = np.linalg.solve(np.eye(2) - A, r)
print("steady state (x_bar, y_bar):", z_bar)

# same thing from the closed-form (corrected) formula, as a check
den = alpha + beta - alpha * beta * h
x_formula = (alpha * p * (1 - beta * h) + beta * q) / den
y_formula = (beta * q * (1 - alpha * h) + alpha * p) / den
print("closed-form check:", x_formula, y_formula)

# ---- 5. Iterate z(n+1) = A z(n) + r until it stops changing ----
N_max = 1000
tol = 0.01                     # stop when change < 0.01 deg C
Z = np.zeros((N_max + 1, 2))
Z[0] = [p, p]                  # cold start: drum at air temperature
# Z[0] = [q, q]                # alternative: hot start, drum at gas temperature

for n in range(N_max):
    Z[n + 1] = A @ Z[n] + r
    if np.max(np.abs(Z[n + 1] - Z[n])) < tol:
        break

Z = Z[:n + 2]                  # keep only the steps actually computed
print("steps to steady state:", n + 1)
print("last state:", Z[-1])

# outlet temperatures u (hot air) and v (cooled gas)
u = p + gamma * h * (Z[:, 0] - p)
v = q + delta * h * (Z[:, 1] - q)
print("outlet air u, outlet gas v at the end:", u[-1], v[-1])

# ---- 6. Plot ----
steps = np.arange(len(Z))
plt.figure(figsize=(7, 4))
plt.plot(steps, Z[:, 0], label="x(n): drum leaving gas side")
plt.plot(steps, Z[:, 1], label="y(n): drum leaving air side")
plt.axhline(z_bar[0], linestyle="--", color="gray", linewidth=1)
plt.axhline(z_bar[1], linestyle=":", color="gray", linewidth=1)
plt.xlabel("half-turns n")
plt.ylabel("drum (packing) temperature, °C")
plt.title("RAH reduced model: convergence to steady state")
plt.legend()
plt.tight_layout()
plt.savefig("rah_convergence.png", dpi=200)
plt.show()
