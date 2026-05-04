import numpy as np
import matplotlib.pyplot as plt
import random

###VALORI
def f(x):
    return x**4 - 12*x**3 + 30*x**2 + 12

def f_deriv(x):
    return 4*x**3 - 36*x**2 + 60*x

x0 = 0.0 #capatul stang
xn = 2.0 #capatul drept
n = 10 #numarul de subintervale (n+1 noduri)
m = 3 #gradul polinomului (m < 6)
da = f_deriv(x0) #f'(x0)
db = f_deriv(xn) #f'(xn)
x_bar = 1.5 #punctul in care se evalueaza aproximarile

###GENERARE
random.seed(42)
interior = sorted(random.uniform(x0, xn) for _ in range(n - 1))
nodes = [x0] + interior + [xn]

nodes = sorted(set(nodes))
while len(nodes) < n + 1:
    nodes.append(random.uniform(x0, xn))
    nodes = sorted(set(nodes))
#nodes = nodes[:n + 1]

xi = np.array(nodes)
yi = np.array([f(x) for x in xi])

print("=" * 55)
print("Aproximare prin MMCP si Spline Cubice")
print("=" * 55)
print(f"\nNoduri xi: {np.round(xi, 4)}")
print(f"Valori yi: {np.round(yi, 4)}")
print(f"\nGradul polinomului m = {m}")
print(f"Punct de evaluare x_bar = {x_bar}")
print(f"f(x_bar) real = {f(x_bar):.6f}")

#polinom grad m
A_gram = np.zeros((m + 1, m + 1))
b_vec = np.zeros(m + 1)

for i in range(m + 1):
    for j in range(m + 1):
        A_gram[i, j] = np.sum(xi ** (i + j))
    b_vec[i] = np.sum(yi * xi ** i)

coeffs = np.linalg.solve(A_gram, b_vec) #coeffs[0] + coeffs[1]*x + ...

#schema lui Horner pentru evaluare in x_bar
##def horner(coeffs, x):
    ##result = coeffs[-1]
    ##for c in reversed(coeffs[:-1]):
        ##result = result * x + c
    ##return result

def horner_eval(coeffs_asc, x):
    coeffs_desc = list(reversed(coeffs_asc))
    result = coeffs_desc[0]
    for c in coeffs_desc[1:]:
        result = result * x + c
    return result

Pm_xbar = horner_eval(coeffs, x_bar)

#Eroarea globala MCMMP
eroare_mmcp = sum(abs(horner_eval(coeffs, xi[i]) - yi[i]) for i in range(len(xi)))

print("\n" + "-" * 55)
print("  APROXIMARE POLINOMIALA (MCMMP) - Schema Horner")
print("-" * 55)
print(f"Coeficienti P_m: {np.round(coeffs, 6)}")
print(f"P_{m}(x_bar)  = {Pm_xbar:.6f}")
print(f"|P_{m}(x_bar) - f(x_bar)| = {abs(Pm_xbar - f(x_bar)):.6f}")
print(f"Suma |P_m(xi) - yi| = {eroare_mmcp:.6f}")

###SPLINE CUBICE
N = len(xi) - 1
h = np.diff(xi) #h[i] = xi[i+1] - xi[i]

#HA=f
size = N + 1
H_mat = np.zeros((size, size))
f_vec = np.zeros(size)

#ecuatia 0
H_mat[0, 0] = 2 * h[0]
H_mat[0, 1] = h[0]
f_vec[0] = 6 * ((yi[1] - yi[0]) / h[0] - da)

#i = 1 ... N-1
for i in range(1, N):
    H_mat[i, i - 1] = h[i - 1]
    H_mat[i, i] = 2 * (h[i - 1] + h[i])
    H_mat[i, i + 1] = h[i]
    f_vec[i] = 6 * ((yi[i + 1] - yi[i]) / h[i] - (yi[i] - yi[i - 1]) / h[i - 1])

#ecuatia N
H_mat[N, N - 1] = h[N - 1]
H_mat[N, N] = 2 * h[N - 1]
f_vec[N] = 6 * (db - (yi[N] - yi[N - 1]) / h[N - 1])

#HA = f pentru a gasi vectorul de coeficienti A
A_coef = np.linalg.solve(H_mat, f_vec)


def spline_eval_curs(x_val):
    #i0 astfel incat x_val apartine [xi_i0, xi_{i0+1}]
    idx = np.searchsorted(xi, x_val, side='right') - 1 #inserez
    idx = int(np.clip(idx, 0, N - 1))
    i0 = idx

    hi0 = h[i0]

    #b_i0 si c_i0
    b_i0 = (yi[i0 + 1] - yi[i0]) / hi0 - (hi0 * (A_coef[i0 + 1] - A_coef[i0])) / 6.0
    c_i0 = (xi[i0 + 1] * yi[i0] - xi[i0] * yi[i0 + 1]) / hi0 - (
                hi0 * (xi[i0 + 1] * A_coef[i0] - xi[i0] * A_coef[i0 + 1])) / 6.0

    # Calculam valoarea finala a polinomului
    termen1 = ((x_val - xi[i0]) ** 3 * A_coef[i0 + 1]) / (6.0 * hi0)
    termen2 = ((xi[i0 + 1] - x_val) ** 3 * A_coef[i0]) / (6.0 * hi0)
    termen3 = b_i0 * x_val
    termen4 = c_i0

    return termen1 + termen2 + termen3 + termen4


Sf_xbar = spline_eval_curs(x_bar)

print("\n" + "-" * 55)
print("SPLINE CUBICE")
print("-" * 55)
print(f"Coeficienti A: {np.round(A_coef, 6)}")
print(f"S_f(x_bar)  = {Sf_xbar:.6f}")
print(f"|S_f(x_bar) - f(x_bar)| = {abs(Sf_xbar - f(x_bar)):.6f}")

###GRAFIC
x_plot = np.linspace(x0, xn, 500)
y_real  = np.array([f(x) for x in x_plot])
y_poly  = np.array([horner_eval(coeffs, x) for x in x_plot])
y_spline = np.array([spline_eval_curs(x) for x in x_plot])

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_real,   'k-',  linewidth=2,   label='f(x) reala')
plt.plot(x_plot, y_poly,   'b--', linewidth=1.8, label=f'P_{m}(x) MCMMP grad {m}')
plt.plot(x_plot, y_spline, 'r-.',  linewidth=1.8, label='$S_f(x)$ Spline Cubice')
plt.scatter(xi, yi, color='green', zorder=5, s=60, label='Noduri interpolate')
plt.axvline(x=x_bar, color='purple', linestyle=':', linewidth=1.2, label=f'x̄ = {x_bar}')
plt.scatter([x_bar], [f(x_bar)], color='purple', zorder=6, s=80, marker='*')
plt.title('Aproximare prin MCMMP si Spline Cubic', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafic_aproximare.png', dpi=150)
plt.show()
print("\nGraficul a fost salvat ca 'grafic_aproximare.png'")
print("=" * 55)
