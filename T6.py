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
n  = 10 #numarul de subintervale (n+1 noduri)
m  = 3 #gradul polinomului (m < 6)
da = f_deriv(x0) #f'(x0)
db = f_deriv(xn) #f'(xn)
x_bar = 1.5 #punctul in care se evalueaza aproximarile

###GENERARE
random.seed(42)
interior = sorted(random.uniform(x0, xn) for _ in range(n - 1))
#x_{i-1} < x_i
nodes = [x0] + interior + [xn]
#verificam monotonia
nodes = sorted(set(nodes))
while len(nodes) < n + 1:
    nodes.append(random.uniform(x0, xn))
    nodes = sorted(set(nodes))
nodes = nodes[:n + 1]

xi = np.array(nodes)
yi = np.array([f(x) for x in xi])

print("=" * 55)
print("CALCUL NUMERIC - Aproximare prin MMCP si Spline Cubic")
print("=" * 55)
print(f"\nNoduri xi: {np.round(xi, 4)}")
print(f"Valori yi: {np.round(yi, 4)}")
print(f"\nGradul polinomului m = {m}")
print(f"Punct de evaluare x_bar = {x_bar}")
print(f"f(x_bar) real = {f(x_bar):.6f}")

#polinom grad m
A_gram = np.zeros((m + 1, m + 1))
b_vec  = np.zeros(m + 1)

for i in range(m + 1):
    for j in range(m + 1):
        A_gram[i, j] = np.sum(xi ** (i + j))
    b_vec[i] = np.sum(yi * xi ** i)

coeffs = np.linalg.solve(A_gram, b_vec)  #coeffs[0] + coeffs[1]*x + ...

#schema lui Horner pentru evaluare in x_bar
def horner(coeffs, x):
    result = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result

def horner_eval(coeffs_asc, x):
    coeffs_desc = list(reversed(coeffs_asc))
    result = coeffs_desc[0]
    for c in coeffs_desc[1:]:
        result = result * x + c
    return result

Pm_xbar = horner_eval(coeffs, x_bar)

#Eroarea globala MMCP
eroare_mmcp = sum(abs(horner_eval(coeffs, xi[i]) - yi[i]) for i in range(len(xi)))

print("\n" + "-" * 55)
print("  APROXIMARE POLINOMIALA (MMCP) - Schema Horner")
print("-" * 55)
print(f"Coeficienti P_m: {np.round(coeffs, 6)}")
print(f"P_{m}(x_bar)  = {Pm_xbar:.6f}")
print(f"|P_{m}(x_bar) - f(x_bar)| = {abs(Pm_xbar - f(x_bar)):.6f}")
print(f"Suma |P_m(xi) - yi| = {eroare_mmcp:.6f}")

###SPLINE CUBICE
N = len(xi) - 1

h = np.diff(xi) #h[i] = xi[i+1] - xi[i]

size = N + 1
mat = np.zeros((size, size))
rhs = np.zeros(size)

mat[0, 0] = h[0] / 3.0
mat[0, 1] = h[0] / 6.0
rhs[0] = (yi[1] - yi[0]) / h[0] - da

#ecuatii interioare pentru M[i], i=1..N-1
for i in range(1, N):
    mat[i, i-1] = h[i-1] / 6.0
    mat[i, i]   = (h[i-1] + h[i]) / 3.0
    mat[i, i+1] = h[i] / 6.0
    rhs[i] = (yi[i+1] - yi[i]) / h[i] - (yi[i] - yi[i-1]) / h[i-1]

#ecuatia pentru M[N] din conditia S'(xn) = db
mat[N, N-1] = h[N-1] / 6.0
mat[N, N]   = h[N-1] / 3.0
rhs[N] = db - (yi[N] - yi[N-1]) / h[N-1]

M = np.linalg.solve(mat, rhs)

def spline_eval(x_val):
    #intervalul [xi[i], xi[i+1]]
    idx = np.searchsorted(xi, x_val, side='right') - 1
    idx = int(np.clip(idx, 0, N - 1))
    i = idx
    hi = h[i]
    t = x_val - xi[i]
    s = t / hi

    val = (yi[i] * (1 - s)
           + yi[i+1] * s
           + hi**2 / 6.0 * ((1 - s) * ((1 - s)**2 - 1) * M[i]
                             + s * (s**2 - 1) * M[i+1]))
    return val

Sf_xbar = spline_eval(x_bar)

print("\n" + "-" * 55)
print("  SPLINE CUBIC CLAMPED C^2")
print("-" * 55)
print(f"Momente M: {np.round(M, 6)}")
print(f"S_f(x_bar)  = {Sf_xbar:.6f}")
print(f"|S_f(x_bar) - f(x_bar)| = {abs(Sf_xbar - f(x_bar)):.6f}")

###GRAFIC
x_plot = np.linspace(x0, xn, 500)
y_real  = np.array([f(x) for x in x_plot])
y_poly  = np.array([horner_eval(coeffs, x) for x in x_plot])
y_spline = np.array([spline_eval(x) for x in x_plot])

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_real,   'k-',  linewidth=2,   label='f(x) reala')
plt.plot(x_plot, y_poly,   'b--', linewidth=1.8, label=f'P_{m}(x) MMCP grad {m}')
plt.plot(x_plot, y_spline, 'r-.',  linewidth=1.8, label='$S_f(x)$ Spline Cubice')
plt.scatter(xi, yi, color='green', zorder=5, s=60, label='Noduri interpolate')
plt.axvline(x=x_bar, color='purple', linestyle=':', linewidth=1.2, label=f'x̄ = {x_bar}')
plt.scatter([x_bar], [f(x_bar)], color='purple', zorder=6, s=80, marker='*')
plt.title('Aproximare prin MMCP si Spline Cubic', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafic_aproximare.png', dpi=150)
plt.show()
print("\nGraficul a fost salvat ca 'grafic_aproximare.png'")
print("=" * 55)
