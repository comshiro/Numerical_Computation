import numpy as np

###EXERCITIUL 1
#################################
def find_pivot(A):
    n = A.shape[0]
    p, q = 0, 1
    max_val = 0.0
    for i in range(n):
        for j in range(i + 1, n):   # j > i => triunghiul superior
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p, q = i, j
    return p, q, max_val

def compute_tcs(A, p, q):
    alpha = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
    semn_alpha = 1.0 if alpha >= 0 else -1.0
    t = -alpha + semn_alpha * np.sqrt(alpha**2 + 1.0)   #formula (3)
    c = 1.0 / np.sqrt(1.0 + t**2)                        #formula (4)
    s = t * c                                              # formula (4)
    return t, c, s

def update_A(A, p, q, c, s):
    n = A.shape[0]
    a_p = A[p, :].copy()
    a_q = A[q, :].copy()

    #elemente off-diagonale (liniile/coloanele p si q, j != p, j != q)
    for j in range(n):
        if j == p or j == q:
            continue
        A[p, j] =  c * a_p[j] - s * a_q[j]
        A[j, p] =  A[p, j]                    # simetrie
        A[q, j] =  s * a_p[j] + c * a_q[j]
        A[j, q] =  A[q, j]                    # simetrie

    #elemente diagonale (formula exacta din R*A*R^T)
    A[p, p] =  c**2 * a_p[p] + s**2 * a_q[q] - 2.0 * c * s * a_p[q]
    A[q, q] =  s**2 * a_p[p] + c**2 * a_q[q] + 2.0 * c * s * a_p[q]

    #elementul pivot devine 0
    A[p, q] = 0.0
    A[q, p] = 0.0

    return A

def update_U(U, p, q, c, s):
    n = U.shape[0]
    for i in range(n):
        u_ip_vechi = U[i, p]
        u_iq_vechi = U[i, q]
        U[i, p] =  c * u_ip_vechi - s * u_iq_vechi
        U[i, q] =  s * u_ip_vechi + c * u_iq_vechi
    return U

def jacobi(A_init, eps=1e-10, k_max=10000):
    n = A_init.shape[0]
    A = A_init.copy().astype(float)

    # k = 0; U = I_n
    k = 0
    U = np.eye(n)

    #p, q (formula 1) si parametrii initiali (formule 3, 4)
    p, q, max_val = find_pivot(A)
    t, c, s = compute_tcs(A, p, q)

    # while (A != matrice diagonala si k <= k_max)
    while max_val > eps and k <= k_max:

        #formula (5): A = R_pq * A * R_pq^T, in-place
        A = update_A(A, p, q, c, s)

        #formula (7): U = U * R_pq^T, in-place
        U = update_U(U, p, q, c, s)

        #p, q si parametrii (formule 1, 3, 4)
        p, q, max_val = find_pivot(A)
        if max_val > eps:
            t, c, s = compute_tcs(A, p, q)

        k += 1

    return np.diag(A), U, k, A

#matrice de test 4x4 simetrica (hardcodata)

A_init = np.array([
    [ 4.0,  1.0, -2.0,  2.0],
    [ 1.0,  2.0,  0.0,  1.0],
    [-2.0,  0.0,  3.0, -2.0],
    [ 2.0,  1.0, -2.0, -1.0]
])

print("=" * 55)
print("         METODA JACOBI - Valori si vectori proprii")
print("=" * 55)
print("\nMatricea initiala A_init:")
print(A_init)

eps = 1e-10
k_max = 10000

eigenvalues, U, k, A_final = jacobi(A_init, eps=eps, k_max=k_max)

#sortam descrescator
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
U = U[:, idx]

print(f"\nNumar de iteratii: {k}")
print(f"Precizie folosita (eps): {eps}")

print("\nValori proprii aproximative (lambda_i):")
for i, lam in enumerate(eigenvalues):
    print(f"  lambda_{i+1} = {lam:.10f}")

print("\nMatricea U (vectori proprii pe coloane):")
print(np.round(U, 8))

#verificare: ||A_init * U - U * Lambda||

Lambda = np.diag(eigenvalues)
norma = np.linalg.norm(A_init @ U - U @ Lambda)
print(f"\nVerificare: ||A_init * U - U * Lambda|| = {norma:.2e}")

# Referinta numpy
print("\n--- Referinta numpy.linalg.eigh ---")
np_vals, _ = np.linalg.eigh(A_init)
np_vals = np.sort(np_vals)[::-1]
print("Valori proprii numpy:")
for i, lam in enumerate(np_vals):
    print(f"  lambda_{i+1} = {lam:.10f}")

diff = np.abs(np.sort(eigenvalues)[::-1] - np.sort(np_vals)[::-1])
print(f"\nDiferenta max |Jacobi - numpy|: {np.max(diff):.2e}")

print("\nMatricea A_final (aproximativ diagonala):")
print(np.round(A_final, 8))
print("=" * 55)

###EXERCITIUL 2
#################################

def cholesky_eigen_algorithm(A_init, eps=1e-10, k_max=10000):
    A_k = A_init.copy().astype(float)
    k = 0

    while k < k_max:
        #A^(k) = L * L^T
        try:
            L = np.linalg.cholesky(A_k)
        except np.linalg.LinAlgError:
            print("Eroare: Matricea nu este pozitiv definita! Algoritmul se opreste.")
            return None, k

        #A^(k+1) = L^T * L
        A_next = L.T @ L

        diferenta = np.linalg.norm(A_next - A_k)
        if diferenta < eps:
            A_k = A_next
            k += 1
            break

        A_k = A_next
        k += 1

    return A_k, k

A_init_ex2 = np.array([
    [5.0, 1.0, -1.0, 0.0],
    [1.0, 4.0, 1.0, -1.0],
    [-1.0, 1.0, 6.0, 2.0],
    [0.0, -1.0, 2.0, 7.0]
])

print("\n\n" + "=" * 55)
print("     EXERCITIUL 2 - Metoda Cholesky pentru Valori Proprii")
print("=" * 55)
print("\nMatricea initiala A_init_ex2:")
print(A_init_ex2)

eps_ex2 = 1e-10
A_final_ex2, iteratii_ex2 = cholesky_eigen_algorithm(A_init_ex2, eps=eps_ex2)

if A_final_ex2 is not None:
    print(f"\nAlgoritmul a convergit in {iteratii_ex2} iteratii.")
    print("\nUltima matrice calculata A^(final):")
    print(np.round(A_final_ex2, 6))

    #referinta pentru a verifica rezultatul
    valori_proprii_referinta = np.sort(np.linalg.eigvalsh(A_init_ex2))[::-1]
    print("\nValori proprii reale (din numpy, sortate pentru comparatie):")
    for i, vp in enumerate(valori_proprii_referinta):
        print(f"  lambda_{i + 1} = {vp:.6f}")
print("=" * 55)


###EXERCITIUL 3
#################################

p, n_dim = 5, 3
np.random.seed(42)
A_ex3 = np.random.rand(p, n_dim)

print("\n\n" + "=" * 55)
print("     EXERCITIUL 3 - SVD si Pseudoinversa")
print("=" * 55)
print(f"\nMatricea A_ex3 ({p}x{n_dim}):")
print(np.round(A_ex3, 4))

U, s, Vt = np.linalg.svd(A_ex3, full_matrices=True)

print("\n--- 1. Valorile Singulare ---")
print("s =", np.round(s, 6))

toleranta = np.max(A_ex3.shape) * np.spacing(np.max(s))
rang_calculat = np.sum(s > toleranta)


rang_numpy = np.linalg.matrix_rank(A_ex3)

print("\n--- 2. Rangul Matricei ---")
print(f"Rang calculat (din SVD): {rang_calculat}")
print(f"Rang numpy:              {rang_numpy}")

cond_calculat = np.max(s) / np.min(s)

cond_numpy = np.linalg.cond(A_ex3)

print("\n--- 3. Numarul de Conditionare ---")
print(f"Conditionare calculata: {cond_calculat:.6f}")
print(f"Conditionare numpy:     {cond_numpy:.6f}")

#pseudoinversa Moore-Penrose (A^I)
S_I = np.zeros((n_dim, p))
for i in range(len(s)):
    if s[i] > toleranta:
        S_I[i, i] = 1.0 / s[i]

V = Vt.T
UT = U.T

#A^I = V * S^I * U^T
A_I = V @ S_I @ UT

print("\n--- 4. Pseudoinversa Moore-Penrose (A^I) ---")
print(np.round(A_I, 4))

#A^J = (A^T * A)^-1 * A^T
A_J = np.linalg.inv(A_ex3.T @ A_ex3) @ A_ex3.T

print("\n--- 5. Pseudoinversa celor mai mici patrate (A^J) ---")
print(np.round(A_J, 4))

#|| A^I - A^J ||_1
norma_diff = np.linalg.norm(A_I - A_J, ord=1)

print("\n--- 6. Verificare Norma ||A^I - A^J||_1 ---")
print(f"Norma = {norma_diff:.4e}")

if norma_diff < 1e-10:
    print("Matricele coincid! Pentru matrice full-rank, A^I este matematic egala cu A^J.")
print("=" * 55)
