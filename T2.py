import numpy as np
import scipy.linalg as la

def rezolva_tema_calcul_numeric():
    n = 150
    print(f"--- Initializare sistem (n = {n}) ---")

    B = np.random.rand(n, n)
    A = np.dot(B, B.T)
    b = np.random.rand(n)
    A_init = A.copy()

    ### SUBPUNCTUL 1 ###
    print("\n1. Descompunere LU si solutie x_lib...")
    P, L_lib, U_lib = la.lu(A_init)
    print(f"   LU calculat. L shape: {L_lib.shape}, U shape: {U_lib.shape}")
    x_lib = np.linalg.solve(A_init, b)

    ### SUBPUNCTUL 2 ###
    print("2. Descompunere LDL^T in-place...")
    d = np.zeros(n)

    for j in range(n):
        # d[j] = A[j,j] - sum_{k<j} L[j,k]^2 * d[k]
        sum_d = 0.0
        for k in range(j):
            sum_d += A[j, k] * A[j, k] * d[k]
        d[j] = A[j, j] - sum_d

        # L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]*d[k]) / d[j]
        for i in range(j + 1, n):
            sum_l = 0.0
            for k in range(j):
                sum_l += A[i, k] * A[j, k] * d[k]
            A[i, j] = (A[i, j] - sum_l) / d[j]

    ### SUBPUNCTUL 3 ###
    det_A = np.prod(d)
    print(f"3. Determinant eficient: {det_A:.6e}")
    # Verificare
    sign, logdet = np.linalg.slogdet(A_init)
    det_numpy = sign * np.exp(logdet)
    print(f"   Determinant numpy (referinta): {det_numpy:.6e}")

    ### SUBPUNCTUL 4 ###
    print("4. Calcul x_Chol prin substitutie...")

    # Pas 1: Lz = b (substitutie directa, L_ii = 1)
    z = np.zeros(n)
    for i in range(n):
        sum_lz = 0.0
        for j in range(i):
            sum_lz += A[i, j] * z[j]
        z[i] = b[i] - sum_lz

    # Pas 2: Dy = z
    y = np.zeros(n)
    for i in range(n):
        y[i] = z[i] / d[i]

    # Pas 3: L^T x = y (substitutie inversa)
    x_Chol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_lx = 0.0
        for j in range(i + 1, n):
            sum_lx += A[j, i] * x_Chol[j]  # L^T[i,j] = L[j,i] = A[j,i]
        x_Chol[i] = y[i] - sum_lx

    ### SUBPUNCTUL 5 ###
    print("5. Verificare norme...")

    Ax_calc = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i <= j:
                val_A_init = A[i, j]
            else:
                val_A_init = A[j, i]

            Ax_calc[i] += val_A_init * x_Chol[j]

    norma1 = np.linalg.norm(Ax_calc - b, ord=2)
    norma2 = np.linalg.norm(x_Chol - x_lib, ord=2)

    print(f"   ||A^init * x_Chol - b||_2 = {norma1:.4e}")
    print(f"   ||x_Chol - x_lib||_2      = {norma2:.4e}")

    if norma1 < 1e-8 and norma2 < 1e-7:
        print("\n=> SUCCES! your did it.")
    else:
        print("\n=> EROARE: Normele sunt prea mari.")


if __name__ == "__main__":
    rezolva_tema_calcul_numeric()
