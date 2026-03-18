import numpy as np
import math

##EXERCITIUL 1##
def calculeaza_vector_b(A, s, n):
    b = np.zeros(n)
    for i in range(n):
        suma = 0
        for j in range(n):
            suma += s[j] * A[i][j]
        b[i] = suma
    return b

##EXERCITIUL 2##
def descompunere_qr_householder(A):
    n = A.shape[0]
    R = A.copy().astype(float)
    Q_tilde = np.eye(n)
    u = np.zeros(n)
    epsilon = 1e-12

    for r in range(n - 1):
        # sigma = suma(a_ir^2)
        sigma = 0.0
        for i in range(r, n):
            sigma += R[i, r] ** 2

        if sigma <= epsilon:
            break

        # k = sqrt(sigma)
        k = math.sqrt(sigma)
        if R[r, r] > 0:
            k = -k

        # beta = sigma - k * a_rr
        beta = sigma - k * R[r, r]

        # u_r = a_rr - k; u_i = a_ir, i = r+1...n
        u[r] = R[r, r] - k
        for i in range(r + 1, n):
            u[i] = R[i, r]

        for j in range(r + 1, n):
            # gamma = (Ae_j, u) / beta
            suma_A = 0.0
            for i in range(r, n):
                suma_A += u[i] * R[i, j]
            gamma = suma_A / beta

            for i in range(r, n):
                R[i, j] = R[i, j] - gamma * u[i]

        R[r, r] = k
        for i in range(r + 1, n):
            R[i, r] = 0.0

        for j in range(n):
            suma_Q = 0.0
            for i in range(r, n):
                suma_Q += u[i] * Q_tilde[i, j]
            gamma = suma_Q / beta

            for i in range(r, n):
                Q_tilde[i, j] = Q_tilde[i, j] - gamma * u[i]
    Q = Q_tilde.T

    return Q, R

##EXERCITIUL 3##
def substitutie_inapoi(R, y):
    n = R.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        suma_cunoscuta = 0
        for j in range(i + 1, n):
            suma_cunoscuta += R[i, j] * x[j]

        if R[i, i] == 0:
            raise ValueError("Matricea R are un zero pe diagonala. Sistemul nu are solutie unica.")
        x[i] = (y[i] - suma_cunoscuta) / R[i, i]

    return x


def rezolva_sistem_qr(Q, R, b):
    y = Q.T @ b
    return substitutie_inapoi(R, y)


def exercitiul_3(A, b):
    Q_np, R_np = np.linalg.qr(A)
    x_QR = rezolva_sistem_qr(Q_np, R_np, b)

    Q_h, R_h = descompunere_qr_householder(A)
    x_Householder = rezolva_sistem_qr(Q_h, R_h, b)
    norma_diferentei = np.linalg.norm(x_QR - x_Householder, ord=2)

    return x_QR, x_Householder, norma_diferentei

##EXERCITIUL 4##
def exercitiul_4(A_init, b_init, x_householder, x_QR, s):
    norm_s = np.linalg.norm(s, ord=2)

    err_rezidual_house = np.linalg.norm(A_init @ x_householder - b_init, ord=2)
    err_rezidual_qr = np.linalg.norm(A_init @ x_QR - b_init, ord=2)

    err_relativ_house = np.linalg.norm(x_householder - s, ord=2) / norm_s
    err_relativ_qr = np.linalg.norm(x_QR - s, ord=2) / norm_s

    return err_rezidual_house, err_rezidual_qr, err_relativ_house, err_relativ_qr

##EXERCITIUL 5##
def calculeaza_inversa_qr(Q, R):
    n = R.shape[0]
    A_inv = np.zeros((n, n))
    I = np.eye(n)

    for i in range(n):
        e_i = I[:, i]
        coloana_inversa = rezolva_sistem_qr(Q, R, e_i)
        A_inv[:, i] = coloana_inversa

    return A_inv


def exercitiul_5(A, Q, R):
    A_inv_householder = calculeaza_inversa_qr(Q, R)
    A_inv_bibl = np.linalg.inv(A)

    norma_diferentei = np.linalg.norm(A_inv_householder - A_inv_bibl)

    return A_inv_householder, A_inv_bibl, norma_diferentei

##EXERCITIUL 6##
def exercitiul_6_test_random(n):
    print(f"\n{'=' * 50}")
    print(f"--- testam pentru dimensiunea n = {n} ---")
    print(f"{'=' * 50}")

    A_rand = np.random.uniform(-10, 10, (n, n))
    s_rand = np.random.uniform(-10, 10, n)

    b_rand = calculeaza_vector_b(A_rand, s_rand, n)

    Q_h, R_h = descompunere_qr_householder(A_rand)
    x_QR, x_Householder, eroare_ex3 = exercitiul_3(A_rand, b_rand)

    e_rez_h, e_rez_qr, e_rel_h, e_rel_qr = exercitiul_4(A_rand, b_rand, x_Householder, x_QR, s_rand)

    A_inv_H, A_inv_np, eroare_inv = exercitiul_5(A_rand, Q_h, R_h)

    print("\nRezultatele(erori obtinute):")
    print(f"Eroare reziduala Householder: {e_rez_h:.2e}")
    print(f"Eroare reziduala NumPy:       {e_rez_qr:.2e}")
    print(f"Eroare relativa Householder:  {e_rel_h:.2e}")
    print(f"Eroare relativa NumPy:        {e_rel_qr:.2e}")
    print(f"Eroare la calculul inversei:  {eroare_inv:.2e}")

    limita = 10 ** (-6)
    toate_erorile = [e_rez_h, e_rez_qr, e_rel_h, e_rel_qr, eroare_inv]

    if all(eroare < limita for eroare in toate_erorile):
        print(f"\nV Toate erorile pentru n={n} au ramas sub 10^-6.")
    else:
        print(f"\npentru n={n}, limitarile procesorului incep sa se simta.")

if __name__ == "__main__":
    exercitiul_6_test_random(5)
    exercitiul_6_test_random(100)
    
