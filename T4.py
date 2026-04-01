import math
import os

def citeste_vector(nume_fisier):
    with open(nume_fisier, 'r') as f:
        valori = [float(x) for x in f.read().split()]
    return valori


def rezolva_tema(index_sistem, p_precizie=5):
    eps = 10 ** (-p_precizie)

    fisier_d0 = f"d0_{index_sistem}.txt"
    fisier_d1 = f"d1_{index_sistem}.txt"
    fisier_d2 = f"d2_{index_sistem}.txt"
    fisier_b = f"b_{index_sistem}.txt"

    print(f"=== Rezultate pentru sistemul {index_sistem} ===")

    ###EXERCITIUL 1##
    try:
        d0 = citeste_vector(fisier_d0)
        b = citeste_vector(fisier_b)
    except FileNotFoundError:
        print(f"Eroare: Nu s-au gasit fisierele pentru sistemul {index_sistem}.")
        return

    n = len(d0)
    print(f"1. Dimensiunea sistemului (n): {n}")

    ###EXERCITIUL 2##
    d1 = citeste_vector(fisier_d1)
    d2 = citeste_vector(fisier_d2)

    p = n - len(d1)
    q = n - len(d2)

    print(f"2.Ordinul diagonalei superioare (p): {p}")
    print(f"2. Ordinul diagonalei inferioare (q): {q}")

    ###EXERCITIUL 3##
    zero_gasit = False
    for i in range(n):
        if d0[i] == 0.0:
            print(f"3. EROARE: element nul pe diagonala principala la indexul {i}.")
            zero_gasit = True
            break

    if not zero_gasit:
        print("3. Verificare cu succes: Toate elementele din d0 sunt nenule")
    else:
        print("stop: Gauss-Seidel necesita elemente nenule pe diagonala principala")
        return

    ###EXERCITIUL 4##
    k_max = 10000
    x_c = [0.0] * n
    x_p = [0.0] * n
    k = 0

    while True:
        for i in range(n):
            x_p[i] = x_c[i]

        for i in range(n):
            sum_val = 0.0

            if i >= p:
                sum_val += d1[i - p] * x_c[i - p]
            if i >= q:
                sum_val += d2[i - q] * x_c[i - q]

            if i + p < n:
                sum_val += d1[i] * x_p[i + p]
            if i + q < n:
                sum_val += d2[i] * x_p[i + q]

            x_c[i] = (b[i] - sum_val) / d0[i]

        delta_x = 0.0
        for i in range(n):
            err = abs(x_c[i] - x_p[i])
            if err > delta_x:
                delta_x = err

        k += 1

        if not (delta_x >= eps and k <= k_max and delta_x <= 10 ** 10):
            break

    if delta_x < eps:
        print(f"4. Metoda Gauss-Seidel a convers dupa {k} iteratii.")
        x_GS = x_c
    else:
        print("4. EROARE: Divergenta (metoda nu a convers sau a depasit limita).")
        return

    ###EXERCITIUL 5##
    y = [0.0] * n

    for i in range(n):
        y[i] += d0[i] * x_GS[i]

        if i < len(d1):
            y[i] += d1[i] * x_GS[i + p]
            y[i + p] += d1[i] * x_GS[i]

        if i < len(d2):
            y[i] += d2[i] * x_GS[i + q]
            y[i + q] += d2[i] * x_GS[i]

    print("5. Vectorul y = A * x_GS a fost calculat cu succes")

    ###EXERCITIUL 6##
    norma_infinit = 0.0
    for i in range(n):
        diferenta = abs(y[i] - b[i])
        if diferenta > norma_infinit:
            norma_infinit = diferenta

    print(f"6. Norma ||A * x_GS - b||_infinit este: {norma_infinit}")

    return n, p, q, d0, d1, d2, b, x_GS, y

###EX 7...folosim d0,d1,d2

###RULARE PENTRU TOATE CELE 5 SISTEME
###presupunand fisierul pt i=1, ..., 5
for i in range(1, 6):
    rezolva_tema(index_sistem=i, p_precizie=5)
    print("\n" + "=" * 50 + "\n")
