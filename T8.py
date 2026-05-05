import numpy as np

###EX 1
def aproximare_gradient(F, x, h=1e-5):
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        x_plus_2h = x.copy()
        x_plus_2h[i] += 2 * h
        F1 = F(x_plus_2h)

        x_plus_h = x.copy()
        x_plus_h[i] += h
        F2 = F(x_plus_h)

        x_minus_h = x.copy()
        x_minus_h[i] -= h
        F3 = F(x_minus_h)

        x_minus_2h = x.copy()
        x_minus_2h[i] -= 2 * h
        F4 = F(x_minus_2h)

        grad[i] = (-F1 + 8 * F2 - 8 * F3 + F4) / (12 * h)

    return grad

###EX 2
def backtracking_line_search(F, x, grad_x, beta=0.8):
    eta = 1.0
    p = 1
    norm_grad_sq = np.linalg.norm(grad_x) ** 2

    while F(x - eta * grad_x) > F(x) - (eta / 2) * norm_grad_sq and p < 8:
        eta = eta * beta
        p += 1

    return eta

###EX 3
def gradient_descendent(F, grad_F_analitic, x0, metoda_rata='constant', metoda_grad='analitic', epsilon=1e-5,
                        k_max=30000):
    x = np.array(x0, dtype=float)
    k = 0

    eta_constanta = 1e-3

    while True:
        if metoda_grad == 'analitic':
            grad = grad_F_analitic(x)
        else:
            grad = aproximare_gradient(F, x)

        norma_grad = np.linalg.norm(grad)

        if metoda_rata == 'backtracking':
            eta = backtracking_line_search(F, x, grad)
        else:
            eta = eta_constanta

        x_urmator = x - eta * grad
        k += 1

        conditie_continuare = (eta * norma_grad >= epsilon) and (k <= k_max) and (eta * norma_grad <= 10 ** 10)

        x = x_urmator

        if not conditie_continuare:
            break

    if eta * norma_grad <= epsilon:
        return x, k, "succes"
    else:
        return x, k, "divergenta"

###EX 4
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

#exemplul 1
def l_func(w):
    w0, w1 = w[0], w[1]
    eps_val = 1e-15
    return -np.log(1 - sigmoid(w0 - w1) + eps_val) - np.log(sigmoid(w0 + w1) + eps_val)

def grad_l_func(w):
    w0, w1 = w[0], w[1]
    g1 = sigmoid(w0 - w1) + sigmoid(w0 + w1) - 1
    g2 = sigmoid(w0 + w1) - sigmoid(w0 - w1) - 1
    return np.array([g1, g2])

#exemplul 2
def F2(x):
    return x[0] ** 2 + x[1] ** 2 - 2 * x[0] - 4 * x[1] - 1

def grad_F2(x):
    return np.array([2 * x[0] - 2, 2 * x[1] - 4])

#exemplul 3
def F3(x):
    return 3 * x[0] ** 2 - 12 * x[0] + 2 * x[1] ** 2 + 16 * x[1] - 10

def grad_F3(x):
    return np.array([6 * x[0] - 12, 4 * x[1] + 16])

#exemplul 4
def F4(x):
    return x[0] ** 2 - 4 * x[0] * x[1] + 4.5 * x[1] ** 2 - 4 * x[1] + 3

def grad_F4(x):
    return np.array([2 * x[0] - 4 * x[1], -4 * x[0] + 9 * x[1] - 4])

#exemplul 5
def F5(x):
    return x[0] ** 2 * x[1] - 2 * x[0] * x[1] ** 2 + 3 * x[0] * x[1] + 4

def grad_F5(x):
    return np.array([2 * x[0] * x[1] - 2 * x[1] ** 2 + 3 * x[1], x[0] ** 2 - 4 * x[0] * x[1] + 3 * x[0]])

###TESTARE
if __name__ == "__main__":
    functii_test = [
        {"nume": "l(w0, w1)", "F": l_func, "grad": grad_l_func, "x_start": [0.0, 0.0]},
        {"nume": "F2(x1, x2)", "F": F2, "grad": grad_F2, "x_start": [0.0, 0.0]},
        {"nume": "F3(x1, x2)", "F": F3, "grad": grad_F3, "x_start": [0.0, 0.0]},
        {"nume": "F4(x1, x2)", "F": F4, "grad": grad_F4, "x_start": [0.0, 0.0]},
        {"nume": "F5(x1, x2)", "F": F5, "grad": grad_F5, "x_start": [1.0, 1.0]}
    ]

    for test in functii_test:
        print(f"\n--- Testare functie: {test['nume']} ---")
        x0 = test['x_start']

        pentru_comparatie = [
            ('constant', 'analitic'),
            ('constant', 'aproximativ'),
            ('backtracking', 'analitic'),
            ('backtracking', 'aproximativ')
        ]

        for met_rata, met_grad in pentru_comparatie:
            sol, iteratii, status = gradient_descendent(
                F=test['F'],
                grad_F_analitic=test['grad'],
                x0=x0,
                metoda_rata=met_rata,
                metoda_grad=met_grad
            )

            print(
                f"Rata: {met_rata:12} | Grad: {met_grad:11} | Iter: {iteratii:5} | Stare: {status:10} | Sol: [{sol[0]:.4f}, {sol[1]:.4f}]")
