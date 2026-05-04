

import math
import os

EPSILON = 1e-9      
KMAX    = 1000      
DIV_LIMIT = 1e8


def horner(coeffs, v):
   
    n = len(coeffs) - 1  

    # -- Pasul 1: sirul b pentru P(v) --
    b = coeffs[0]     
    b_vals = [coeffs[0]]
    for i in range(1, n + 1):
        b = coeffs[i] + b * v
        b_vals.append(b)
    Pv = b_vals[n]

    # -- Pasul 2: sirul c pentru P'(v) --
    # Aplicam Horner pe b_vals[0..n-1]
    if n >= 1:
        c = b_vals[0]
        c_vals = [b_vals[0]]
        for i in range(1, n):
            c = b_vals[i] + c * v
            c_vals.append(c)
        dPv = c_vals[n - 1]
    else:
        dPv = 0.0
        c_vals = []

    # -- Pasul 3: sirul d pentru P''(v) --
    # Aplicam Horner pe c_vals[0..n-2]
    if n >= 2:
        d = c_vals[0]
        for i in range(1, n - 1):
            d = c_vals[i] + d * v
        d2Pv = 2.0 * d        
    else:
        d2Pv = 0.0

    return Pv, dPv, d2Pv


def compute_R(coeffs):
    a0 = coeffs[0]
    A  = max(abs(c) for c in coeffs[1:])
    R  = (abs(a0) + A) / abs(a0)
    return R


def newton(coeffs, x0, eps=EPSILON, kmax=KMAX):
   
    x = x0
    for k in range(1, kmax + 1):
        Pv, dPv, _ = horner(coeffs, x)

        if abs(dPv) <= eps:
            return x, k, False   # impartire la zero

        delta_x = Pv / dPv 
        x = x - delta_x

        if abs(delta_x) < eps:
            return x, k, True

        if abs(delta_x) > DIV_LIMIT:
            return x, k, False

    return x, kmax, False


def olver(coeffs, x0, eps=EPSILON, kmax=KMAX):
    x = x0
    for k in range(1, kmax + 1):
        Pv, dPv, d2Pv = horner(coeffs, x)

        if abs(dPv) <= eps:
            return x, k, False

        c_k      = (Pv ** 2 * d2Pv) / (dPv ** 3)
        delta_x  = Pv / dPv + 0.5 * c_k
        x        = x - delta_x

        if abs(delta_x) < eps:
            return x, k, True

        if abs(delta_x) > DIV_LIMIT:
            return x, k, False

    return x, kmax, False

def is_new_root(root, existing_roots, eps=EPSILON):
    for r in existing_roots:
        if abs(root - r) <= eps:
            return False
    return True


def is_valid_root(coeffs, x, eps=EPSILON):
    Pv, _, _ = horner(coeffs, x)
    return abs(Pv) < math.sqrt(eps)   


def find_roots(coeffs, num_x0=60, eps=1e-4):
    
    R = compute_R(coeffs)
    print(f"\n  Interval radacini: [-{R:.6f}, {R:.6f}]")

    x0_list = [R * (-1 + 2 * i / (num_x0 - 1)) for i in range(num_x0)]

    results = []

    all_results_newton = {}   
    all_results_olver  = {} 

    found_roots = []      

    for x0 in x0_list:

        root_n, iters_n, conv_n = newton(coeffs, x0, eps)
        root_o, iters_o, conv_o = olver(coeffs, x0, eps)

        if conv_n and is_valid_root(coeffs, root_n, eps):
            if is_new_root(root_n, found_roots, eps):
                found_roots.append(root_n)
                results.append({
                    'root'       : root_n,
                    'iters_newton': iters_n,
                    'iters_olver' : None,   
                    'found_by'   : 'Newton'
                })
            key = round(root_n, 8)
            if key not in all_results_newton or all_results_newton[key][1] > iters_n:
                all_results_newton[key] = (root_n, iters_n)

        if conv_o and is_valid_root(coeffs, root_o, eps):
            if is_new_root(root_o, found_roots, eps):
                found_roots.append(root_o)
                results.append({
                    'root'       : root_o,
                    'iters_newton': None,
                    'iters_olver' : iters_o,
                    'found_by'   : 'Olver'
                })
            key = round(root_o, 8)
            if key not in all_results_olver or all_results_olver[key][1] > iters_o:
                all_results_olver[key] = (root_o, iters_o)

    for entry in results:
        r = entry['root']
        key = round(r, 8)

        if entry['iters_newton'] is None:
            if key in all_results_newton:
                entry['iters_newton'] = all_results_newton[key][1]

        if entry['iters_olver'] is None:
            if key in all_results_olver:
                entry['iters_olver'] = all_results_olver[key][1]

    results.sort(key=lambda e: e['root'])

    return results, R


def print_polynomial(coeffs):
    n = len(coeffs) - 1
    terms = []
    for i, a in enumerate(coeffs):
        if abs(a) < 1e-15:
            continue
        exp = n - i
        coef_str = f"{a:+.4g}"
        if exp == 0:
            terms.append(coef_str)
        elif exp == 1:
            terms.append(f"{coef_str}x")
        else:
            terms.append(f"{coef_str}x^{exp}")
    poly_str = " ".join(terms).replace("+-", "- ").strip()
    if poly_str.startswith("+"):
        poly_str = poly_str[1:].strip()
    return poly_str


def display_results(results, coeffs, R):
    print("\n" + "═" * 68)
    print("  POLINOM:", print_polynomial(coeffs))
    print("  Interval [-R, R]: R =", f"{R:.6f}")
    print("═" * 68)

    if not results:
        print("  Nu s-au gasit radacini reale.")
        return

    print(f"\n  {'Radacina':>18}  {'P(radacina)':>14}  {'Newton iters':>13}  {'Olver iters':>12}")
    print("  " + "─" * 64)

    for entry in results:
        r = entry['root']
        Pv, _, _ = horner(coeffs, r)
        n_it = entry['iters_newton'] if entry['iters_newton'] else "—"
        o_it = entry['iters_olver']  if entry['iters_olver']  else "—"
        print(f"  {r:>18.10f}  {Pv:>14.2e}  {str(n_it):>13}  {str(o_it):>12}")

    print("  " + "─" * 64)
    print(f"\n  Total radacini distincte gasite: {len(results)}")

    # Tabel comparativ Newton vs Olver
    print("\n" + "═" * 68)
    print("  COMPARATIE Newton vs Olver (numar de iteratii)")
    print("═" * 68)
    print(f"  {'Radacina':>18}  {'Newton':>10}  {'Olver':>10}  {'Diferenta':>12}")
    print("  " + "─" * 56)

    for entry in results:
        r   = entry['root']
        n_i = entry['iters_newton']
        o_i = entry['iters_olver']
        if n_i is not None and o_i is not None:
            diff = n_i - o_i
            diff_str = f"{diff:+d}" if diff != 0 else "egal"
        else:
            diff_str = "—"
        n_str = str(n_i) if n_i else "—"
        o_str = str(o_i) if o_i else "—"
        print(f"  {r:>18.10f}  {n_str:>10}  {o_str:>10}  {diff_str:>12}")

    print("  " + "─" * 56)
    print()


def save_to_file(results, coeffs, R, filename="radacini_distincte.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  TEMA 7 - Radacinile Polinomului\n")
        f.write("=" * 60 + "\n")
        f.write(f"  P(x) = {print_polynomial(coeffs)}\n")
        f.write(f"  Grad: {len(coeffs)-1}\n")
        f.write(f"  Precizie (epsilon): {EPSILON}\n")
        f.write(f"  Interval: [-{R:.6f}, {R:.6f}]\n\n")

        f.write("  Radacini distincte gasite:\n")
        f.write("  " + "-" * 40 + "\n")

        for i, entry in enumerate(results, 1):
            r   = entry['root']
            Pv, _, _ = horner(coeffs, r)
            n_i = entry['iters_newton']
            o_i = entry['iters_olver']
            f.write(f"  [{i}] x* = {r:.12f}\n")
            f.write(f"       P(x*) = {Pv:.4e}\n")
            if n_i:
                f.write(f"       Iteratii Newton : {n_i}\n")
            if o_i:
                f.write(f"       Iteratii Olver  : {o_i}\n")
            f.write("\n")

        f.write(f"  Total radacini: {len(results)}\n")

    print(f"  [✓] Radacinile au fost salvate in '{filename}'")


def run_for_polynomial(name, coeffs, filename):
    print("\n" + "█" * 68)
    print(f"  {name}")
    print("█" * 68)

    results, R = find_roots(coeffs)
    display_results(results, coeffs, R)
    save_to_file(results, coeffs, R, filename)
    return results


if __name__ == "__main__":

    # ─── Polinom 1 din PDF ───────────────────────────────────────────────────
    # P(x) = (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    # Radacini exacte: 1, 2, 3
    p1_coeffs = [1.0, -6.0, 11.0, -6.0]
    run_for_polynomial(
        "P1(x) = x^3 - 6x^2 + 11x - 6   [radacini: 1, 2, 3]",
        p1_coeffs,
        "radacini_P1.txt"
    )

    # ─── Polinom 2 din PDF ───────────────────────────────────────────────────
    # P(x) = (1/42)(42x^4 - 55x^3 - 42x^2 + 49x - 6)
    # Radacini exacte: 2/3, 1/7, -1, 3/2
    p2_coeffs = [42.0, -55.0, -42.0, 49.0, -6.0]
    run_for_polynomial(
        "P2(x) = 42x^4 - 55x^3 - 42x^2 + 49x - 6   [radacini: -1, 1/7, 2/3, 3/2]",
        p2_coeffs,
        "radacini_P2.txt"
    )

    # ─── Polinom 3 din PDF ───────────────────────────────────────────────────
    # P(x) = (1/8)(8x^4 - 38x^3 + 49x^2 - 22x + 3)
    # Radacini exacte: 1/4, 1/2, 1, 3
    p3_coeffs = [8.0, -38.0, 49.0, -22.0, 3.0]
    run_for_polynomial(
        "P3(x) = 8x^4 - 38x^3 + 49x^2 - 22x + 3   [radacini: 1/4, 1/2, 1, 3]",
        p3_coeffs,
        "radacini_P3.txt"
    )

    # ─── Polinom 4 din PDF ───────────────────────────────────────────────────
    # P(x) = (x-1)^2(x-2)^2 = x^4 - 6x^3 + 13x^2 - 12x + 4
    # Radacini: 1 (multiplicitate 2), 2 (multiplicitate 2)
    p4_coeffs = [1.0, -6.0, 13.0, -12.0, 4.0]
    run_for_polynomial(
        "P4(x) = x^4 - 6x^3 + 13x^2 - 12x + 4   [radacini: 1 (x2), 2 (x2)]",
        p4_coeffs,
        "radacini_P4.txt"
    )

    print("\n" + "═" * 68)
    print("  Executie completa. Fisierele de output au fost create.")
    print("═" * 68 + "\n")