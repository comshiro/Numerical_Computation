import math
import numpy 

import time
import random

#Problema 1
u = 1.0
k = 0
while 1.0 + u != 1.0:
    u = u / 10
    k = k + 1

u = u * 10
k = k - 1

print(f"u = {u}")
print(f"k = {k}")
print(f"1.0 + u = {1.0 + u}")

#Problema 2

x=1.0
y=u/10
z=u/10

left=(x+y)+z 
right=x+(y+z)
print(f"left = {left}")
print(f"right = {right}")
print(f"left == right: {left == right}")
print(f"numpy.isclose(left, right): {numpy.isclose(left, right)}")

x_mul = 1e308  
y_mul = 1e-200
z_mul = 1e-200

left_mul = (x_mul * y_mul) * z_mul
right_mul = x_mul * (y_mul * z_mul)

print(f"\nVerificare neasociativitate pentru înmulțire:")
print(f"x = {x_mul}, y = {y_mul}, z = {z_mul}")
print(f"(x * y) * z = {left_mul}")
print(f"x * (y * z) = {right_mul}")
print(f"(x * y) * z == x * (y * z): {left_mul == right_mul}")
print(f"numpy.isclose((x * y) * z, x * (y * z)): {numpy.isclose(left_mul, right_mul)}")

#Problema 3
#Cerinte:
# Doua metode de aproximare a valorii functiei tangenta: 
# m fractiilor continui si 
# o aproximare polinomiala
#Aplicare pe x apartine -pi/2, pi/2
#in afara intervalului - se foloseste periodicitatea si antisimetria
#valorile multiplu de pi/2 sunt tratate separat
#valoarea obtinuta prin cele 2 valori sa fie comparata cu cea din numpy
#afisati modul de tanx(x)-my_tan(x)


#Generezi 10,000 de numere aleatorii în intervalul (-π/2, π/2)
#Pentru fiecare metodă, calculezi tan(x) pentru toate cele 10,000 de valori
#Compari:
#Eroarea de calcul: |tan(x) - my_tan(x)| pentru fiecare metodă
#Timpul de execuție: cât durează calculul celor 10,000 de valori


#=================Metoda cu polinoame===================

c1 = 1.0/3
c2 = 2.0/15
c3 = 17.0/315
c4 = 62.0/2835

print("Coeficienti pentru aproximarea polinomiala:" )
print(f"c1 = {c1}")
print(f"c2 = {c2}")
print(f"c3 = {c3}")
print(f"c4 = {c4}")

def my_polinomial_tan(x):

    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x6 = x4 * x2
    
    return (x + (c1 + c2 * x2 + c3 * x4 + c4 * x6) * x3)

#Reduce from (-π/2, π/2) to [-π/4, π/4]
def reduce_to_optimal_interval(x) -> (float, bool):
    
    pi_4 = math.pi / 4
    pi_2 = math.pi / 2

    if x < 0:
        x_reduced, needs_reciprocal = reduce_to_optimal_interval(-x)
        return -x_reduced, needs_reciprocal
    
    if x <= pi_4:
        return x, False
    
    if x < pi_2:
        return pi_2 - x, True  # True means we need to take reciprocal later
    
    return x, False



#Normalize any x to (-π/2, π/2)
def normalize_angle(x):

    pi = math.pi
    pi_2 = pi / 2
    
    # Check for special cases: x = k·π/2
    # Use modulo to find remainder
    remainder = x % pi_2
    
    # If remainder is very close to 0 or π/2, it's a special case
    if abs(remainder) < 1e-10 or abs(remainder - pi_2) < 1e-10:
        # Check if it's a multiple of π (k·π where k is integer)
        if abs(x % pi) < 1e-10:
            return 0.0  # tan(k·π) = 0
        else:
            # It's an odd multiple of π/2: ±π/2, ±3π/2, etc.
            return float('inf') if x > 0 else float('-inf')
    
    # Use periodicity: tan has period π
    # Reduce x to range [0, π)
    normalized = x % pi
    
    # If in [π/2, π), shift to (-π/2, 0) by subtracting π
    if normalized > pi_2:
        normalized -= pi
    
    return normalized


# 4. Main function - applies everything in sequence
def my_tan(x):
    """
    Main entry point - handles any x value
    """
    # Step 1: Normalize to (-π/2, π/2)
    x_normalized = normalize_angle(x)
    
    # Step 2: Handle special cases (±∞)
    if abs(x_normalized) == float('inf'):
        return x_normalized
    
    # Step 3: Reduce to [-π/4, π/4]
    x_reduced, needs_reciprocal = reduce_to_optimal_interval(x_normalized)
    
    # Step 4: Calculate using basic polynomial
    result = my_polinomial_tan(x_reduced)
    
    # Step 5: Apply reciprocal if needed
    if needs_reciprocal:
        return 1.0 / result
    
    return result


#=================Metoda cu fracții continue===================

def my_continued_fraction_tan_basic(x, epsilon=1e-10):
    """
    Calculează tan(x) folosind metoda fracțiilor continue (Lentz modificat)
    Funcționează pentru x în [-π/4, π/4]
    
    tan(x) = x / (1 + (-x²)/(3 + (-x²)/(5 + (-x²)/(7 + ...))))
    
    Pentru fracția continuă:
    a_j = -x² pentru j >= 1
    b_0 = 0
    b_j = 2j - 1 pentru j >= 1 (adică: 1, 3, 5, 7, ...)
    
    Args:
        x: argumentul funcției tan, ar trebui să fie în [-π/4, π/4]
        epsilon: precizia aproximării (default 1e-10)
    
    Returns:
        aproximarea lui tan(x)
    """
    mic = 1e-12  # Valoare foarte mică pentru a evita împărțirea la 0
    
    # Inițializare
    b0 = 0
    f = b0 if b0 != 0 else mic  # f_0 = b_0
    C = f  # C_0 = b_0
    D = 0  # D_0 = 0
    
    x_squared = x * x  # x²
    j = 1
    
    # Iterare până când converge
    while True:
        # Calculăm a_j și b_j pentru iterația curentă
        if j == 1:
            a_j = x  # a_1 = x (numărător special pentru prima iterație)
            b_j = 1  # b_1 = 1
        else:
            a_j = -x_squared  # a_j = -x² pentru j >= 2
            b_j = 2 * j - 1   # b_j = 2j - 1 (1, 3, 5, 7, ...)
        
        # Algoritmul Lentz modificat
        D = b_j + a_j * D
        if D == 0:
            D = mic
        
        C = b_j + a_j / C
        if C == 0:
            C = mic
        
        D = 1.0 / D
        delta = C * D
        f = delta * f
        
        # Verificăm convergența
        if abs(delta - 1.0) < epsilon:
            break
        
        j += 1
        
        # Safeguard: limită de iterații pentru a evita loop-uri infinite
        if j > 1000:
            break
    
    return f


def my_continued_fraction_tan(x, epsilon=1e-10):
    """
    Calculează tan(x) pentru orice x folosind metoda fracțiilor continue.
    Aplică normalizare și reducere la interval optim.
    
    Args:
        x: argumentul funcției tan
        epsilon: precizia aproximării
    
    Returns:
        aproximarea lui tan(x)
    """
    # Step 1: Normalize to (-π/2, π/2)
    x_normalized = normalize_angle(x)
    
    # Step 2: Handle special cases (±∞)
    if abs(x_normalized) == float('inf'):
        return x_normalized
    
    # Step 3: Reduce to [-π/4, π/4]
    x_reduced, needs_reciprocal = reduce_to_optimal_interval(x_normalized)
    
    # Step 4: Calculate using continued fractions
    result = my_continued_fraction_tan_basic(x_reduced, epsilon)
    
    # Step 5: Apply reciprocal if needed
    if needs_reciprocal:
        return 1.0 / result
    
    return result


# ============================================================
# TESTARE SI BENCHMARK pt problema 3
# ============================================================

# Generare 10.000 de numere aleatoare in (-pi/2, pi/2)
# Folosim math.pi pentru limite, dar numpy.tan pentru rezultate
random.seed(42)
pi_2 = math.pi / 2
test_values = [random.uniform(-pi_2 + 1e-9, pi_2 - 1e-9) for _ in range(10000)]

# ---- Metoda polinomiala ----
start_poly = time.perf_counter()
results_poly = [my_tan(x) for x in test_values]
end_poly = time.perf_counter()
time_poly = end_poly - start_poly

# ---- Metoda fractii continue ----
start_cf = time.perf_counter()
results_cf = [my_continued_fraction_tan(x) for x in test_values]
end_cf = time.perf_counter()
time_cf = end_cf - start_cf

# ---- Valori de referinta folosind NUMPY ----
ref_values = [numpy.tan(x) for x in test_values]

# ---- Calculul erorilor ----
errors_poly = [abs(ref - approx) for ref, approx in zip(ref_values, results_poly)]
errors_cf   = [abs(ref - approx) for ref, approx in zip(ref_values, results_cf)]

# ---- Afisare rezultate ----
print("\n" + "="*55)
print("           COMPARATIE METODE - tan(x)")
print("="*55)

print(f"\n{'Metrica':<30} {'Polinomiala':>12} {'Fract. Continue':>15}")
print("-"*55)
print(f"{'Eroare medie':<30} {sum(errors_poly)/len(errors_poly):>12.2e} {sum(errors_cf)/len(errors_cf):>15.2e}")
print(f"{'Eroare maxima':<30} {max(errors_poly):>12.2e} {max(errors_cf):>15.2e}")
print(f"{'Eroare minima':<30} {min(errors_poly):>12.2e} {min(errors_cf):>15.2e}")
print(f"{'Timp executie (s)':<30} {time_poly:>12.4f} {time_cf:>15.4f}")
print("="*55)

# ---- Afisare primele 5 valori ca exemplu ----
print("\nExemple (primele 5 valori):")
print(f"{'x':>12} {'numpy tan':>15} {'Polinomial':>15} {'Fract.Continue':>15} {'Err Poly':>12} {'Err CF':>12}")
print("-"*85)
for i in range(5):
    x = test_values[i]
    print(f"{x:>12.6f} {ref_values[i]:>15.10f} {results_poly[i]:>15.10f} {results_cf[i]:>15.10f} {errors_poly[i]:>12.2e} {errors_cf[i]:>12.2e}")
