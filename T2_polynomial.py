import math
import random
import time

# Coeficienți precalculați pentru polinomul MacLaurin
c1 = 0.33333333333333333
c2 = 0.133333333333333333
c3 = 0.053968253968254
c4 = 0.0218694885361552


def my_tan_polynomial(x):
    """
    Aproximarea funcției tangentă folosind seria MacLaurin.
    tan(x) ≈ x + (1/3)x³ + (2/15)x⁵ + (17/315)x⁷ + (62/2835)x⁹
    
    Funcționează optim pentru x ∈ (-π/4, π/4)
    """
    # Reducere la intervalul [-π/4, π/4]
    reduced_x, needs_reciprocal = reduce_to_optimal_interval(x)
    
    # Calcul puteri ale lui x
    x_2 = reduced_x * reduced_x
    x_3 = x_2 * reduced_x
    x_4 = x_2 * x_2
    x_6 = x_4 * x_2
    x_8 = x_4 * x_4
    
    # Calcul polinom: x + c1*x³ + c2*x⁵ + c3*x⁷ + c4*x⁹
    result = reduced_x + c1 * x_3 + c2 * x_6 * reduced_x + c3 * x_6 * x_3 + c4 * x_8 * reduced_x
    
    # Dacă am folosit relația tan(x) = 1/tan(π/2 - x), returnăm reciproca
    if needs_reciprocal:
        return 1.0 / result
    
    return result


def reduce_to_optimal_interval(x):
    """
    Reduce argumentul x din (-π/2, π/2) în intervalul [-π/4, π/4]
    folosind antisimetria și relația: tan(x) = 1/tan(π/2 - x)
    
    Returns:
        (reduced_x, needs_reciprocal): x redus și flag dacă trebuie reciproca
    """
    pi_4 = math.pi / 4
    pi_2 = math.pi / 2
    
    # Gestionare antisimetrie: tan(-x) = -tan(x)
    if x < 0:
        reduced, needs_recip = reduce_to_optimal_interval(-x)
        return -reduced, needs_recip
    
    # x ∈ [0, π/4] - interval optim, nu e nevoie de reducere
    if x <= pi_4:
        return x, False
    
    # x ∈ (π/4, π/2) - folosim tan(x) = 1/tan(π/2 - x)
    if x < pi_2:
        return pi_2 - x, True
    
    # x = π/2 - caz special (tangenta -> infinit)
    return x, False


def normalize_argument(x):
    """
    Normalizează argumentul x din orice valoare în intervalul (-π/2, π/2)
    folosind periodicitatea: tan(x + π) = tan(x)
    """
    pi = math.pi
    
    # Tratare cazuri speciale: x = k*π/2
    if abs(x % (pi / 2)) < 1e-10:
        return float('inf') if x % pi != 0 else 0.0
    
    # Reducere la (-π/2, π/2) folosind periodicitatea
    # tan(x) are perioadă π
    reduced = x % pi
    
    # Aducem în intervalul (-π/2, π/2)
    if reduced > pi / 2:
        reduced -= pi
    elif reduced < -pi / 2:
        reduced += pi
    
    return reduced


def my_tan_full(x):
    """
    Calculează tan(x) pentru orice valoare a lui x,
    gestionând periodicitatea și cazurile speciale.
    """
    # Normalizare la (-π/2, π/2)
    normalized_x = normalize_argument(x)
    
    # Cazuri speciale
    if abs(normalized_x) == float('inf') or abs(normalized_x) >= math.pi / 2:
        return float('inf')
    
    # Calcul folosind aproximarea polinomială
    return my_tan_polynomial(normalized_x)


def test_individual_values():
    """
    Testează metoda polinomială pentru câteva valori individuale.
    """
    print("=" * 70)
    print("TEST VALORI INDIVIDUALE - Metoda Polinomială")
    print("=" * 70)
    
    test_values = [0, math.pi/6, math.pi/4, math.pi/3, -math.pi/6, -math.pi/4, 
                   0.5, 1.0, -0.5, -1.0]
    
    print(f"{'x':<15} {'math.tan(x)':<20} {'my_tan(x)':<20} {'Eroare':<15}")
    print("-" * 70)
    
    for x in test_values:
        true_value = math.tan(x)
        approx_value = my_tan_full(x)
        error = abs(true_value - approx_value)
        
        print(f"{x:<15.6f} {true_value:<20.15f} {approx_value:<20.15f} {error:<15.2e}")


def test_10000_values():
    """
    Generează 10,000 de valori aleatorii în (-π/2, π/2) și compară:
    - Eroarea de calcul
    - Timpul de execuție
    """
    print("\n" + "=" * 70)
    print("TEST 10,000 VALORI ALEATORII - Metoda Polinomială")
    print("=" * 70)
    
    n = 10000
    pi_2 = math.pi / 2
    
    # Generare valori aleatorii în (-π/2, π/2)
    random.seed(42)  # Pentru reproducibilitate
    x_values = [random.uniform(-pi_2 + 0.01, pi_2 - 0.01) for _ in range(n)]
    
    # Calcul folosind math.tan
    start_time = time.time()
    true_values = [math.tan(x) for x in x_values]
    math_time = time.time() - start_time
    
    # Calcul folosind my_tan_polynomial
    start_time = time.time()
    approx_values = [my_tan_full(x) for x in x_values]
    my_time = time.time() - start_time
    
    # Calcul erori
    errors = [abs(true - approx) for true, approx in zip(true_values, approx_values)]
    
    # Statistici
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)
    min_error = min(errors)
    
    print(f"\nStatistici Erori:")
    print(f"  Eroare minimă:  {min_error:.2e}")
    print(f"  Eroare medie:   {avg_error:.2e}")
    print(f"  Eroare maximă:  {max_error:.2e}")
    
    print(f"\nTimpi de Execuție:")
    print(f"  math.tan():     {math_time:.6f} secunde")
    print(f"  my_tan():       {my_time:.6f} secunde")
    print(f"  Raport:         {my_time/math_time:.2f}x")


if __name__ == "__main__":
    test_individual_values()
    test_10000_values()
