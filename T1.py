import math
import numpy 

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


