import os
import numpy as np

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = input("Ingrese el grado de ecuacion que desea usar")
Matrix = np.empty((8,5,))

with open('datos.txt') as file:
    for line in file:
        

        print (Matrix)
