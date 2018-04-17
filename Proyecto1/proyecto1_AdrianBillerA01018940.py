import os
import numpy as np
import matplotlib.pyplot as plt




#funcion para graficad los datos usando matplotlib
def grafica_datos(x, y, theta):
    prediction = y
    #plottear datos ingresados
    plt.plot(x,y, 'ro')
    #plottea theta
    for i in range(0, len(x)):
        prediction[i] = theta[0] + theta[1]* x[i]

    plt.plot(x,prediction, color='black')
    plt.show()
    pass

#funcion del algoritmo de gradiente descendiente
def gradiente_descendente(x, y, theta, alpha, iteraciones):
    for i in range(0, iteraciones):
        temp_zero = theta[0] - (alpha * calcula_costo(x,y,theta, 0))
        temp_one = theta[1] - (alpha * calcula_costo(x,y,theta, 1))
        theta[0] = temp_zero
        theta[1] = temp_one
    return theta

#funcion de calculo de costo de error, se agrega index para saber si es para theta cero o theta uno
def calcula_costo(x, y, theta, index):
    sum_theta_zero = 0
    sum_theta_one = 0
    entries = len(x)
    #calcular hipotesis para cada valor de x y y
    for k in range(0, entries):
        sum_theta_zero += calcula_hipotesis(x[k], theta) - y[k]
        sum_theta_one += (calcula_hipotesis(x[k], theta) - y[k]) * x[k]
    #haciendo division final de la sumatoria
    sum_theta_zero /= entries
    sum_theta_one /= entries

    #dependiendo del index se regresa el valor de theta ya sea para j=0 o j=1
    if index == 0:
        return sum_theta_zero
    else:
        return sum_theta_one


#funcion que calcula la hipotesis de theta para poder calcular el costo
def calcula_hipotesis(x, theta):
    #calcular hipotesis para un valor determinado de x
    hip = theta[0] + theta[1]*x**3
    return hip


#obteniendo datos del archivo
data = np.genfromtxt("ex1data1.txt", delimiter = ",")
#dividiendo datos en dos vectores x , y
splits = np.hsplit(data, 2)
x = splits[0]
y = splits[1]

#razon de aprendizaje alpha
alpha = 0.01
#iteraciones para el gradiente descendiende
iteraciones = 1500
#valor de theta inicial
theta = [0,0]
theta = gradiente_descendente(x,y,theta,alpha,iteraciones)
print ("Vector theta final: ", theta)
grafica_datos(x,y,theta)
