import os
import numpy as np
import matplotlib.pyplot as plt
import math
from pylab import scatter, show, legend, xlabel, ylabel


'''Proyecto 3
Adrian Biller A01018940
4 de septiempre de 2017'''

######################################################################################################################################################
#grafica datos con x y o para saber los aprobados y reprobados
def graficaDatos(x,y,theta):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    #scatter de los puntos en los que los alumnos aprobaron
    scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
    #scatter de los pintos en los que los alumnos reprobaron
    scatter(x[neg, 0], x[neg, 1], marker='x', c='r')



    #graficar theta
    maxmin = [np.amin(x,axis=0)[0], np.amax(x,axis=0)[0]]
    #formula para generar valor para poder graficar theta como recta
    f = lambda x1,th : (0.5-th[0]-th[1]*x1)/th[2]
    #evaluando cada x con theta y graficandolo
    plt.plot(maxmin,[f(i,theta) for i in maxmin], color = 'black')



    #escribiendo labels para la grafica
    xlabel('Calificacion 1')
    ylabel('Calificacion 2')
    #mostrar datos de las legendas de x y o para reprbados y aprobados
    legend(['Aprobado', 'Reprobado'])
    show()
    pass

#realiza gradiente descendente
def aprende(theta,x,y,iteraciones):
    #variable temporal para asignacion simultanea
    temp = np.float64(np.zeros(len(theta)))
    m = len(x)
    #declarando razon de aprendizaje
    alpha = 3
    #for para iteraciones requeridas
    for k in range(0, iteraciones):
        #for para cada theta que exista
        for i in range(0, len(theta)):
            #for para la sumatoria de cada valor de los datos obtenidos
            for j in range(0,len(x)):
                #sumatoria de la formula"
                temp[i] += ((   (sigmoidal(x[j].dot(theta.T)))   - y[j]    ) * (x[j,i]))
            #multiplicar por alpha/m al final de la sumatoria
            temp[i] = theta[i] - ((alpha/m) * temp[i])
        #asignar todas las thetas
        theta = temp
        temp = np.zeros(len(theta))
    return theta

def predice(theta,x):
    p = np.zeros(len(x))
    for i in range(0, len(x)):
        if sigmoidal(x[i].dot(theta)) >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p

#funcion sigmoidal la cual recibe una z con respecto a theta y x y regresa la probabilidad
def sigmoidal(z):
    return 1.0 / (1.0 + (np.exp(-z)))

#funcion calcula costo y gradiente de costo
def funcionCosto(theta,x,y):
    j = np.float64(0.0)
    grad = np.zeros(x.shape[1])
    theta = np.reshape(theta,(len(theta),1))
    #caso donde y es 0
    primero = -y.T.dot(   np.log(   sigmoidal(x.dot(theta))   )   )
    #caso donde y es 1
    segundo = (1 - y.T).dot(np.log(1 - sigmoidal(x.dot(theta))))
    #sintesis de funcion
    j = primero - segundo
    #obtencin de gradiente de costo
    grad = ((  sigmoidal(x.dot(theta)) - y ).T.dot(x))
    #haciendo division final de la sumatoria
    j /= len(x)
    return j, grad

def normalizar_datos(old_x):
    media = old_x.mean()
    #obtener sigma (desviacion estandar)
    Sig = old_x.std()
    #Normalizacion de vector x
    new_x = np.vectorize( lambda xi: (xi-media)/Sig)
    #regresar el valor como vector
    return new_x(old_x)

#obteniendo datos del archivo
data = np.genfromtxt("ex2data1.txt", delimiter = ",")
#dividiendo datos en dos vectores los valores de x y y
splits = np.hsplit(data, np.array([2,6]))
x = np.array(splits[0])
y = splits[1]

#normalizacion de datos
x = np.apply_along_axis(normalizar_datos, 0, x)
x_graficar = np.apply_along_axis(normalizar_datos, 0, splits[0])
#agregando unos a matriz de x
one_column = np.ones((len(x),1))
x = np.append(one_column,x,axis=1)

print(x)
#iteraciones para el gradiente descendiende
iteraciones = 100

#valor de theta inicial de ceros dependiendo el numero de variables x
theta = np.zeros(x.shape[1])


costo, gradienteCosto = funcionCosto(theta,x,y)
print ("Costo inicial: ", costo)
print ("Gradiente del costo", gradienteCosto)
theta = aprende(theta,x,y,iteraciones)
print ("Theta: ", theta)
costo, gradienteCosto = funcionCosto(theta,x,y)
print ("Costo final: ", costo)
print("Predicciones de alumnos", predice(theta, x))
graficaDatos(x_graficar,y,theta)
