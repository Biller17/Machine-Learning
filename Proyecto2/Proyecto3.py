import os
import numpy as np
import matplotlib.pyplot as plt
import math



def normalizacionDeCaracteristicas(x):
    #valor a regresar x, mu, sigma
    #inicializando sigma mu y la sumatoria de x para futuro uso
    mu = np.zeros(x.shape[1])
    sigma = np.zeros(x.shape[1])
    x_sums = np.sum(x,axis=0)

    #creando for para todas las caracteristicas
    for i in range(0, len(mu)):
        mu[i] = x_sums[i]/len(x)
        for j in range(0,len(x)):
            #elevando la diferencia de cada valor de x al cuadrado
            sigma[i] += (x[j,i] - mu[i])**2
        #sacando raiz cuadrada de sigma de la caracteristica i
        sigma[i] = math.sqrt(sigma[i]/len(x))

    #generando x normalizadas
    for i in range(0, len(x)):
        for j in range(0, x.shape[1]):
            x[i,j] = (x[i,j] - mu[j])/sigma[j]

    return mu, sigma, x


def gradienteDescendenteMultivariable(x,y,theta,alpha,iteraciones):
    #variable temporal para asignacion simultanea
    temp = np.float64(np.zeros(len(theta)))
    m = len(x)
    j_historial = np.zeros(iteraciones)
    #for para iteraciones requeridas
    for k in range(0, iteraciones):
        #for para cada theta que exista
        for i in range(0, len(theta)):
            #for para la sumatoria de cada valor de los datos obtenidos
            for j in range(0,len(x)):
                #sumatoria de la formula"
                temp[i] += (calcula_hipotesis(x, theta, j) - y[j]) * x[j,i]
            #multiplicar por alpha/m al final de la sumatoria
            temp[i] = theta[i] - ((alpha/m) * temp[i])
        #asignar todas las thetas
        theta = temp
        temp = np.zeros(len(theta))
        j_historial[k] = calculaCosto(x,y,theta)
    return theta, j_historial



#funcion que calcula la hipotesis de theta para poder calcular el costo
def calcula_hipotesis(x, theta, index):
    #calcular hipotesis para un valor determinado de x
    hip = np.float64(0.0)
    for i in range(1 , len(theta)+1):
        hip += theta[i-1]* x[index,i-1]**i

    #print "hipotesis: ", hip
    return hip


def graficaError(j_historial):
    #recibe vextor de historial de errores y lo grafica
    plt.plot(j_historial)
    #muestra historial de errores
    plt.show()




def calculaCosto(x,y,theta):
    cost = 0

    #calcular hipotesis para cada valor de x y y
    for i in range(0,len(x)):
        cost += (calcula_hipotesis(x, theta, i) - y[i])**2
    #haciendo division final de la sumatoria

    cost /= 2 * len(x)

    return cost



def ecuacionNormal(x,y):
    #creando x transpuesta
    transposedX = x.T
    #obteniendo theta por medio de la ecuacion normal
    theta  = np.linalg.inv(transposedX.dot(x)).dot(transposedX.dot(y))
    print ("Valor theta utilizando ecuacion normal: ", theta)
    return theta

def predicePrecio(x,theta):
    #recibe datos de un solo dato de x y obtiene su precio final
    res = 0
    for i in range(0,len(theta)):
        res += theta[i]*x[i]
    return res



#obteniendo datos del archivo
data = np.genfromtxt("ex1data2.txt", delimiter = ",")
#dividiendo datos en dos vectores los valores de x y y
splits = np.hsplit(data, np.array([2,6]))
x = np.array(splits[0], dtype='float64')
y = splits[1]

mu, sigma, x_normalized = normalizacionDeCaracteristicas(x)
#concatenando arreglo de unos a matriz de variables x
one_column = np.ones((len(x),1))
x = np.append(one_column,x,axis=1)
x_normalized = np.append(one_column,x,axis=1)

old_settings = np.seterr(all='warn', over='raise')
#old_settings = np.seterr(all='print', over='raise')

#razon de aprendizaje alpha
alpha = 0.3
#iteraciones para el gradiente descendiende
iteraciones = 100

#valor de theta inicial de ceros dependiendo el numero de variables x
theta = np.zeros(x.shape[1])
print ("Costo inicial: ", calculaCosto(x,y,theta))
theta, j_historial = gradienteDescendenteMultivariable(x,y,theta,alpha,iteraciones)
theta_normalized, temp = gradienteDescendenteMultivariable(x_normalized,y,theta,alpha,iteraciones)
print ("Vector theta final: ", theta)
print ("Vector theta final con datos normalizados: ", theta_normalized)
print ("Costo final: ", calculaCosto(x,y,theta))
ecuacionNormal(x,y)

graficaError(j_historial)
