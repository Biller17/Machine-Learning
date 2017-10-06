#Proyecto 4
#Perceptron y Adaline
#Adrian Biller A01018940
#from pylab import scatter, show, legend, xlabel, ylabel
import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, ylim


def graficar_error(errors):
    #pasando la lista de errores a otra debido al formato en el que esta
    plot_errors = np.zeros(len(errors))
    for i in range(len(errors)):
        plot_errors[i] = errors[i]
    #graficar error
    plt.plot(plot_errors)
    plt.ylabel('Error')
    plt.show()

def sigmoidal(z):
    return 1.0 / (1.0 + (np.exp(-z)))

def funcionCostoPerceptron(theta,x,y):
    y_obtenida = predicePerceptron(theta,x)
    costo = 0
    n = x.shape[0]
    for i in range(x.shape[0]):
        costo+= y[i] - y_obtenida[i]
    costo/=n
    grad = ((  sigmoidal(x.dot(theta)) - y ).T.dot(x))
    return costo, grad


def step_function(n):
    if n >= 0.5:
        return 1
    else:
        return 0


def entrenaPerceptron(x,y,theta):
    #funcion de escalon para definir el resultado
    #arreglo de errores
    errors = []
    #constante alpha
    alpha = 0.3
    n = theta.shape[0]
    error = 1
    for z in range(150):
        for i in range(n):
            result = 0
            for k in range(theta.shape[0]):
                #obteniendo resultado utilizando los pesos
                result += theta[k] * x[i,k]
            #obteniendo el error con base a el resultado obtenido y el resultado esperado
            error = y[i] - step_function(result)
            errors.append(error)
            for k in range(theta.shape[0]):
                theta[k] += alpha * error * x[i,k]

    graficar_error(errors)
    print("Costo:", funcionCostoPerceptron(theta,x,y))
    return theta

def predicePerceptron(theta,x):

    p = np.zeros(x.shape[0])
    n = 0
    for i in range(x.shape[0]):
        for k in range(theta.shape[0]):
            n += theta[k] * x[i,k]

        p[i] = step_function(n)
        n = 0
    return p

def funcionCostoAdaline(theta, x, y):
    y_obtenida = predicePerceptron(theta,x)
    costo = 0
    n = x.shape[0]
    for i in range(x.shape[0]):
        costo+= y[i] - y_obtenida[i]
    costo/=n
    grad = ((  sigmoidal(x.dot(theta)) - y ).T.dot(x))
    return costo, grad

def entrenaAdaline(x,y,theta):
    #funcion de escalon para definir el resultado

    #arreglo de errores
    errors = []
    #constante alpha
    alpha = 0.7
    n = theta.shape[0]
    error = 1
    for z in range(150):
        for i in range(n):
            result = 0
            for k in range(theta.shape[0]):
                #obteniendo resultado utilizando los pesos
                result += theta[k] * x[i,k]
            #obteniendo el error con base a el resultado obtenido y la funcion net para que sea adaline
            error = (y[i] - result)
            errors.append(error)
            for k in range(theta.shape[0]):
                theta[k] += alpha * error * x[i,k]

    graficar_error(errors)
    print("Costo:", funcionCostoAdaline(theta,x,y))
    return theta

def prediceAdaline(theta, x):
    p = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for k in range(theta.shape[0]):
            p[i] += theta[k] * x[i,k]
            p[i] = step_function(p[i])
    return p





if __name__ == '__main__':
    #creando matriz de x como input
    x = np.matrix('0,0,1;0,1,1;1,0,1;1,1,1')
    y_and = np.matrix('0;0;0;1')
    y_or = np.matrix('0;1;1;1')
    theta = np.zeros(x.shape[1])

    #inicializando theta con numeros aleatorios entre 0 y 1
    for i in range(theta.shape[0]):
        theta[i] = random.uniform(0, 1)

    print(entrenaPerceptron(x,y_and,theta))
    print(entrenaAdaline(x,y_and,theta))
