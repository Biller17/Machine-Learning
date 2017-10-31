import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, ylim
'''Programa realizado por Adrian Biller A01018940
Proyecto 6
Aprendizaje automatico
Octubre 30'''
#funconi que obtiene el costo dependiendo de la funcion de activacion
def get_cost(a, y):
    m = y.shape[0]
    J = 0
    for i in range(m):
        first_part = -1*y[i]*np.log(a[i])
        second_part = (1-y[i])*np.log(1 - a[i])
        J+=first_part - second_part
    J/=m
    return J

#funcion que recibe los errores y crea la grafica
def graficar_error(errors):
    #pasando la lista de errores a otra debido al formato en el que esta
    plot_errors = np.zeros(len(errors))
    for i in range(len(errors)):
        plot_errors[i] = errors[i]
    #graficar error
    plt.plot(plot_errors)
    plt.ylabel('Error')
    plt.show()



def entrenaRN(input_layer_size, hidden_layer_size, num_labels, x, y):
    m = x.shape[0]
    alpha = 0.03
    w1 = randInicializaPesos(hidden_layer_size, input_layer_size)
    w2 = randInicializaPesos(num_labels, hidden_layer_size)
    b1 = initialize_bias(hidden_layer_size)
    b2 = initialize_bias(num_labels)
    errors = []
    iterations = 5000
    for i in range(iterations):#error < 0.28):

        #feedforward
        z1 = np.dot(x, w1) + b1
        #se obtiene a de la funcion dependiendo de la funcion de activacion
        a1 = sigmoidal(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoidal(z2)
        J = get_cost(a2, y)
        #backpropagation
        dz2 = a2-y
        dw2 = 1/m * dz2.transpose().dot(a1)
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)

        dz1 = w2.dot(dz2.transpose()).dot(sigmoidal(z1))
        #dz1 = np.multiply(w2.transpose() * dz2, sigmoidalGradiente(x))
        dw1 = 1/m * dz1.dot(x)
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)
        # dw = np.asarray(((1/m) * x.transpose().dot(dz)).transpose()).reshape(-1)
        # db = (1/m) * np.sum(dz)
        #actualizacion de pesos y bias
        #dz1 = np.multiplpy(w2.t * dz2, sigmoidalGradiente(a1))
        print("w1", w1.shape)
        print("dw1", dw1.shape)
        print("w2", w2.shape)
        print("dw2", dw2.shape)
        w1 -= alpha * dw1
        b1 -= alpha * db1
        w2 -= alpha * dw2
        b2 -= alpha * db2
        #errors.append(J)
    return w1, b1, w2, b2


def sigmoidalGradiente(z):######################################################################
    return (sigmoidal(z))*(1- sigmoidal(z))


#funcion de activacion sigmoidal
def sigmoidal(z):###########################################################################
    return 1.0 / (1.0 + (np.exp(-z)))

#inicializacion de los valores de pesos dependiendo del numero de neuronas y capas
def randInicializaPesos(L_in, L_out):                               ##############################################
    epsilon = 0.12
    weights = np.empty([L_out, L_in])
    for i in range(L_out):
        for j in range(L_in):
            weights[i, j] = random.uniform(-epsilon, epsilon)
    return weights


def initialize_bias(size):#############################################################################
    bias = np.zeros(size)
    epsilon = 0.12
    for i in range(size):
        bias[i] = random.uniform(-epsilon, epsilon)
    return bias


#funcion de prediccion dependiendo de la activacion
#se incluyen los valores x, los pesos, la b y la funcion de activacion
def prediceRNYaEntrenada(x, w1, b1, w2, b2):
    prediction = np.array(x.shape[0])
    return prediction


#funcion que lee el archivo y obtiene las entradas en x y y
def load_data(filename):#############################################################################################
    #x = np.array()
    #y = np.array()
    #prueba con cero.txt
    if (filename == "cero.txt"):
        data = np.genfromtxt(filename, delimiter = " ")
        data = np.asarray(data).reshape(-1)
        x = data
        y = np.array(10)
        one_column = np.ones((len(x),1))
        x = np.append(one_column,x,axis=1)

    elif(filename == "digitos.txt"):
        #prueba con digitos.txt
        data = np.genfromtxt(filename, delimiter = " ")
        y = data[:, 400]
        x = np.delete(data,400, 1)
        #agregando unos a matriz de xx
        one_column = np.ones((len(x),1))
        x = np.append(one_column,x,axis=1)
        finalY = np.zeros(shape=(5000, 10))
        for i in range(len(y)):
            temp = np.zeros(10)
            if (y[i] ==10):
                y[i]=0
            index = np.int(y[i])
            temp[index] = 1
            finalY[i] = temp
    return x,finalY

if __name__ == '__main__':
    print("Proyecto 6 Adrian Biller A01018940")
    x,y = load_data("digitos.txt")
    w1, b1, w2, b2 = entrenaRN(x.shape[1], 25, 10, x, y)
    trained_data = np.array(w1, b1, w2, b2)
