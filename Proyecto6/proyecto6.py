import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, ylim
'''Programa realizado por Adrian Biller A01018940
Proyecto 6
Aprendizaje automatico
Octubre 30'''
#funcion que obtiene el costo promedio por prediccion hecha en a
def get_cost(a, y):
    m, n = y.shape
    J = 0
    for i in range(m):
        for j in range(n):
            first_part = -1*y[i][j]*np.log(a[i][j])
            second_part = (1-y[i][j])*np.log(1 - a[i][j])
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


'''funcion de entrena red neuronal, recibe el numero de inputs que recibe,
hidden_layer_size, numero de neuronas en la capa intermedia
num_labels, numero de salidas de la red neuronal
valores x y y de los ejemplos
utiliza backpropagation para poder predecir los resultados'''
def entrenaRN(input_layer_size, hidden_layer_size, num_labels, x, y):
    m = x.shape[0]
    alpha = 3
    #inicializacion de w1, w2, b1, b2 con numeros aleatorios definidos por una epsilon
    w1 = randInicializaPesos(hidden_layer_size, input_layer_size)
    w2 = randInicializaPesos(num_labels, hidden_layer_size)
    b1 = initialize_bias(hidden_layer_size)
    b2 = initialize_bias(num_labels)
    errors = []
    #numero de iteraciones para el entrenamiento
    iterations = 5000
    for i in range(iterations):#error < 0.28):

        #feedforward
        z1 = np.dot(x, w1) + b1
        a1 = sigmoidal(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoidal(z2)
        J = get_cost(a2, y)
        print(J)

        #backpropagation
        dz2 = a2-y #5000x10
        dw2 = (1/m) * a1.transpose().dot(dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        #dz1 = w2.dot(dz2.transpose()).transpose().dot(sigmoidalGradiente(z1).transpose())
        dz1 = np.multiply((w2.dot(dz2.transpose())), sigmoidalGradiente(z1).transpose())
        #dz1 = np.multiply(w2.transpose() * dz2, sigmoidalGradiente(x))
        dw1 = (1/m) * dz1.dot(x)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
        # dw = np.asarray(((1/m) * x.transpose().dot(dz)).transpose()).reshape(-1)
        # db = (1/m) * np.sum(dz)
        #actualizacion de pesos y bias
        db1temp = np.asarray(db1).reshape(-1)
        db2temp = np.asarray(db2).reshape(-1)
        w1 -= alpha * dw1.transpose()
        b1 -= alpha * db1temp
        w2 -= alpha * dw2
        b2 -= alpha * db2temp
        errors.append(J)
    #graficar todos los errores obtenidos
    graficar_error(errors)
    return w1, b1, w2, b2


def sigmoidalGradiente(z):######################################################################
    return (sigmoidal(z))*(1- sigmoidal(z))


#funcion de activacion sigmoidal
def sigmoidal(z):###########################################################################
    return 1.0 / (1.0 + (np.exp(-z)))

#inicializacion de los valores de pesos dependiendo del numero de neuronas y capas
def randInicializaPesos(L_in, L_out): ################################################################
    epsilon = 0.12
    weights = np.empty([L_out, L_in])
    for i in range(L_out):
        for j in range(L_in):
            weights[i, j] = random.uniform(-epsilon, epsilon)
    return weights

#inicializacion de los vectores bias
def initialize_bias(size):#############################################################################
    bias = np.zeros(size)
    epsilon = 0.12
    for i in range(size):
        bias[i] = random.uniform(-epsilon, epsilon)
    return bias



#funcion que traduce el arreglo de cada registro de a2 a la posicion del numero mas grande del arreglo
def translate_a(a):
    #numero maximo dentro de cada arreglo que define su clase
    max_class = 0
    #index del numero maximo
    max_index = 100
    for i in range(a.shape[0]):
        if(max_class < a[i]):
            max_class = a[i]
            max_index = i

    if(i == 10):
        translation = 0
    else:
        translation = i
    return translation

#funcion de prediccion dependiendo de la activacion
#se incluyen los valores x, los pesos, la b y la funcion de activacion
def prediceRNYaEntrenada(x, w1, b1, w2, b2):
    #feedforward
    z1 = np.dot(x, w1) + b1
    a1 = sigmoidal(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoidal(z2)
    #resultado a comparar
    prediction = np.zeros(a2.shape[0])
    #creando prediccion a base de los 1 prendidos en cada registro de a2
    for i in range(a2.shape[0]):
        prediction[i] = translate_a(a2[i])
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
        #one_column = np.ones((len(x),1))
        #x = np.append(one_column,x,axis=1)
        finalY = np.zeros(shape=(5000, 10))
        for i in range(len(y)):
            temp = np.zeros(10)
            if (y[i] ==10):
                y[i]=0
            index = np.int(y[i])
            temp[index] = 1
            finalY[i] = temp
    return x,finalY



def load_trained_data():
    w1 = np.genfromtxt("w1.txt", delimiter = ",")
    w2 = np.genfromtxt("w2.txt", delimiter = ",")
    b1 = np.genfromtxt("b1.txt", delimiter = ",")
    b2 = np.genfromtxt("b2.txt", delimiter = ",")
    return w1,w2,b1,b2


# def check_error_percentage(y, prediction):
#     aproximate = 0
#     m = y.shape[0]
#     translated_y = np.zeros(m)
#     for i in range(m):
#         translated_y[i] = translate_a(y[i])
#
#     for i in range(m):
#         print("y",translated_y[i])
#         print("pred",prediction[i])
#         if(translated_y[i] == prediction[i]):
#             aproximate+=1
#     percentage = (m - aproximate)/ m
#     return percentage


if __name__ == '__main__':
    print("Proyecto 6 Adrian Biller A01018940")
    #cargar datos del archivo
    x,y = load_data("digitos.txt")
    w1, b1, w2, b2 = entrenaRN(x.shape[1], 25, 10, x, y)
    #w1, w2, b1, b2 = load_trained_data()
    prediction = np.array(prediceRNYaEntrenada(x, w1, b1, w2, b2))
    #print("Error", check_error_percentage(y, prediction))
    #guardando pesos y bias en archivos.txt
    np.savetxt("w1.txt", w1, delimiter=",")
    np.savetxt("w2.txt", w2, delimiter=",")
    np.savetxt("b1.txt", b1, delimiter=",")
    np.savetxt("b2.txt", b2, delimiter=",")
