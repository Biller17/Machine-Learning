import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, ylim
'''Programa realizado por Adrian Biller A01018940
Proyecto 5
Aprendizaje automatico
Octubre 11'''
#funconi que obtiene el costo dependiendo de la funcion de activacion
def get_costo(activacion, x, y ,b, w):
    error = 0
    m = x.shape[0]
    prediction = np.array(prediceRNYaEntrada(x, w,b,activacion)).transpose()
    print(prediction)
    if activacion == "lineal":
        for i in range(m):
            error+= (y[i] - prediction[i])**2
        error/=m
        return error

    else:
        for i in range(m):
            primero = (-y[i] * np.log(prediction[i]))
            #caso donde y es 1
            segundo = (1 - y[i])* (np.log(1 - prediction[i]))
            #sintesis de funcion
            error += primero - segundo
        error/=m
        return error

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


#neurona de backpropagation
def bpnUnaNeurona(w, input_layer_size, x, y, alpha, activacion):
    m = x.shape[0]
    b = randInicializaPesos(x.shape[1])[0]
    errors = []
    dw = np.zeros(x.shape[1])
    for i in range(1000):
        z = x.dot(w) + b
        #se obtiene a de la funcion dependiendo de la funcion de activacion
        a = function_A(z, activacion)
        #se obtiene dz por medio de la funcion dependiendo de la activacin
        dz = function_dz(a.T,y,activacion)
        dw = np.asarray(((1/m) * x.transpose().dot(dz)).transpose()).reshape(-1)
        db = (1/m) * np.sum(dz)
        w -= alpha * dw
        b -= alpha * db
        errors.append(get_costo(activacion, x, y ,b, w))
    return b, w, errors


#funcion que regresa a dependiendo de activacion
def function_A(z, activacion):
    if activacion == "lineal":
        return z
    else:
        return sigmoidal(z)

#funcoin que obtiene dz dpeendiendo de la activacion
def function_dz(a, y, activacion):
    dz = a-y
    return dz




def sigmoidGradiente(z):
    return (sigmoidal(z))*(1- sigmoidal(z))

def linealGradiente(z):
    return 1


#funcion de activacion sigmoidal
def sigmoidal(z):
    return 1.0 / (1.0 + (np.exp(-z)))

#inicializacion aleatoria de los pesos dependiendo de la longitud
def randInicializaPesos(L_in):
    weights = np.zeros(L_in)

    epsilon = 0.12
    for i in range(L_in):
        weights[i] = random.uniform(-epsilon, epsilon)
    return weights


#funcion de prediccion dependiendo de la activacion
#se incluyen los valores x, los pesos, la b y la funcion de activacion
def prediceRNYaEntrada(x, w, b, activacion):
    prediction = np.array(x.shape[0])
    n = 0
    if activacion == "lineal":
        prediction = np.asarray(x.dot(w.transpose()) + b).reshape(-1)
    else:
        prediction = np.asarray(x.dot(w.transpose()) + b).reshape(-1)
        for i in range(prediction.shape[0]):
            prediction[i] = sigmoidal(prediction[i])
    return prediction

#normalizacion de datos
def normalizar_datos(old_x):
    media = old_x.mean()
    #obtener sigma (desviacion estandar)
    Sig = old_x.std()
    #Normalizacion de vector x
    new_x = np.vectorize( lambda xi: (xi-media)/Sig)
    #regresar el valor como vector
    return new_x(old_x)


if __name__ == '__main__':
    print("Proyecto 5 Adrian Biller A01018940")
    activacion = "lineal"

    if activacion == "lineal":
        data = np.genfromtxt("casas.txt", delimiter = ",")
        #dividiendo datos en dos vectores los valores de x y y
        splits = np.hsplit(data, np.array([2,6]))
        x = np.array(splits[0])
        y = np.array(splits[1])
        #normalizacion de datos
        x = np.apply_along_axis(normalizar_datos, 0, x)
        #x_graficar = np.apply_along_axis(normalizar_datos, 0, splits[0])
        #agregando unos a matriz de x
        #one_column = np.ones((len(x),1))
        #x = np.append(one_column,x,axis=1)
        weights = randInicializaPesos(x.shape[1])
        b, weights, errors = bpnUnaNeurona(weights, x.shape[1],x,y,0.1, activacion)
        graficar_error(errors)
    else:
        x = np.matrix('0,0;0,1;1,0;1,1')
        #y_and = np.matrix('0;0;0;1')
        y_or = np.matrix('0;1;1;1')
        weights = randInicializaPesos(x.shape[1])
        b, weights, errors = bpnUnaNeurona(weights, x.shape[1],x,y_or,0.1, activacion)
        graficar_error(errors)
