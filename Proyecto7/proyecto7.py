import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, ylim
'''Programa realizado por Adrian Biller A01018940
Proyecto 7
Aprendizaje automatico
Noviembre 8'''


#Esta funcion va encontrando los centroides mas cerca basado en los primeros encontrados de manera aleatoria
def findClosestCentroids(x, initial_centroids):
    #creando una lista de listas del tamaño de los centroides para poder agregar la lista de centroides cercanos
    idx = [[] for z in range(len(initial_centroids))]
    for i in range(len(x)):
        #creando float infinito para poder comprarrlo con los centroides
        minimum_distance =  float('inf')
        #definiendo un espacio de 4 centroides cercanos
        centroid_temp_index = 0
        for k in range(len(initial_centroids)):
            #obteniendo la distancia del centroide y el punto
            distance = np.linalg.norm(x[i]-initial_centroids[k])
            #comparar la distancia, si es menor entonces se escoge para agregarla a idx
            if (distance < minimum_distance):
                minimum_distance = distance
                centroid_temp_index = k
        idx[centroid_temp_index].append(i)
    return idx

''' basado en la funcion anterior para encontrar los centroides mas cercanos
esta funcion utilizando el numero de centroides obtenidos anteriormente le asigna la media al nuevo centroide'''
def computeCentroids(x, idx, k):
    new_centroids = [[] for j in range(k)]
    for i in range(len(idx)):
        #sacando el numero de elementos por clusters
        m = len(idx[i])
        #incializando xy de cada cluster para sacar la media
        x_mean, y_mean = 0, 0
        for j in range(m):
            x_mean+=x[idx[i][j], 0]
            y_mean+=x[idx[i][j], 1]

        new_centroids[i].append(x_mean/m)
        new_centroids[i].append(y_mean/m)
    return new_centroids


#funcion que implementa el algoritmo de kmeans utilizando las funciones de inicializacion de centroides
#obtencion de centroides cerca y su computo
def runkMeans(x,initial_centroids,iters, graficar=False):
    k = len(initial_centroids)
    #inicializacion de centroides
    centroids = initial_centroids
    #lista de centroides para graficar en caso de que se requiera
    centroid_list = []
    #iteraciones implementando el algoritmo
    for i in range(iters):
        idx = findClosestCentroids(x, centroids)
        centroids = computeCentroids(x, idx, k)
        centroid_list.append(centroids)
    if graficar:
        plot_clusters(x, np.array(centroid_list))
    return centroids



#funcion que aleatoriamente obtiene del vector x k numero de centroides utilizando permutaciones
def kMeansInitCentroids(x, k):
    #haciendo permutaciones aleatorias para obtener los centroides de posiciones aleatorias
    random_indexes = np.random.permutation(x.shape[0])
    centroids = np.zeros(shape = [k, 2])
    for i in range(k):
        centroids[i] = x[random_indexes[i]]
    return centroids



'''funcion obtenida de internet con la que se puede facilmente
graficar los datos junto con la lista de centroides cercanos'''
def plot_clusters(x,centroid_list):
    #Numero de centroides
    k = len(centroid_list[0])
    #Numro de Iteraciones
    h = len(centroid_list)
    #Garfica de puntos en este caso los puntos en necgro
    for centroid in x:
        plt.scatter(*centroid, color="red")
    #Linea de historial de centroides
    for i in range(k):
        plt.plot(*np.array([centroid_list[j][i] for j in range(h)]).T ,color="green" )
    #Poner como identificador una X para el centoride basandode en el historial
    for cl in centroid_list:
        for centroid in cl:
            plt.scatter(*centroid,marker="X",s=100,color="blue")
    plt.show()

if __name__ == '__main__':
    print("Proyecto 7 Adrian Biller A01018940")
    x = np.genfromtxt("ex7data2.txt", delimiter = " ")
    #llamando la inicializacion de centroides con 3 clusters y el vector x
    initial_centroids = kMeansInitCentroids(x,3)
    runkMeans(x, initial_centroids, 30, True)
    # closest_centroids = findClosestCentroids(x,centroids)
    # print(closest_centroids)
    # new_centroids = computeCentroids(x,closest_centroids,3)
    # print(new_centroids)
