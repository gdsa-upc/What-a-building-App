# -*- coding: utf-8 -*-
from funciones.get_assignments import get_assignments
import matplotlib.pyplot as plt
from funciones.train_codebook import train_codebook
from funciones.get_local_features import get_local_features
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
#from sklearn import preprocessing
#from array import *

#Esta funcion devuelve un vector de tamaño n_clusters con el histograma de cada region.

def build_bow(assignments, n_clusters):

    #assig_orden = assigments.sort()

    bow = np.zeros((n_clusters,))  # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.zeros.html
                                    #http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.zeros_like.html#numpy.zeros_like

    #for i in range(n_clusters-1):
       # bow[i]=0.0

    for n_assig in assignments:
        bow[n_assig]+=1

    #counter_v= np.bincount(assignments)
    #counter_nor= preprocessing.normalize(counter_v)


    #se tiene que devolver counter_nor, pero para pruebas es mejor verlo con counter_v y además suelta warnings
    #return counter_nor
    #bow_norm = whiten(bow)

    return bow


descriptoresss = whiten(get_local_features("../TerrassaBuildings900/train/images/4406-18633-1754.jpg"))

codebook1 = train_codebook(50, descriptoresss)
descriptores2= whiten(get_local_features("../TerrassaBuildings900/val/images/22053-9694-8921.jpg"))
assig = get_assignments(codebook1, descriptores2)

#print(assig) #Crea un vector ordenado con los descriptores que equivalen a cada region (k=5)
asdf= build_bow(assig,50)
print asdf
print ("Numero de regiones diferentes: " + str(len(asdf))) # tiene que ser igual a
#plt.scatter(descriptores2[:,0], descriptores2[:,1]), plt.scatter(codebook1[:,0], codebook1[:,1], color='r'), plt.show()
