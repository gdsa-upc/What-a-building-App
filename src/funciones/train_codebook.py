# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy import array
import os

from funciones.get_local_features import get_features


import sklearn.metrics
import scipy
from scipy.cluster.vq import vq, kmeans, whiten

#Teoria clusters: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#Cluster con kmeans: http://scikit-learn.org/stable/modules/clustering.html#k-means
#mas: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html


#Esta funcion debe devolver los centroides o puntos de enteres.

#whiten(obs[, check_finite]) 	Normalize a group of observations on a per feature basis.
#vq(obs, code_book[, check_finite]) 	Assign codes from a code book to observations.
#kmeans(obs, k_or_guess[, iter, thresh, ...]) 	Performs k-means on a set of observation vectors forming k clusters.
#kmeans2(data, k[, iter, thresh, minit, ...]) 	Classify a set of observations into k clusters using the k-means algorithm.


def train_codebook(numClusters, descriptores): #Només para las imagenes de train

        # codebook={}
    norm_descriptores = whiten(descriptores, check_finite=True) #Normaliza descriptores
        # assign_code = scipy.cluster.vq.vq(descriptores, codebook, check_finite=True)

        #book = array((norm_descriptores[0], norm_descriptores[1]))
    codebook,_ = kmeans(norm_descriptores, numClusters, iter=20, thresh=1e-05, check_finite=True)


        #con  libreria cv2 pruebas de otros posibles valores
            #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            #b,a,codebook = cv2.kmeans(norm_descriptores, numClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return codebook; # Devuelve el vector codebook

    #return norm_descriptores
    #return norm_descriptores

#Pruebas de la función, pruebas hechas desde la carpeta /src, cuidado que al estar la funcion en el directorio funciones, se suele cambiar solo y se lia parda.

descriptoresss = get_features("../TerrassaBuildings900/train/images/4406-18633-1754.jpg")
codebook1 = train_codebook(1, descriptoresss)


print(codebook1.size)
print(codebook1.shape)
print(codebook1[:])
