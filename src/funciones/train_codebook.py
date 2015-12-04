# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy import array
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from funciones import *
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

    #Con KMeans
    #centroides,_= kmeans(descriptores, numClusters)

    #Con MiniBatchKMeans
    centroides= MiniBatchKMeans(numClusters)
    centroides.fit(descriptores)

    return centroides # Devuelve el vector codebook


if __name__== "__main__":
    descriptoresss = get_local_features("../TerrassaBuildings900/train/images/4406-18633-1754.jpg")
    codebook1 = train_codebook(1, descriptoresss)
#    print codebook1 
#    print(codebook1.size)
#    print(codebook1.shape)
#    print(codebook1[:])
