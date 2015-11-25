# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy import array
import os
from functions.get_features_local import get_features



import sklearn.metrics
from scipy.cluster.vq import vq, kmeans, whiten
#Teoria clusters: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

def train_codebook(descriptores):
    whitened = whiten(descriptores)
    book = array((whitened[0],whitened[2]))
    kmeans(whitened,book)

    return kmeans

des = get_features("../../archivosPrueba/IMG_7057.jpg")
