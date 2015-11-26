import os
import cv2
import sklearn.metrics
import scipy

from funciones.train_codebook import train_codebook
from funciones.get_local_features import get_features


def get_assignments(codebook, descriptores):

    norm_descriptores = scipy.cluster.vq.whiten(descriptores, check_finite=True) #Normaliza descriptores

    code,_ = scipy.cluster.vq.vq(norm_descriptores, codebook)

    #He hecho pruebas por cada nueva linia que le metia a la funcion, aqui me he quedao.


    return code



#codebook1 = train_codebook(2, get_features("../TerrassaBuildings900/train/images/4406-18633-1754.jpg"))
#descriptoresss = get_features("../TerrassaBuildings900/train/images/4406-18633-1754.jpg")

#assig = get_assignments(codebook1, descriptoresss)
