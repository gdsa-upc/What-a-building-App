import os
import cv2
import scipy

from funciones.train_codebook import train_codebook
from funciones.get_local_features import get_features
from scipy.cluster.vq import vq, kmeans, whiten



def get_assignments(codebook, descriptores):

    norm_descriptores = whiten(descriptores) #Normaliza descriptores

    code,_ = vq(norm_descriptores, codebook) # genera el vector de assigments

    code.sort() #ordena el vector code

    return code



codebook1 = train_codebook(5, get_features("../TerrassaBuildings900/train/images/4406-18633-1754.jpg"))
descriptoresss = get_features("../TerrassaBuildings900/train/images/4406-18633-1754.jpg")
assig = get_assignments(codebook1, descriptoresss)
print(assig) #Crea un vector ordenado con los descriptores que equivalen a cada region (k=5)

