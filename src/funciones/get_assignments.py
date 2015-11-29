from funciones import *
from scipy.cluster.vq import vq, whiten
import matplotlib.pyplot as plt


def get_assignments(codebook, descriptores):
    norm_descriptores = whiten(descriptores) # Normaliza descriptores
    assignments,_ = vq(descriptores, codebook) # genera el vector de assigments

    return assignments


if __name__== "__main__":

    descriptor1 = get_local_features("../TerrassaBuildings900/train/images/4406-18633-1754.jpg")
    codebook = train_codebook(5, descriptor1)
    descriptor2= get_local_features("../TerrassaBuildings900/val/images/22053-9694-8921.jpg")
    assig = get_assignments(codebook, descriptor2)

    print(assig)
    print "Longuitud del assignments= " + str(len(assig))
    
    plt.scatter(descriptor2[:,0], descriptor2[:,1]), plt.scatter(codebook[:,0], codebook[:,1], color='r'), plt.show()



