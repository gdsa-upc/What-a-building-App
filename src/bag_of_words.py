from funciones.get_assignments import get_assignments
import matplotlib.pyplot as plt
from funciones.train_codebook import train_codebook
from funciones.get_local_features import get_local_features
from scipy.cluster.vq import vq, kmeans, whiten
import collections
from sklearn import preprocessing

def bag_of_words(assignments):
    counter= collections.Counter(assignments)
    counter_v= counter.values()
    counter_nor= preprocessing.normalize(counter_v)
    #se tiene que devolver counter_nor, pero para pruebas es mejor verlo con counter_v
    return counter_v



"""
descriptoresss = whiten(get_local_features("../TerrassaBuildings900/train/images/4406-18633-1754.jpg"))

codebook1 = train_codebook(5, descriptoresss)
descriptores2= whiten(get_local_features("../TerrassaBuildings900/val/images/22053-9694-8921.jpg"))
assig = get_assignments(codebook1, descriptores2)

#print(assig) #Crea un vector ordenado con los descriptores que equivalen a cada region (k=5)
asdf= bag_of_words(assig)
print asdf
#plt.scatter(descriptores2[:,0], descriptores2[:,1]), plt.scatter(codebook1[:,0], codebook1[:,1], color='r'), plt.show()
"""
